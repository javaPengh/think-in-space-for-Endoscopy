import logging
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from qwen_vl_utils import process_vision_info
import torch
import gc
import re
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.blind_eval import is_blind_mode, normalize_visual_input_mode, strip_visual_context
from loguru import logger as eval_logger

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=1024,
    do_sample=False,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@register_model("qwen3vl")
class Qwen3VL(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        modality: str = "video",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        max_frames_num: int = 32,
        max_pixels: int = 602112,
        video_sampling_strategy: str = "uniform",
        video_sample_fps: float = None,
        keyframe_mapping_path: str = "data/keyframe_mapping.json",
        visual_input_mode: str = "visual",
        **kwargs,
    ):
        super().__init__()

        self.visual_input_mode = normalize_visual_input_mode(visual_input_mode)
        self.max_frames_num = int(max_frames_num)
        self.video_sample_fps = None if video_sample_fps in (None, "", "none", "None") else float(video_sample_fps)
        self.video_sampling_strategy = str(video_sampling_strategy).lower()
        if self.video_sample_fps is not None and self.video_sampling_strategy == "uniform":
            self.video_sampling_strategy = "fps"
        if self.video_sampling_strategy in {"fps_uniform", "uniform_fps"}:
            self.video_sampling_strategy = "fps"
        if not is_blind_mode(self.visual_input_mode):
            if self.video_sampling_strategy not in {"uniform", "specific", "fps"}:
                raise ValueError(f"Unsupported video_sampling_strategy for Qwen3VL: {self.video_sampling_strategy}")
            if self.video_sampling_strategy == "fps" and (self.video_sample_fps is None or self.video_sample_fps <= 0):
                raise ValueError("video_sample_fps must be a positive number when video_sampling_strategy='fps'.")

        self.keyframe_mapping = {}
        if self.video_sampling_strategy == "specific" and not is_blind_mode(self.visual_input_mode):
            if not os.path.exists(keyframe_mapping_path):
                raise ValueError(f"Keyframe mapping file not found at {keyframe_mapping_path}. Required when video_sampling_strategy is 'specific'.")
            import json
            with open(keyframe_mapping_path, "r", encoding="utf-8") as f:
                self.keyframe_mapping = json.load(f)

        self.path = pretrained
        self.max_pixels = int(max_pixels) if max_pixels is not None else None

        # Load model
        if device_map == "auto":
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, trust_remote_code=True, device_map=device_map).eval()
        else:
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()

        processor_kwargs = {"trust_remote_code": True}
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self._processor = AutoProcessor.from_pretrained(self.path, **processor_kwargs)

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for Qwen3VL, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        # Accelerator setup
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.modality = modality
        self.sample_frames_version = None
        self.token_usage_path = str(Path(__file__).resolve().parents[2] / "docs" / "qwen3vl_token_usage.jsonl")
        self._last_aggregated_token_usage = None
        self._reset_token_usage()

    @property
    def config(self):
        return self._model.config

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _determine_sample_frames_version(self):
        # 分布式安全地确定本次评估的 sample_frames 时间戳目录 (仅执行一次)
        import os
        import torch.distributed as dist

        if self.sample_frames_version is not None:
            return self.sample_frames_version

        extracted_model_name = self.path.split("/")[-1] if "/" in self.path else self.path
        base_model_dir = os.path.join("sample_frames", f"{extracted_model_name}-{self._sample_frames_run_label()}")

        if self.rank == 0:
            os.makedirs(base_model_dir, exist_ok=True)
            version_str = datetime.now().strftime("%Y%m%d%H%M%S")
            # 提前由主进程创建以避免竞争
            os.makedirs(os.path.join(base_model_dir, version_str), exist_ok=True)
            version_obj = [version_str]
        else:
            version_obj = [None]

        if self.world_size > 1 and dist.is_initialized():
            dist.broadcast_object_list(version_obj, src=0)

        self.sample_frames_version = version_obj[0]
        return self.sample_frames_version

    def _flatten_visuals(self, visual):
        if visual is None or visual == []:
            return []
        if isinstance(visual, list):
            return visual
        return [visual]

    def _infer_media_type(self, media_path):
        suffix = os.path.splitext(str(media_path))[1].lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        return self.modality

    def _get_media_info(self, visual):
        visual_items = self._flatten_visuals(visual)
        if not visual_items:
            return "text", None

        first_visual = visual_items[0]
        if isinstance(first_visual, dict):
            media_path = first_visual.get("path") or first_visual.get("media_path")
            if media_path is None:
                raise ValueError(f"Missing media path in visual input: {first_visual}")
            media_type = str(first_visual.get("media_type") or self._infer_media_type(media_path)).lower()
        else:
            media_path = first_visual
            media_type = self._infer_media_type(media_path)

        if media_type not in {"image", "video"}:
            raise ValueError(f"Unsupported media_type for Qwen3VL: {media_type}")
        return media_type, media_path

    def _reset_token_usage(self):
        self._token_usage = {
            "num_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "by_media": {
                "image": {"num_requests": 0, "input_tokens": 0, "output_tokens": 0},
                "video": {"num_requests": 0, "input_tokens": 0, "output_tokens": 0},
                "text": {"num_requests": 0, "input_tokens": 0, "output_tokens": 0},
            },
        }

    def _record_token_usage(self, media_type, input_tokens, output_tokens):
        media_key = media_type if media_type in self._token_usage["by_media"] else "text"
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)

        self._token_usage["num_requests"] += 1
        self._token_usage["input_tokens"] += input_tokens
        self._token_usage["output_tokens"] += output_tokens

        media_usage = self._token_usage["by_media"][media_key]
        media_usage["num_requests"] += 1
        media_usage["input_tokens"] += input_tokens
        media_usage["output_tokens"] += output_tokens

    def _aggregate_token_usage_across_ranks(self):
        usage = self._token_usage
        values = [
            usage["num_requests"],
            usage["input_tokens"],
            usage["output_tokens"],
            usage["by_media"]["image"]["num_requests"],
            usage["by_media"]["image"]["input_tokens"],
            usage["by_media"]["image"]["output_tokens"],
            usage["by_media"]["video"]["num_requests"],
            usage["by_media"]["video"]["input_tokens"],
            usage["by_media"]["video"]["output_tokens"],
            usage["by_media"]["text"]["num_requests"],
            usage["by_media"]["text"]["input_tokens"],
            usage["by_media"]["text"]["output_tokens"],
        ]

        if self.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            device = self._device if getattr(self, "_device", None) is not None else torch.device("cpu")
            totals = torch.tensor(values, dtype=torch.long, device=device)
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            values = totals.cpu().tolist()

        return {
            "num_requests": int(values[0]),
            "input_tokens": int(values[1]),
            "output_tokens": int(values[2]),
            "total_tokens": int(values[1] + values[2]),
            "by_media": {
                "image": {
                    "num_requests": int(values[3]),
                    "input_tokens": int(values[4]),
                    "output_tokens": int(values[5]),
                    "total_tokens": int(values[4] + values[5]),
                },
                "video": {
                    "num_requests": int(values[6]),
                    "input_tokens": int(values[7]),
                    "output_tokens": int(values[8]),
                    "total_tokens": int(values[7] + values[8]),
                },
                "text": {
                    "num_requests": int(values[9]),
                    "input_tokens": int(values[10]),
                    "output_tokens": int(values[11]),
                    "total_tokens": int(values[10] + values[11]),
                },
            },
        }

    def _write_token_usage_summary(self, aggregated_usage):
        if self.rank != 0:
            return

        os.makedirs(os.path.dirname(self.token_usage_path), exist_ok=True)
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "qwen3vl",
            "pretrained": self.path,
            "world_size": self.world_size,
            "max_frames_num": self.max_frames_num,
            "max_pixels": self.max_pixels,
            "video_sampling_strategy": self.video_sampling_strategy,
            "video_sample_fps": self.video_sample_fps,
            "visual_input_mode": self.visual_input_mode,
            "sample_frames_version": self.sample_frames_version,
            "token_usage": aggregated_usage,
            "note": "input_tokens are counted from processor-produced input_ids; output_tokens are newly generated token ids.",
        }
        with open(self.token_usage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        eval_logger.info(f"Qwen3VL token usage saved to {self.token_usage_path}: {aggregated_usage}")

    def get_token_usage_summary(self):
        return self._last_aggregated_token_usage

    def _safe_path_part(self, value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"

    def _sample_frames_run_label(self):
        if is_blind_mode(self.visual_input_mode):
            return "blind"
        if self.video_sampling_strategy == "fps":
            fps_label = f"{self.video_sample_fps:g}".replace(".", "p")
            return f"{fps_label}fps"
        return f"{self.max_frames_num}f"

    def _get_question_key(self, doc_id, task, split):
        doc_data = self.task_dict[task][split][doc_id]
        for possible_key in ["question_id", "id", "ID", "Question_ID", "questionId"]:
            if possible_key in doc_data:
                return str(doc_data[possible_key])
        return str(doc_id)

    def _sample_frame_indices(self, vr, total_frames, doc_id, task, split):
        import numpy as np

        if total_frames <= 0:
            raise ValueError("Video has no frames.")

        if self.video_sampling_strategy == "specific":
            question_key = self._get_question_key(doc_id, task, split)

            found_indices = None
            for dataset_name, questions in self.keyframe_mapping.items():
                if question_key in questions:
                    found_indices = questions[question_key]
                    break

            if found_indices is None or len(found_indices) == 0:
                raise ValueError(f"Specific frame indices not found or empty for question ID: {question_key} (doc_id: {doc_id}) in {task}")
            frame_indices = [int(index) for index in found_indices]
        elif self.video_sampling_strategy == "fps":
            avg_fps = float(vr.get_avg_fps())
            if avg_fps <= 0:
                raise ValueError(f"Cannot use fps sampling because video avg_fps is invalid: {avg_fps}")

            duration = total_frames / avg_fps
            timestamps = np.arange(0, duration, 1.0 / self.video_sample_fps)
            if len(timestamps) == 0:
                timestamps = np.array([0.0])
            frame_indices = np.floor(timestamps * avg_fps).astype(int).tolist()
            frame_indices = list(dict.fromkeys(frame_indices))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames_num, dtype=int).tolist()

        frame_indices = [min(max(0, int(index)), total_frames - 1) for index in frame_indices]
        return list(dict.fromkeys(frame_indices))

    def _save_sampled_frames(self, frames, frame_indices, video_path, task, split, doc_id):
        from pathlib import Path
        from PIL import Image

        extracted_model_name = self.path.split("/")[-1] if "/" in self.path else self.path
        base_model_dir = os.path.join("sample_frames", f"{extracted_model_name}-{self._sample_frames_run_label()}")
        version_dir = self.sample_frames_version

        video_stem = self._safe_path_part(Path(video_path).stem)
        question_key = self._safe_path_part(self._get_question_key(doc_id, task, split))
        sample_dir = os.path.join(base_model_dir, version_dir)
        os.makedirs(sample_dir, exist_ok=True)

        file_prefix = "_".join(
            [
                self._safe_path_part(task),
                self._safe_path_part(split),
                f"rank{self.rank}",
                f"doc{self._safe_path_part(doc_id)}",
                question_key,
                video_stem,
            ]
        )
        frame_paths = []
        for idx, (frame_arr, source_index) in enumerate(zip(frames, frame_indices)):
            img = Image.fromarray(frame_arr).convert("RGB")
            save_path = os.path.join(sample_dir, f"{file_prefix}_frame{idx:04d}_src{int(source_index):06d}.jpg")
            temp_path = f"{save_path}.tmp_rank{self.rank}_pid{os.getpid()}"
            with open(temp_path, "wb") as f:
                img.save(f, format="JPEG", quality=95)
            os.replace(temp_path, save_path)
            frame_paths.append(save_path)

        return frame_paths

    def _generate_from_messages(self, messages, gen_kwargs, media_type):
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        processor_kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        inputs = self._processor(**processor_kwargs)
        inputs = inputs.to(self._device)
        input_tokens = int(inputs.input_ids.numel())

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_tokens = sum(int(ids.numel()) for ids in generated_ids_trimmed)
        output_text = self._processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self._record_token_usage(media_type, input_tokens, output_tokens)

        del inputs
        del generated_ids
        del generated_ids_trimmed
        gc.collect()
        torch.cuda.empty_cache()
        return output_text

    def generate_until(self, requests) -> List[str]:
        self._reset_token_usage()
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Merge generation kwargs
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            if is_blind_mode(self.visual_input_mode):
                contexts = strip_visual_context(contexts)
                media_type, media_path = "text", None
            else:
                media_type, media_path = self._get_media_info(doc_to_visual(self.task_dict[task][split][doc_id]))
            if media_type == "text":
                pbar.set_postfix_str("Text")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{contexts}"},
                        ],
                    }
                ]
                output_text = self._generate_from_messages(messages, gen_kwargs, media_type)
            elif media_type == "image":
                unique_image_name = os.path.join(*str(media_path).split(os.sep)[-3:])
                pbar.set_postfix_str(f"Image: {unique_image_name}")
                image_content = {
                    "type": "image",
                    "image": media_path,
                }
                if self.max_pixels is not None:
                    image_content["max_pixels"] = self.max_pixels
                messages = [
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {"type": "text", "text": f"{contexts}"},
                        ],
                    }
                ]
                output_text = self._generate_from_messages(messages, gen_kwargs, media_type)
            elif media_type == "video":
                if self.sample_frames_version is None:
                    self._determine_sample_frames_version()
                video_path = str(media_path)
                # ====== 新增：提取唯一视频名称用于日志核查 ======
                # 将完整路径截取为倒数 2 层目录 + 视频名称的格式
                unique_video_name = os.path.join(*video_path.split(os.sep)[-3:])
                # 取消单独打印，而是将视频名字动态挂载到进度条的右侧
                pbar.set_postfix_str(f"Video: {unique_video_name}")
                # ==============================================

                # ================= 正确的视觉参数传入方式与严谨抽帧 =================
                # 为了不依赖 qwen_vl_utils 底层的黑盒抽帧，确保采样策略由评估协议显式控制。
                # 尤其针对医学视频评估中时序连贯性的严谨要求，这里显式使用 decord 进行抽取
                import decord
                decord.bridge.set_bridge("torch")
                vr = decord.VideoReader(video_path, num_threads=1)
                total_frames = len(vr)
                frame_indices = self._sample_frame_indices(vr, total_frames, doc_id, task, split)
                
                # 不依赖底层对于 `video` 路径的处理。我们将严格依据索引拉取图像矩阵并转存传递。
                frames = vr.get_batch(frame_indices).numpy()
                del vr  # 释放内存与文件句柄

                # 将截取的帧存成特定格式交给官方组件解析。Qwen 的 qwen_vl_utils 支持 
                # {"type": "video", "video": [ "path/to/frame1.jpg", "path/to/frame2.jpg", ... ]}
                # 这样它就会将这些帧组装为连续时序视频而不再去利用 cv2/av 等库对单一视频文件盲目抽帧
                
                # 每个样本使用独立目录并原子替换 jpg，避免多进程/多题复用同一视频时读到半写入文件。
                frame_paths = self._save_sampled_frames(frames, frame_indices, video_path, task, split, doc_id)
                
                # 这种传法彻底锁死了视频内容为我们手动截取的帧数
                video_content = {
                    "type": "video",
                    "video": frame_paths,
                    "fps": 1.0 # 这里的 fps 在列表模式下仅用于控制底层时间戳计算(如每秒1帧)，对截断无影响
                }
                if self.max_pixels is not None:
                    video_content["max_pixels"] = self.max_pixels
                messages = [
                    {
                        "role": "user",
                        "content": [
                            video_content,
                            {"type": "text", "text": f"{contexts}"},
                        ],
                    }
                ]

                output_text = self._generate_from_messages(messages, gen_kwargs, media_type)
            else:
                raise NotImplementedError
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        aggregated_usage = self._aggregate_token_usage_across_ranks()
        self._last_aggregated_token_usage = aggregated_usage
        self._write_token_usage_summary(aggregated_usage)
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."
