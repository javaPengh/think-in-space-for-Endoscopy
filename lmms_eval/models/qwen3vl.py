import logging
import os
from datetime import timedelta
from typing import List, Tuple
from qwen_vl_utils import process_vision_info
import torch
import gc
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger as eval_logger

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=1024,
    do_sample=False,
)


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
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained

        # Load model
        if device_map == "auto":
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, trust_remote_code=True, device_map=device_map).eval()
        else:
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()

        self._processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)

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
        self.max_frames_num = max_frames_num
        self.max_pixels = max_pixels
        self.sample_frames_version = None

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
        # 分布式安全地确定 sample_frames 的版本目录 (仅执行一次)
        import os
        import re
        import torch.distributed as dist

        if self.sample_frames_version is not None:
            return self.sample_frames_version

        extracted_model_name = self.path.split("/")[-1] if "/" in self.path else self.path
        base_model_dir = os.path.join("sample_frames", f"{extracted_model_name}-{self.max_frames_num}f")

        if self.rank == 0:
            os.makedirs(base_model_dir, exist_ok=True)
            existing_versions = []
            for d in os.listdir(base_model_dir):
                if os.path.isdir(os.path.join(base_model_dir, d)) and re.match(r"^v_\d+$", d):
                    existing_versions.append(int(d.split("_")[1]))

            next_version = max(existing_versions) + 1 if existing_versions else 1
            version_str = f"v_{next_version:02d}"
            # 提前由主进程创建以避免竞争
            os.makedirs(os.path.join(base_model_dir, version_str), exist_ok=True)
            version_obj = [version_str]
        else:
            version_obj = [None]

        if self.world_size > 1 and dist.is_initialized():
            dist.broadcast_object_list(version_obj, src=0)

        self.sample_frames_version = version_obj[0]
        return self.sample_frames_version

    def generate_until(self, requests) -> List[str]:
        # 在处理批次前确定好目录版本，确保多进程统一
        if self.modality == "video":
            self._determine_sample_frames_version()

        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Merge generation kwargs
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if self.modality == "image":
                raise NotImplementedError("Image inference for Qwen3VL is not supported yet.")
            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]
                # ====== 新增：提取唯一视频名称用于日志核查 ======
                # 将完整路径截取为倒数 2 层目录 + 视频名称的格式
                unique_video_name = os.path.join(*video_path.split(os.sep)[-3:])
                # 取消单独打印，而是将视频名字动态挂载到进度条的右侧
                pbar.set_postfix_str(f"Video: {unique_video_name}")
                # ==============================================

                # ================= 正确的视觉参数传入方式与严谨抽帧 =================
                # 为了不依赖 qwen_vl_utils 底层的黑盒抽帧，确保无论视频多长都绝对均匀抽取指定帧数
                # 尤其针对医学视频评估中时序连贯性的严谨要求，这里显式使用 decord 进行抽取
                import decord
                import numpy as np
                decord.bridge.set_bridge("torch")
                vr = decord.VideoReader(video_path, num_threads=1)
                total_frames = len(vr)
                
                # 计算绝对均匀的帧索引
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames_num, dtype=int).tolist()
                
                # 不依赖底层对于 `video` 路径的处理。我们将严格依据索引拉取图像矩阵并转存传递。
                from PIL import Image
                
                frames = vr.get_batch(frame_indices).numpy()
                del vr  # 释放内存与文件句柄

                # 将截取的帧存成特定格式交给官方组件解析。Qwen 的 qwen_vl_utils 支持 
                # {"type": "video", "video": [ "path/to/frame1.jpg", "path/to/frame2.jpg", ... ]}
                # 这样它就会将这些帧组装为连续时序视频而不再去利用 cv2/av 等库对单一视频文件盲目抽帧
                
                # ====== 持久化保存到 sample_frames 目录 ======
                from pathlib import Path
                
                extracted_model_name = self.path.split("/")[-1] if "/" in self.path else self.path
                base_model_dir = os.path.join("sample_frames", f"{extracted_model_name}-{self.max_frames_num}f")
                
                version_dir = self.sample_frames_version
                base_save_dir = os.path.join(base_model_dir, version_dir)
                os.makedirs(base_save_dir, exist_ok=True) # 仅防万一，Rank 0 已创建
                video_stem = Path(video_path).stem
                
                frame_paths = []
                for idx, frame_arr in enumerate(frames):
                    img = Image.fromarray(frame_arr)
                    save_path = os.path.join(base_save_dir, f"{video_stem}_frame{idx:04d}.jpg")
                    img.save(save_path, format="JPEG", quality=95)
                    frame_paths.append(save_path)
                
                # 这种传法彻底锁死了视频内容为我们手动截取的帧数
                video_content = {
                    "type": "video",
                    "video": frame_paths,
                    "max_pixels": self.max_pixels,
                    "fps": 1.0 # 这里的 fps 在列表模式下仅用于控制底层时间戳计算(如每秒1帧)，对截断无影响
                }
                messages = [
                    {
                        "role": "user",
                        "content": [
                            video_content,
                            {"type": "text", "text": f"{contexts}"},
                        ],
                    }
                ]

                # ================= 极其关键的修复 =================
                # 1. 仅让 HuggingFace 处理纯文本提示词，不让它碰视频
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # 2. 强制使用 Qwen 官方工具！它才会真正读取 max_pixels
                image_inputs, video_inputs = process_vision_info(messages)

                # 3. 将降采样后体积的安全张量交给模型
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self._device)
                # ===================================================

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)

                # Trim input tokens from output
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self._processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                # ================= 新增：强力显存回收机制 =================
                # 1. 彻底切断当前样本的所有大张量引用
                del inputs
                del generated_ids
                del generated_ids_trimmed

                # 2. 强制 Python 立即回收对象
                gc.collect()

                # 3. 强制 PyTorch 清空 CUDA 缓存池，把显存还给操作系统
                torch.cuda.empty_cache()
                
                # ==========================================================
            else:
                raise NotImplementedError
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."
