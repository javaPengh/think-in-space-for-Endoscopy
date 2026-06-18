import gc
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils import blind_eval
from lmms_eval.models.model_utils.question_id import get_question_key

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=1024,
    do_sample=False,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@register_model("medgemma_27b")
class MedGemma27B(lmms):
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/google/medgemma-27b-it"
    MODEL_DISPLAY_NAME = "MedGemma-27B-IT"
    TOKEN_USAGE_MODEL_NAME = "medgemma_27b"
    TOKEN_USAGE_FILENAME = "medgemma_27b_token_usage.jsonl"

    def __init__(
        self,
        pretrained: str = None,
        modality: str = "video",
        device: str = "cuda:0",
        device_map: str = "auto",
        batch_size: str = "1",
        max_frames_num: int = 32,
        video_sampling_strategy: str = "uniform",
        video_sample_fps: float = None,
        keyframe_mapping_path: str = "data/keyframe_mapping.json",
        visual_input_mode: str = "visual",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        local_files_only: bool = True,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            eval_logger.warning(f"{self.MODEL_DISPLAY_NAME} ignoring unsupported model_args: {sorted(kwargs.keys())}")

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for {self.MODEL_DISPLAY_NAME}, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        self.visual_input_mode = blind_eval.normalize_visual_input_mode(visual_input_mode)
        self.modality = modality
        self.max_frames_num = int(max_frames_num)
        if self.max_frames_num < 1:
            raise ValueError(f"max_frames_num must be >= 1 for {self.MODEL_DISPLAY_NAME}, got {max_frames_num}.")

        self.video_sample_fps = None if video_sample_fps in (None, "", "none", "None") else float(video_sample_fps)
        self.video_sampling_strategy = str(video_sampling_strategy).lower()
        if self.video_sample_fps is not None and self.video_sampling_strategy == "uniform":
            self.video_sampling_strategy = "fps"
        if self.video_sampling_strategy in {"fps_uniform", "uniform_fps"}:
            self.video_sampling_strategy = "fps"
        if not blind_eval.is_blind_mode(self.visual_input_mode):
            if self.video_sampling_strategy not in {"uniform", "specific", "fps"}:
                raise ValueError(f"Unsupported video_sampling_strategy for {self.MODEL_DISPLAY_NAME}: {self.video_sampling_strategy}")
            if self.video_sampling_strategy == "fps" and (self.video_sample_fps is None or self.video_sample_fps <= 0):
                raise ValueError("video_sample_fps must be a positive number when video_sampling_strategy='fps'.")

        self.keyframe_mapping = {}
        if self.video_sampling_strategy == "specific" and not blind_eval.is_blind_mode(self.visual_input_mode):
            if not os.path.exists(keyframe_mapping_path):
                raise ValueError(f"Keyframe mapping file not found at {keyframe_mapping_path}. Required when video_sampling_strategy is 'specific'.")
            with open(keyframe_mapping_path, "r", encoding="utf-8") as f:
                self.keyframe_mapping = json.load(f)

        self.local_files_only = self._to_bool(local_files_only)
        self.path = os.path.expanduser(pretrained or self.DEFAULT_PRETRAINED)
        if self.local_files_only and not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.MODEL_DISPLAY_NAME} checkpoint was not found at {self.path}. " "Download google/medgemma-27b-it from ModelScope to this local path, or pass a local pretrained path.")

        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        self.attn_implementation = None if attn_implementation in (None, "", "none", "None") else str(attn_implementation)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if self.accelerator.num_processes > 1 and device_map == "auto":
            raise ValueError(f"{self.MODEL_DISPLAY_NAME} with device_map=auto should be launched with one process. " "Use --num_processes 1 and expose multiple GPUs through CUDA_VISIBLE_DEVICES.")

        self.device_map = device_map
        if torch.cuda.is_available():
            self._device = torch.device(device if device_map == "auto" else f"cuda:{self.accelerator.local_process_index}")
        else:
            self._device = torch.device("cpu")

        model_class = self._resolve_model_class()
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
        }
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        if device_map == "auto":
            model_kwargs["device_map"] = device_map
            self._model = model_class.from_pretrained(self.path, **model_kwargs).eval()
            eval_logger.info(f"Using {self.MODEL_DISPLAY_NAME} with model parallel device_map=auto")
        else:
            self._model = model_class.from_pretrained(self.path, **model_kwargs).eval().to(self._device)
            eval_logger.info(f"Using {self.MODEL_DISPLAY_NAME} on device {self._device}")

        self._processor = AutoProcessor.from_pretrained(self.path, local_files_only=self.local_files_only)
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self.sample_frames_version = None
        self.token_usage_path = str(Path(__file__).resolve().parents[2] / "docs" / self.TOKEN_USAGE_FILENAME)
        self._last_aggregated_token_usage = None
        self._reset_token_usage()

    @property
    def config(self):
        return self._model.config

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
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

    def _resolve_model_class(self):
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            raise ImportError(f"{self.MODEL_DISPLAY_NAME} requires Transformers with AutoModelForImageTextToText. " "Use the MedGemma-compatible Transformers environment.")
        return model_class

    def _resolve_torch_dtype(self, torch_dtype):
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        dtype_name = str(torch_dtype or "bfloat16").lower()
        dtype_map = {
            "auto": "auto",
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype for {self.MODEL_DISPLAY_NAME}: {torch_dtype}")
        return dtype_map[dtype_name]

    def _to_bool(self, value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "none", ""}

    def _flatten_visuals(self, visual):
        if visual is None or visual == []:
            return []
        if isinstance(visual, list):
            return visual
        return [visual]

    def _infer_media_type(self, media_path):
        if isinstance(media_path, Image.Image):
            return "image"
        suffix = os.path.splitext(str(media_path))[1].lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        return self.modality

    def _get_media_inputs(self, visual):
        visual_items = self._flatten_visuals(visual)
        if not visual_items:
            return "text", []

        first_visual = visual_items[0]
        if isinstance(first_visual, dict):
            media_inputs = [item.get("path") or item.get("media_path") for item in visual_items]
            if any(item is None for item in media_inputs):
                raise ValueError(f"Missing media path in visual input: {visual_items}")
            media_type = str(first_visual.get("media_type") or self._infer_media_type(media_inputs[0])).lower()
        else:
            media_inputs = visual_items
            media_type = self._infer_media_type(first_visual)

        if media_type not in {"image", "video"}:
            raise ValueError(f"Unsupported media_type for {self.MODEL_DISPLAY_NAME}: {media_type}")
        return media_type, media_inputs

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
            totals = torch.tensor(values, dtype=torch.long, device=self._device)
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
            "model": self.TOKEN_USAGE_MODEL_NAME,
            "pretrained": self.path,
            "world_size": self.world_size,
            "max_frames_num": self.max_frames_num,
            "video_sampling_strategy": self.video_sampling_strategy,
            "video_sample_fps": self.video_sample_fps,
            "visual_input_mode": self.visual_input_mode,
            "sample_frames_version": self.sample_frames_version,
            "token_usage": aggregated_usage,
            "note": "input_tokens are counted from processor-produced input_ids; output_tokens are newly generated token ids.",
        }
        with open(self.token_usage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        eval_logger.info(f"{self.MODEL_DISPLAY_NAME} token usage saved to {self.token_usage_path}: {aggregated_usage}")

    def get_token_usage_summary(self):
        return self._last_aggregated_token_usage

    def _safe_path_part(self, value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"

    def _sample_frames_run_label(self):
        if blind_eval.is_blind_mode(self.visual_input_mode):
            return "blind"
        if self.video_sampling_strategy == "fps":
            fps_label = f"{self.video_sample_fps:g}".replace(".", "p")
            return f"{fps_label}fps"
        return f"{self.max_frames_num}f"

    def _determine_sample_frames_version(self):
        if self.sample_frames_version is not None:
            return self.sample_frames_version

        extracted_model_name = self._safe_path_part(Path(self.path).name)
        base_model_dir = os.path.join("sample_frames", f"{extracted_model_name}-{self._sample_frames_run_label()}")

        if self.rank == 0:
            os.makedirs(base_model_dir, exist_ok=True)
            version_str = datetime.now().strftime("%Y%m%d%H%M%S")
            os.makedirs(os.path.join(base_model_dir, version_str), exist_ok=True)
            version_obj = [version_str]
        else:
            version_obj = [None]

        if self.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(version_obj, src=0)

        self.sample_frames_version = version_obj[0]
        return self.sample_frames_version

    def _get_question_key(self, doc_id, task, split):
        return get_question_key(self.task_dict[task][split][doc_id], doc_id)

    def _sample_frame_indices(self, vr, total_frames, doc_id, task, split):
        if total_frames <= 0:
            raise ValueError("Video has no frames.")

        if self.video_sampling_strategy == "specific":
            question_key = self._get_question_key(doc_id, task, split)
            found_indices = None
            for questions in self.keyframe_mapping.values():
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
            if len(frame_indices) > self.max_frames_num:
                keep_indices = np.linspace(0, len(frame_indices) - 1, self.max_frames_num, dtype=int)
                frame_indices = [frame_indices[index] for index in keep_indices]
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames_num, dtype=int).tolist()

        frame_indices = [min(max(0, int(index)), total_frames - 1) for index in frame_indices]
        return list(dict.fromkeys(frame_indices))

    def _save_sampled_frames(self, frames, frame_indices, video_path, task, split, doc_id):
        extracted_model_name = self._safe_path_part(Path(self.path).name)
        base_model_dir = os.path.join("sample_frames", f"{extracted_model_name}-{self._sample_frames_run_label()}")
        sample_dir = os.path.join(base_model_dir, self.sample_frames_version)
        os.makedirs(sample_dir, exist_ok=True)

        video_stem = self._safe_path_part(Path(video_path).stem)
        question_key = self._safe_path_part(self._get_question_key(doc_id, task, split))
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
            image = Image.fromarray(frame_arr).convert("RGB")
            save_path = os.path.join(sample_dir, f"{file_prefix}_frame{idx:04d}_src{int(source_index):06d}.jpg")
            temp_path = f"{save_path}.tmp_rank{self.rank}_pid{os.getpid()}"
            with open(temp_path, "wb") as f:
                image.save(f, format="JPEG", quality=95)
            os.replace(temp_path, save_path)
            frame_paths.append(save_path)
        return frame_paths

    def _load_image(self, image_input):
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        with Image.open(os.path.expanduser(str(image_input))) as image:
            return image.convert("RGB").copy()

    def _load_video_frame_images(self, video_path, doc_id, task, split):
        if self.sample_frames_version is None:
            self._determine_sample_frames_version()

        video_path = os.path.expanduser(str(video_path))
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        frame_indices = self._sample_frame_indices(vr, total_frames, doc_id, task, split)
        frames = vr.get_batch(frame_indices).asnumpy()
        del vr

        frame_paths = self._save_sampled_frames(frames, frame_indices, video_path, task, split, doc_id)
        return [self._load_image(frame_path) for frame_path in frame_paths], frame_indices

    def _move_inputs_to_device(self, inputs):
        inputs = inputs.to(self._device)
        if self.torch_dtype == "auto":
            return inputs
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[key] = value.to(dtype=self.torch_dtype)
        return inputs

    def _build_inputs(self, messages, images):
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            processor_kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
            if images:
                processor_kwargs["images"] = images
            inputs = self._processor(**processor_kwargs)
        return self._move_inputs_to_device(inputs)

    def _generate_from_content(self, content, images, gen_kwargs, media_type):
        messages = [{"role": "user", "content": content}]
        inputs = self._build_inputs(messages, images)
        input_tokens = int(inputs["input_ids"].numel())

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        generated_ids_trimmed = generated_ids[:, input_length:]
        output_tokens = int(generated_ids_trimmed.numel())
        output_text = self._processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self._record_token_usage(media_type, input_tokens, output_tokens)

        del inputs
        del generated_ids
        del generated_ids_trimmed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output_text

    def _prepare_generation_kwargs(self, gen_kwargs):
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.pop("until", None)
        for key, value in DEFAULT_GEN_KWARGS.items():
            gen_kwargs.setdefault(key, value)

        tokenizer = getattr(self._processor, "tokenizer", None)
        if "pad_token_id" not in gen_kwargs and tokenizer is not None and tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
        return gen_kwargs

    def generate_until(self, requests) -> List[str]:
        self._reset_token_usage()
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            gen_kwargs = self._prepare_generation_kwargs(gen_kwargs)

            if blind_eval.is_blind_mode(self.visual_input_mode):
                contexts = blind_eval.strip_visual_context(contexts)
                media_type, visuals = "text", []
            else:
                media_type, visuals = self._get_media_inputs(doc_to_visual(self.task_dict[task][split][doc_id]))

            if media_type == "text":
                pbar.set_postfix_str("Text")
                content = [{"type": "text", "text": f"{contexts}"}]
                output_text = self._generate_from_content(content, [], gen_kwargs, media_type)
            elif media_type == "image":
                unique_image_name = os.path.join(*str(visuals[0]).split(os.sep)[-3:])
                pbar.set_postfix_str(f"Image: {unique_image_name}")
                images = [self._load_image(image_input) for image_input in visuals]
                content = [{"type": "image", "image": image} for image in images]
                content.append({"type": "text", "text": f"{contexts}"})
                output_text = self._generate_from_content(content, images, gen_kwargs, media_type)
            elif media_type == "video":
                assert len(visuals) == 1, f"{self.MODEL_DISPLAY_NAME} supports one video per request, got {len(visuals)}."
                video_path = str(visuals[0])
                unique_video_name = os.path.join(*video_path.split(os.sep)[-3:])
                pbar.set_postfix_str(f"Video: {unique_video_name}")

                images, _ = self._load_video_frame_images(video_path, doc_id, task, split)
                content = [{"type": "text", "text": "The following images are sampled frames from one video, ordered by time."}]
                content.extend({"type": "image", "image": image} for image in images)
                content.append({"type": "text", "text": f"{contexts}"})
                output_text = self._generate_from_content(content, images, gen_kwargs, media_type)
            else:
                raise NotImplementedError(f"Unsupported media_type for {self.MODEL_DISPLAY_NAME}: {media_type}")

            res.append(output_text)
            pbar.update(1)

        pbar.close()
        aggregated_usage = self._aggregate_token_usage_across_ranks()
        self._last_aggregated_token_usage = aggregated_usage
        self._write_token_usage_summary(aggregated_usage)
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(f"{self.MODEL_DISPLAY_NAME} loglikelihood is not implemented.")
