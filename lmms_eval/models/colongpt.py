import gc
import json
import os
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.blind_eval import is_blind_mode, normalize_visual_input_mode, strip_visual_context
from lmms_eval.models.model_utils.question_id import get_question_key
from loguru import logger as eval_logger

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=512,
    do_sample=False,
    temperature=0,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
DEFAULT_STOP_STR = "<|endoftext|>"


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keyword, tokenizer, input_ids):
        self.keyword_ids = tokenizer(keyword, add_special_tokens=False).input_ids
        self.start_len = input_ids.shape[1]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.keyword_ids:
            return False
        if input_ids.shape[1] <= self.start_len:
            return False
        keyword = torch.tensor(self.keyword_ids, device=input_ids.device)
        return bool(input_ids[0, -len(self.keyword_ids) :].equal(keyword))


@register_model("colongpt")
class ColonGPT(lmms):
    def __init__(
        self,
        pretrained: str = "ai4colonoscopy/ColonGPT-v1",
        vision_tower_pretrained: str = None,
        modality: str = "video",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        max_frames_num: int = 4,
        video_sampling_strategy: str = "uniform",
        video_sample_fps: float = None,
        keyframe_mapping_path: str = "data/keyframe_mapping.json",
        visual_input_mode: str = "visual",
        torch_dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            eval_logger.warning(f"ColonGPT ignoring unsupported model_args: {sorted(kwargs.keys())}")

        self.path = os.path.expanduser(pretrained)
        self.vision_tower_pretrained = os.path.expanduser(vision_tower_pretrained) if vision_tower_pretrained else None
        self.modality = modality
        self.visual_input_mode = normalize_visual_input_mode(visual_input_mode)
        self.max_frames_num = int(max_frames_num)
        if self.max_frames_num < 1:
            raise ValueError(f"max_frames_num must be >= 1 for ColonGPT, got {max_frames_num}.")

        self.video_sample_fps = None if video_sample_fps in (None, "", "none", "None") else float(video_sample_fps)
        self.video_sampling_strategy = str(video_sampling_strategy).lower()
        if self.video_sample_fps is not None and self.video_sampling_strategy == "uniform":
            self.video_sampling_strategy = "fps"
        if self.video_sampling_strategy in {"fps_uniform", "uniform_fps"}:
            self.video_sampling_strategy = "fps"

        if not is_blind_mode(self.visual_input_mode):
            if self.video_sampling_strategy not in {"uniform", "specific", "fps"}:
                raise ValueError(f"Unsupported video_sampling_strategy for ColonGPT: {self.video_sampling_strategy}")
            if self.video_sampling_strategy == "fps" and (self.video_sample_fps is None or self.video_sample_fps <= 0):
                raise ValueError("video_sample_fps must be a positive number when video_sampling_strategy='fps'.")

        self.keyframe_mapping = {}
        if self.video_sampling_strategy == "specific" and not is_blind_mode(self.visual_input_mode):
            if not os.path.exists(keyframe_mapping_path):
                raise ValueError(f"Keyframe mapping file not found at {keyframe_mapping_path}. Required when video_sampling_strategy is 'specific'.")
            with open(keyframe_mapping_path, "r", encoding="utf-8") as f:
                self.keyframe_mapping = json.load(f)

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for ColonGPT, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if self.accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            self.device_map = f"cuda:{self.accelerator.local_process_index}"
        elif self.accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}") if torch.cuda.is_available() else torch.device("cpu")
            self.device_map = f"cuda:{self.accelerator.local_process_index}" if torch.cuda.is_available() else "cpu"

        model_dtype = self._resolve_dtype(torch_dtype)
        model_kwargs = {
            "torch_dtype": model_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.device_map == "auto":
            model_kwargs["device_map"] = self.device_map
            self._model = AutoModelForCausalLM.from_pretrained(self.path, **model_kwargs).eval()
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.path, **model_kwargs).eval().to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._patch_vision_tower_path()
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self._input_device = self._infer_input_device()
        eval_logger.info(f"Using ColonGPT on rank={self.rank}, device={self._input_device}, world_size={self.world_size}")

    @property
    def config(self):
        return self.model.config

    @property
    def tokenizer(self):
        return self._tokenizer

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
        return self._input_device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _resolve_dtype(self, torch_dtype):
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        dtype = str(torch_dtype or "float16").lower()
        if dtype in {"auto"}:
            return "auto"
        if dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype for ColonGPT: {torch_dtype}")

    def _infer_input_device(self):
        if hasattr(self.model, "get_model"):
            projector = getattr(self.model.get_model(), "mm_projector", None)
            if projector is not None:
                try:
                    return next(projector.parameters()).device
                except StopIteration:
                    pass
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self._device

    def _get_vision_tower(self):
        for target in [self.model, getattr(self.model, "model", None)]:
            if target is not None and hasattr(target, "get_vision_tower"):
                vision_tower = target.get_vision_tower()
                if vision_tower is not None:
                    return vision_tower
        return None

    def _patch_vision_tower_path(self):
        if not self.vision_tower_pretrained:
            return

        for config in [getattr(self.model, "config", None), getattr(getattr(self.model, "model", None), "config", None)]:
            if config is not None and hasattr(config, "mm_vision_tower"):
                config.mm_vision_tower = self.vision_tower_pretrained

        vision_tower = self._get_vision_tower()
        if vision_tower is not None:
            if getattr(vision_tower, "is_loaded", False):
                eval_logger.warning("ColonGPT vision tower was already loaded before path patching; verify it used the intended local SigLIP path.")
            vision_tower.vision_tower_name = self.vision_tower_pretrained

    def _move_vision_tower_to_device(self):
        vision_tower = self._get_vision_tower()
        if vision_tower is None or not getattr(vision_tower, "is_loaded", False):
            return
        vision_tower.to(device=self.device, dtype=self.model.dtype)

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

    def _open_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        with Image.open(image) as raw_image:
            return raw_image.convert("RGB")

    def _get_media_inputs(self, visual, doc_id=None, task=None, split=None):
        visual_items = self._flatten_visuals(visual)
        if not visual_items:
            return "text", []

        first_visual = visual_items[0]
        if isinstance(first_visual, Image.Image):
            return "image", [item.convert("RGB") for item in visual_items]

        if isinstance(first_visual, dict):
            media_type = str(first_visual.get("media_type") or "").lower()
            media_paths = [item.get("path") or item.get("media_path") for item in visual_items]
            if any(path is None for path in media_paths):
                raise ValueError(f"Missing media path in visual input: {visual_items}")
            if not media_type:
                media_type = self._infer_media_type(media_paths[0])
        else:
            media_paths = visual_items
            media_type = self._infer_media_type(media_paths[0])

        if media_type == "image":
            return "image", [self._open_image(path) for path in media_paths]
        if media_type == "video":
            if len(media_paths) != 1:
                raise ValueError(f"ColonGPT supports one video per request, got {len(media_paths)}.")
            question_key = get_question_key(self.task_dict[task][split][doc_id], doc_id) if task is not None and split is not None and doc_id is not None else None
            return "video", self._load_video_frames(media_paths[0], question_key=question_key, task=task)
        raise ValueError(f"Unsupported media_type for ColonGPT: {media_type}")

    def _sample_frame_indices(self, vr, total_frames, question_key=None, task=None):
        if total_frames <= 0:
            raise ValueError("Video has no frames.")

        if self.video_sampling_strategy == "specific":
            found_indices = None
            if self.keyframe_mapping and question_key is not None:
                for _, questions in self.keyframe_mapping.items():
                    if question_key in questions:
                        found_indices = questions[question_key]
                        break
            if found_indices is None or len(found_indices) == 0:
                raise ValueError(f"Specific frame indices not found or empty for question ID: {question_key} in {task}")
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

    def _load_video_frames(self, video_path, question_key=None, task=None):
        from decord import VideoReader, cpu

        video_path = os.path.expanduser(str(video_path))
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_indices = self._sample_frame_indices(vr, len(vr), question_key=question_key, task=task)
        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in frames]

    def _build_prompt(self, question, num_images):
        if num_images > 0:
            image_prefix = "\n".join([IMAGE_TOKEN] * num_images)
            return f"USER: {image_prefix}\n{question} ASSISTANT:"
        return f"USER: {question} ASSISTANT:"

    def _tokenize_prompt(self, prompt):
        if IMAGE_TOKEN not in prompt:
            return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

        chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split(IMAGE_TOKEN)]
        input_ids = []
        for idx, chunk in enumerate(chunks):
            input_ids.extend(chunk)
            if idx != len(chunks) - 1:
                input_ids.append(IMAGE_TOKEN_INDEX)
        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    def _prepare_images(self, images):
        if not images:
            return None
        image_tensor = self.model.process_images(images, self.model.config)
        self._move_vision_tower_to_device()
        return image_tensor.to(dtype=self.model.dtype, device=self.device)

    def _normalize_until(self, until):
        if until is None:
            return []
        if isinstance(until, str):
            return [until]
        return [item for item in until if item]

    def _trim_until(self, text, until):
        text = text.replace(DEFAULT_STOP_STR, "").strip()
        for stop in until:
            stop_index = text.find(stop)
            if stop_index != -1:
                text = text[:stop_index].strip()
        return text

    def _generate(self, question, images, gen_kwargs):
        until = self._normalize_until(gen_kwargs.pop("until", None))
        for key, value in DEFAULT_GEN_KWARGS.items():
            gen_kwargs.setdefault(key, value)
        if self.tokenizer.eos_token_id is not None:
            gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
            gen_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        prompt = self._build_prompt(question, len(images))
        input_ids = self._tokenize_prompt(prompt).to(self.device)
        image_tensor = self._prepare_images(images)

        stop_sequences = until + [DEFAULT_STOP_STR]
        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop, self.tokenizer, input_ids) for stop in stop_sequences if stop])
        if len(stopping_criteria) > 0:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        generate_kwargs = {"input_ids": input_ids, **gen_kwargs}
        if image_tensor is not None:
            generate_kwargs["images"] = image_tensor

        with torch.inference_mode():
            output_ids = self.model.generate(**generate_kwargs)

        output_ids = output_ids[0, input_ids.shape[1] :]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return self._trim_until(output_text, until)

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            gen_kwargs = dict(gen_kwargs)
            if is_blind_mode(self.visual_input_mode):
                contexts = strip_visual_context(contexts)
                media_type, visuals = "text", []
            else:
                media_type, visuals = self._get_media_inputs(doc_to_visual(self.task_dict[task][split][doc_id]), doc_id=doc_id, task=task, split=split)

            pbar.set_postfix_str(media_type)
            output_text = self._generate(contexts, visuals, gen_kwargs)
            res.append(output_text)

            del visuals
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("ColonGPT loglikelihood is not implemented.")
