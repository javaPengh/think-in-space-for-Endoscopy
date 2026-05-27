import base64
import gc
import hashlib
import json
import os
import time
from datetime import timedelta
from io import BytesIO
from typing import List, Tuple

import numpy as np
import requests
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.token_usage import TokenUsageTracker, extract_gemini_usage
from loguru import logger as eval_logger

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=1024,
    do_sample=False,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
@register_model("gemini3_1_pro")
@register_model("gemini3_pro")
class Gemini3_1Pro(lmms):
    def __init__(
        self,
        model_version: str = "gemini-3.1-pro-preview",
        modality: str = "video",
        max_frames_num: int = 16,
        video_sampling_strategy: str = "uniform",
        keyframe_mapping_path: str = "data/keyframe_mapping.json",
        api_key: str = None,
        api_key_env: str = "GOOGLE_API_KEY",
        api_url: str = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 180,
        max_retries: int = 5,
        retry_sleep: int = 30,
        cache_dir: str = "api_cache",
        jpeg_quality: int = 95,
        include_frame_metadata: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = int(max_frames_num)
        self.video_sampling_strategy = video_sampling_strategy
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.retry_sleep = int(retry_sleep)
        self.jpeg_quality = int(jpeg_quality)
        self.include_frame_metadata = str(include_frame_metadata).lower() not in {"false", "0", "no"}
        self.keyframe_mapping = {}
        self.token_usage_tracker = TokenUsageTracker()

        self.api_key = api_key or os.getenv(api_key_env) or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(f"Missing Gemini API key. Set {api_key_env} or GEMINI_API_KEY, or pass api_key in model_args.")

        model_path = self.model_version if self.model_version.startswith("models/") else f"models/{self.model_version}"
        self.api_url = api_url or f"{base_url.rstrip('/')}/{model_path}:generateContent"
        self.headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        if self.video_sampling_strategy == "specific":
            if not os.path.exists(keyframe_mapping_path):
                raise ValueError(f"Keyframe mapping file not found at {keyframe_mapping_path}. Required when video_sampling_strategy is 'specific'.")
            with open(keyframe_mapping_path, "r", encoding="utf-8") as f:
                self.keyframe_mapping = json.load(f)

        os.makedirs(cache_dir, exist_ok=True)
        cache_name = self.model_version.replace("/", "_").replace(":", "_")
        self.cache_path = os.path.join(cache_dir, f"{cache_name}_responses.json")
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.response_cache = json.load(f)
        else:
            self.response_cache = {}

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self.batch_size_per_gpu = 1

    @property
    def config(self):
        return None

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

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
            raise ValueError(f"Unsupported media_type for Gemini 3.1 Pro Preview: {media_type}")
        return media_type, media_inputs

    def _image_to_inline_part(self, image):
        image = image.convert("RGB")
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG", quality=self.jpeg_quality)
        encoded = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        return {"inline_data": {"mime_type": "image/jpeg", "data": encoded}}

    def _path_to_inline_part(self, image_path):
        with Image.open(image_path) as image:
            return self._image_to_inline_part(image)

    def _media_to_inline_part(self, media):
        if isinstance(media, Image.Image):
            return self._image_to_inline_part(media)
        return self._path_to_inline_part(media)

    def _frame_to_inline_part(self, frame):
        return self._image_to_inline_part(Image.fromarray(frame).convert("RGB"))

    def _get_question_key(self, doc_id, task, split):
        doc_data = self.task_dict[task][split][doc_id]
        for possible_key in ["question_id", "id", "ID", "Question_ID", "questionId"]:
            if possible_key in doc_data:
                return str(doc_data[possible_key])
        return str(doc_id)

    def _sample_video_frames(self, video_path, doc_id, task, split):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")

        if self.video_sampling_strategy == "specific":
            question_key = self._get_question_key(doc_id, task, split)
            found_indices = None
            for _, questions in self.keyframe_mapping.items():
                if question_key in questions:
                    found_indices = questions[question_key]
                    break
            if found_indices is None or len(found_indices) == 0:
                raise ValueError(f"Specific frame indices not found or empty for question ID: {question_key} (doc_id: {doc_id}) in {task}")
            frame_indices = [int(index) for index in found_indices]
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames_num, dtype=int).tolist()

        frame_indices = [min(max(0, int(index)), total_frames - 1) for index in frame_indices]
        frames = vr.get_batch(frame_indices).asnumpy()
        return frame_indices, frames

    def _short_path(self, path):
        parts = str(path).split(os.sep)
        return os.path.join(*parts[-3:]) if len(parts) >= 3 else str(path)

    def _build_image_parts(self, visuals, contexts):
        parts = [self._media_to_inline_part(visual) for visual in visuals]
        parts.append({"text": contexts})
        return parts

    def _build_frame_video_parts(self, frame_indices, frames, contexts):
        parts = []
        if self.include_frame_metadata:
            sampling_note = "The following images are sampled frames from one video, ordered by time."
            if self.video_sampling_strategy == "specific":
                sampling_note += " They are selected keyframes and may not be uniformly spaced in time."
            parts.append({"text": sampling_note})

        for position, (source_index, frame) in enumerate(zip(frame_indices, frames), start=1):
            if self.include_frame_metadata:
                parts.append({"text": f"Frame {position}, source frame index {source_index}."})
            parts.append(self._frame_to_inline_part(frame))

        parts.append({"text": contexts})
        return parts

    def _normalize_gen_kwargs(self, gen_kwargs):
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.pop("until", None)
        gen_kwargs.pop("num_beams", None)
        for key, value in DEFAULT_GEN_KWARGS.items():
            gen_kwargs.setdefault(key, value)
        if not gen_kwargs.get("do_sample", True):
            gen_kwargs.pop("top_p", None)
        return gen_kwargs

    def _build_payload(self, parts, gen_kwargs):
        generation_config = {
            "maxOutputTokens": int(gen_kwargs.get("max_new_tokens", DEFAULT_GEN_KWARGS["max_new_tokens"])),
        }

        if gen_kwargs.get("temperature") is not None:
            generation_config["temperature"] = gen_kwargs["temperature"]
        elif not gen_kwargs.get("do_sample", True):
            generation_config["temperature"] = 0

        if gen_kwargs.get("top_p") is not None:
            generation_config["topP"] = gen_kwargs["top_p"]

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

    def _extract_response_text(self, response_data):
        chunks = []
        for candidate in response_data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    chunks.append(part["text"])
        return "".join(chunks).strip()

    def _cache_key(self, contexts, gen_kwargs, media_type, media_paths, frame_indices=None):
        key_payload = {
            "model": self.model_version,
            "contexts": contexts,
            "gen_kwargs": gen_kwargs,
            "media_type": media_type,
            "media_paths": [str(path) for path in media_paths],
            "frame_indices": frame_indices,
            "jpeg_quality": self.jpeg_quality,
            "video_sampling_strategy": self.video_sampling_strategy,
            "include_frame_metadata": self.include_frame_metadata,
        }
        return hashlib.sha256(json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _save_cache(self):
        temp_path = f"{self.cache_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.response_cache, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.cache_path)

    def get_token_usage_summary(self):
        return self.token_usage_tracker.summary()

    def _call_api(self, payload, media_type=None):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
                response_data = response.json()
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response_data}")
                self.token_usage_tracker.record(extract_gemini_usage(response_data), media_type=media_type)
                return self._extract_response_text(response_data)
            except Exception as error:
                last_error = error
                eval_logger.warning(f"Gemini 3.1 Pro Preview API attempt {attempt + 1}/{self.max_retries} failed: {error}")
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_sleep)

        eval_logger.error(f"Gemini 3.1 Pro Preview API failed after {self.max_retries} attempts: {last_error}")
        return ""

    def generate_until(self, requests) -> List[str]:
        self.token_usage_tracker.reset()
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            gen_kwargs = self._normalize_gen_kwargs(gen_kwargs)
            media_type, media_inputs = self._get_media_inputs(doc_to_visual(self.task_dict[task][split][doc_id]))

            frame_indices = None
            if media_type == "image":
                pbar.set_postfix_str(f"Image: {self._short_path(media_inputs[0])}")
                parts = self._build_image_parts(media_inputs, contexts)
            elif media_type == "video":
                assert len(media_inputs) == 1, f"Only one video is supported, got {len(media_inputs)}."
                video_path = str(media_inputs[0])
                pbar.set_postfix_str(f"Video: {self._short_path(video_path)}")
                frame_indices, frames = self._sample_video_frames(video_path, doc_id, task, split)
                parts = self._build_frame_video_parts(frame_indices, frames, contexts)
            else:
                raise NotImplementedError(f"Unsupported media_type: {media_type}")

            cache_key = self._cache_key(contexts, gen_kwargs, media_type, media_inputs, frame_indices)
            if cache_key in self.response_cache:
                response_text = self.response_cache[cache_key]
            else:
                response_text = self._call_api(self._build_payload(parts, gen_kwargs), media_type=media_type)
                if response_text:
                    self.response_cache[cache_key] = response_text
                    self._save_cache()

            res.append(response_text)
            del parts
            gc.collect()
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Gemini 3.1 Pro Preview API loglikelihood is not implemented.")
