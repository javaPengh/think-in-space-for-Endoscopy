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
from lmms_eval.models.model_utils.blind_eval import is_blind_mode, normalize_visual_input_mode, strip_visual_context
from lmms_eval.models.model_utils.question_id import get_question_key
from lmms_eval.models.model_utils.token_usage import TokenUsageTracker, extract_openai_responses_usage
from loguru import logger as eval_logger

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=1024,
    do_sample=False,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@register_model("gpt5_4")
class GPT5_4(lmms):
    def __init__(
        self,
        model_version: str = "gpt-5.4",
        modality: str = "video",
        max_frames_num: int = 16,
        image_detail: str = "auto",
        video_sampling_strategy: str = "uniform",
        keyframe_mapping_path: str = "data/keyframe_mapping.json",
        api_key: str = None,
        api_key_env: str = "OPENAI_API_KEY",
        api_url: str = None,
        base_url: str = None,
        timeout: int = 180,
        max_retries: int = 5,
        retry_sleep: int = 30,
        cache_dir: str = "api_cache",
        jpeg_quality: int = 95,
        reasoning_effort: str = None,
        include_frame_metadata: bool = True,
        visual_input_mode: str = "visual",
        **kwargs,
    ):
        super().__init__()

        self.model_version = model_version
        self.modality = modality
        self.visual_input_mode = normalize_visual_input_mode(visual_input_mode)
        self.max_frames_num = int(max_frames_num)
        self.image_detail = image_detail
        self.video_sampling_strategy = video_sampling_strategy
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.retry_sleep = int(retry_sleep)
        self.jpeg_quality = int(jpeg_quality)
        self.reasoning_effort = None if reasoning_effort is None or str(reasoning_effort).lower() in {"", "none", "null"} else str(reasoning_effort)
        self.include_frame_metadata = str(include_frame_metadata).lower() not in {"false", "0", "no"}
        self.keyframe_mapping = {}
        self.token_usage_tracker = TokenUsageTracker()

        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing OpenAI API key. Set {api_key_env} or pass api_key in model_args.")

        base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_url = api_url or os.getenv("OPENAI_API_URL") or f"{base_url.rstrip('/')}/responses"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.video_sampling_strategy == "specific" and not is_blind_mode(self.visual_input_mode):
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
            raise ValueError(f"Unsupported media_type for GPT-5.4: {media_type}")
        return media_type, media_path

    def _image_to_data_url(self, image):
        image = image.convert("RGB")
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG", quality=self.jpeg_quality)
        encoded = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _path_to_data_url(self, image_path):
        with Image.open(image_path) as image:
            return self._image_to_data_url(image)

    def _media_to_data_url(self, media):
        if isinstance(media, Image.Image):
            return self._image_to_data_url(media)
        return self._path_to_data_url(media)

    def _frame_to_data_url(self, frame):
        return self._image_to_data_url(Image.fromarray(frame).convert("RGB"))

    def _get_question_key(self, doc_id, task, split):
        return get_question_key(self.task_dict[task][split][doc_id], doc_id)

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

    def _input_image(self, data_url):
        return {"type": "input_image", "image_url": data_url, "detail": self.image_detail}

    def _build_image_content(self, media_path, contexts):
        return [
            self._input_image(self._media_to_data_url(media_path)),
            {"type": "input_text", "text": contexts},
        ]

    def _build_video_content(self, frame_indices, frames, contexts):
        content = []
        if self.include_frame_metadata:
            sampling_note = "The following images are sampled frames from one video, ordered by time."
            if self.video_sampling_strategy == "specific":
                sampling_note += " They are selected keyframes and may not be uniformly spaced in time."
            content.append({"type": "input_text", "text": sampling_note})

        for position, (source_index, frame) in enumerate(zip(frame_indices, frames), start=1):
            if self.include_frame_metadata:
                content.append({"type": "input_text", "text": f"Frame {position}, source frame index {source_index}."})
            content.append(self._input_image(self._frame_to_data_url(frame)))

        content.append({"type": "input_text", "text": contexts})
        return content

    def _build_text_content(self, contexts):
        return [{"type": "input_text", "text": contexts}]

    def _normalize_gen_kwargs(self, gen_kwargs):
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.pop("until", None)
        gen_kwargs.pop("num_beams", None)
        for key, value in DEFAULT_GEN_KWARGS.items():
            gen_kwargs.setdefault(key, value)
        if not gen_kwargs.get("do_sample", True):
            gen_kwargs.pop("top_p", None)
        return gen_kwargs

    def _build_payload(self, content, gen_kwargs):
        payload = {
            "model": self.model_version,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": int(gen_kwargs.get("max_new_tokens", DEFAULT_GEN_KWARGS["max_new_tokens"])),
        }

        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        if gen_kwargs.get("temperature") is not None:
            payload["temperature"] = gen_kwargs["temperature"]
        elif not gen_kwargs.get("do_sample", True):
            payload["temperature"] = 0

        if gen_kwargs.get("top_p") is not None:
            payload["top_p"] = gen_kwargs["top_p"]

        return payload

    def _extract_response_text(self, response_data):
        if response_data.get("output_text"):
            return str(response_data["output_text"]).strip()

        chunks = []
        for item in response_data.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and "text" in content:
                    chunks.append(str(content["text"]))
                elif content.get("type") == "refusal" and "refusal" in content:
                    chunks.append(str(content["refusal"]))
        return "".join(chunks).strip()

    def _cache_key(self, contexts, gen_kwargs, media_type, media_path, frame_indices=None):
        key_payload = {
            "model": self.model_version,
            "contexts": contexts,
            "gen_kwargs": gen_kwargs,
            "media_type": media_type,
            "media_path": str(media_path),
            "frame_indices": frame_indices,
            "image_detail": self.image_detail,
            "jpeg_quality": self.jpeg_quality,
            "reasoning_effort": self.reasoning_effort,
            "include_frame_metadata": self.include_frame_metadata,
            "visual_input_mode": self.visual_input_mode,
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
                self.token_usage_tracker.record(extract_openai_responses_usage(response_data), media_type=media_type)
                return self._extract_response_text(response_data)
            except Exception as error:
                last_error = error
                eval_logger.warning(f"GPT-5.4 API attempt {attempt + 1}/{self.max_retries} failed: {error}")
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_sleep)

        eval_logger.error(f"GPT-5.4 API failed after {self.max_retries} attempts: {last_error}")
        return ""

    def generate_until(self, requests) -> List[str]:
        self.token_usage_tracker.reset()
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            gen_kwargs = self._normalize_gen_kwargs(gen_kwargs)
            if is_blind_mode(self.visual_input_mode):
                contexts = strip_visual_context(contexts)
                media_type, media_path = "text", None
            else:
                media_type, media_path = self._get_media_info(doc_to_visual(self.task_dict[task][split][doc_id]))

            frame_indices = None
            if media_type == "text":
                pbar.set_postfix_str("Text")
                content = self._build_text_content(contexts)
            elif media_type == "image":
                unique_image_name = os.path.join(*str(media_path).split(os.sep)[-3:]) if not isinstance(media_path, Image.Image) else "PIL.Image"
                pbar.set_postfix_str(f"Image: {unique_image_name}")
                content = self._build_image_content(media_path, contexts)
            elif media_type == "video":
                video_path = str(media_path)
                unique_video_name = os.path.join(*video_path.split(os.sep)[-3:])
                pbar.set_postfix_str(f"Video: {unique_video_name}")
                frame_indices, frames = self._sample_video_frames(video_path, doc_id, task, split)
                content = self._build_video_content(frame_indices, frames, contexts)
            else:
                raise NotImplementedError(f"Unsupported media_type: {media_type}")

            cache_key = self._cache_key(contexts, gen_kwargs, media_type, media_path, frame_indices)
            if cache_key in self.response_cache:
                response_text = self.response_cache[cache_key]
            else:
                response_text = self._call_api(self._build_payload(content, gen_kwargs), media_type=media_type)
                if response_text:
                    self.response_cache[cache_key] = response_text
                    self._save_cache()

            res.append(response_text)
            del content
            gc.collect()
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("GPT-5.4 API loglikelihood is not implemented.")
