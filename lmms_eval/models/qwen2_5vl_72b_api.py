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
from lmms_eval.models.model_utils.token_usage import TokenUsageTracker, extract_openai_chat_usage
from loguru import logger as eval_logger

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=1024,
    do_sample=False,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@register_model("qwen2_5vl_72b_api")
class Qwen2_5VL_72B_API(lmms):
    def __init__(
        self,
        model_version: str = "qwen2.5-vl-72b-instruct",
        modality: str = "video",
        max_frames_num: int = 32,
        video_sampling_strategy: str = "uniform",
        keyframe_mapping_path: str = "data/keyframe_mapping.json",
        api_key: str = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        api_url: str = None,
        base_url: str = None,
        timeout: int = 180,
        max_retries: int = 5,
        retry_sleep: int = 30,
        cache_dir: str = "api_cache",
        jpeg_quality: int = 95,
        video_fps: float = 1.0,
        video_sample_fps: float = None,
        video_input_mode: str = "frames",
        dashscope_base_http_api_url: str = None,
        visual_input_mode: str = "visual",
        **kwargs,
    ):
        super().__init__()

        self.model_version = model_version
        self.modality = modality
        self.visual_input_mode = normalize_visual_input_mode(visual_input_mode)
        self.max_frames_num = int(max_frames_num)
        self.video_fps = float(video_fps)
        self.video_sampling_strategy = video_sampling_strategy
        self.video_sample_fps = None if video_sample_fps in (None, "") else float(video_sample_fps)
        self.video_input_mode = video_input_mode
        self.dashscope_base_http_api_url = dashscope_base_http_api_url or os.getenv("DASHSCOPE_BASE_HTTP_API_URL")
        if not is_blind_mode(self.visual_input_mode):
            if self.video_input_mode not in {"frames", "file"}:
                raise ValueError(f"Unsupported video_input_mode for Qwen2.5-VL-72B API: {self.video_input_mode}")
            if self.video_input_mode == "file" and self.video_sampling_strategy == "uniform" and self.video_sample_fps is None:
                self.video_sampling_strategy = "fps"
                self.video_sample_fps = self.video_fps
            if self.video_sampling_strategy not in {"uniform", "specific", "fps"}:
                raise ValueError(f"Unsupported video_sampling_strategy for Qwen2.5-VL-72B API: {self.video_sampling_strategy}")
            if self.video_sampling_strategy == "fps" and (self.video_sample_fps is None or self.video_sample_fps <= 0):
                raise ValueError("video_sample_fps must be a positive number when video_sampling_strategy is 'fps'.")
            if self.video_input_mode == "file" and self.video_sampling_strategy == "specific":
                raise ValueError("video_input_mode='file' cannot use specific keyframes because server-side video upload only accepts fps-based sampling.")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.jpeg_quality = int(jpeg_quality)
        self.keyframe_mapping = {}
        self.token_usage_tracker = TokenUsageTracker()

        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing DashScope API key. Set {api_key_env} or pass api_key in model_args.")

        base_url = base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.api_url = api_url or os.getenv("DASHSCOPE_API_URL") or f"{base_url.rstrip('/')}/chat/completions"
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
            raise ValueError(f"Unsupported media_type for Qwen2.5-VL-72B API: {media_type}")
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
            sample_count = min(max(self.max_frames_num, 4), 512)
            frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int).tolist()

        frame_indices = [min(max(0, int(index)), total_frames - 1) for index in frame_indices[:512]]
        if len(frame_indices) < 4:
            eval_logger.warning(f"Qwen2.5-VL video-list input requires at least 4 frames; padding {video_path} from {len(frame_indices)} to 4 frames.")
            frame_indices.extend([frame_indices[-1]] * (4 - len(frame_indices)))

        frames = vr.get_batch(frame_indices).asnumpy()
        return frame_indices, frames

    def _build_payload(self, content, gen_kwargs):
        payload = {
            "model": self.model_version,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(gen_kwargs.get("max_new_tokens", DEFAULT_GEN_KWARGS["max_new_tokens"])),
        }

        if gen_kwargs.get("temperature") is not None:
            payload["temperature"] = gen_kwargs["temperature"]
        elif not gen_kwargs.get("do_sample", True):
            payload["temperature"] = 0

        if gen_kwargs.get("top_p") is not None:
            payload["top_p"] = gen_kwargs["top_p"]

        return payload

    def _extract_response_text(self, response_data):
        content = response_data["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    chunks.append(str(item["text"]))
                elif isinstance(item, str):
                    chunks.append(item)
            return "".join(chunks).strip()
        return str(content).strip()

    def _get_obj_value(self, obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _extract_dashscope_response_text(self, response):
        output = self._get_obj_value(response, "output", {})
        choices = self._get_obj_value(output, "choices", []) or []
        if not choices:
            return ""
        message = self._get_obj_value(choices[0], "message", {})
        content = self._get_obj_value(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks = []
            for item in content:
                text = self._get_obj_value(item, "text")
                if text is not None:
                    chunks.append(str(text))
                elif isinstance(item, str):
                    chunks.append(item)
            return "".join(chunks).strip()
        return str(content).strip()

    def _extract_dashscope_usage(self, response):
        usage = self._get_obj_value(response, "usage", {}) or {}
        return {
            "input_tokens": self._get_obj_value(usage, "input_tokens") or self._get_obj_value(usage, "prompt_tokens"),
            "output_tokens": self._get_obj_value(usage, "output_tokens") or self._get_obj_value(usage, "completion_tokens"),
            "total_tokens": self._get_obj_value(usage, "total_tokens"),
        }

    def _cache_key(self, contexts, gen_kwargs, media_type, media_path, frame_indices=None):
        key_payload = {
            "model": self.model_version,
            "contexts": contexts,
            "gen_kwargs": gen_kwargs,
            "media_type": media_type,
            "media_path": str(media_path),
            "frame_indices": frame_indices,
            "video_sampling_strategy": self.video_sampling_strategy,
            "video_sample_fps": self.video_sample_fps,
            "video_input_mode": self.video_input_mode,
            "jpeg_quality": self.jpeg_quality,
            "video_fps": self.video_fps,
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
                self.token_usage_tracker.record(extract_openai_chat_usage(response_data), media_type=media_type)
                return self._extract_response_text(response_data)
            except Exception as error:
                last_error = error
                eval_logger.warning(f"Qwen2.5-VL API attempt {attempt + 1}/{self.max_retries} failed: {error}")
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_sleep)

        eval_logger.error(f"Qwen2.5-VL API failed after {self.max_retries} attempts: {last_error}")
        return ""

    def _build_dashscope_messages(self, video_path, contexts, fps):
        local_video_path = os.path.abspath(video_path)
        return [
            {
                "role": "user",
                "content": [
                    {"video": f"file://{local_video_path}", "fps": fps},
                    {"text": contexts},
                ],
            }
        ]

    def _call_dashscope_sdk(self, messages, gen_kwargs, media_type=None):
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError as error:
            raise ImportError("video_input_mode='file' requires the dashscope Python package on the evaluation server.") from error

        if self.dashscope_base_http_api_url:
            dashscope.base_http_api_url = self.dashscope_base_http_api_url

        call_kwargs = {
            "api_key": self.api_key,
            "model": self.model_version,
            "messages": messages,
        }
        call_kwargs["max_tokens"] = int(gen_kwargs.get("max_new_tokens", DEFAULT_GEN_KWARGS["max_new_tokens"]))
        if gen_kwargs.get("temperature") is not None:
            call_kwargs["temperature"] = gen_kwargs["temperature"]
        elif not gen_kwargs.get("do_sample", True):
            call_kwargs["temperature"] = 0
        if gen_kwargs.get("top_p") is not None:
            call_kwargs["top_p"] = gen_kwargs["top_p"]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = MultiModalConversation.call(**call_kwargs)
                status_code = self._get_obj_value(response, "status_code")
                if status_code is not None and int(status_code) >= 400:
                    code = self._get_obj_value(response, "code", "")
                    message = self._get_obj_value(response, "message", "")
                    raise RuntimeError(f"HTTP {status_code}: {code} {message}")
                self.token_usage_tracker.record(self._extract_dashscope_usage(response), media_type=media_type)
                return self._extract_dashscope_response_text(response)
            except Exception as error:
                last_error = error
                eval_logger.warning(f"Qwen2.5-VL DashScope SDK attempt {attempt + 1}/{self.max_retries} failed: {error}")
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_sleep)

        eval_logger.error(f"Qwen2.5-VL DashScope SDK failed after {self.max_retries} attempts: {last_error}")
        return ""

    def _normalize_gen_kwargs(self, gen_kwargs):
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.pop("until", None)
        gen_kwargs.pop("num_beams", None)
        for key, value in DEFAULT_GEN_KWARGS.items():
            gen_kwargs.setdefault(key, value)
        if not gen_kwargs.get("do_sample", True):
            gen_kwargs.pop("top_p", None)
        return gen_kwargs

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
                content = [{"type": "text", "text": contexts}]
            elif media_type == "image":
                unique_image_name = os.path.join(*str(media_path).split(os.sep)[-3:])
                pbar.set_postfix_str(f"Image: {unique_image_name}")
                content = [
                    {"type": "image_url", "image_url": {"url": self._path_to_data_url(media_path)}},
                    {"type": "text", "text": contexts},
                ]
            elif media_type == "video":
                video_path = str(media_path)
                unique_video_name = os.path.join(*video_path.split(os.sep)[-3:])
                pbar.set_postfix_str(f"Video: {unique_video_name}")
                payload_fps = self.video_sample_fps if self.video_sampling_strategy == "fps" else self.video_fps
                if self.video_input_mode == "file":
                    content = self._build_dashscope_messages(video_path, contexts, payload_fps)
                else:
                    frame_indices, frames = self._sample_video_frames(video_path, doc_id, task, split)
                    content = [
                        {
                            "type": "video",
                            "video": [self._frame_to_data_url(frame) for frame in frames],
                            "fps": payload_fps,
                        },
                        {"type": "text", "text": contexts},
                    ]
            else:
                raise NotImplementedError(f"Unsupported media_type: {media_type}")

            cache_key = self._cache_key(contexts, gen_kwargs, media_type, media_path, frame_indices)
            if cache_key in self.response_cache:
                response_text = self.response_cache[cache_key]
            else:
                if media_type == "video" and self.video_input_mode == "file":
                    response_text = self._call_dashscope_sdk(content, gen_kwargs, media_type=media_type)
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
        raise NotImplementedError("Qwen2.5-VL-72B API loglikelihood is not implemented.")
