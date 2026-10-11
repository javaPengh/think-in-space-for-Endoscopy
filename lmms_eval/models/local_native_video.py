"""Offline Transformers adapters reusing Qwen3-VL's evaluation and sampling protocol."""

import gc
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger
from PIL import Image
from transformers import AutoProcessor

from lmms_eval.api.answer_explanation import AnswerExplanation
from lmms_eval.api.model import lmms
from lmms_eval.models.model_utils.blind_eval import is_blind_mode, normalize_visual_input_mode
from lmms_eval.models.model_utils.internvl_memory import inference_max_memory
from lmms_eval.models.qwen3vl import Qwen3VL

MODEL_PARALLEL_DEVICE_MAPS = {"auto", "balanced", "balanced_low_0", "sequential"}


def local_checkpoint(pretrained, default):
    path = Path(os.path.expandvars(os.path.expanduser(str(pretrained or default)))).resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Local checkpoint missing at {path}. Download the complete ModelScope snapshot first, or pass pretrained=/absolute/local/path. Hugging Face downloads are disabled.")
    return str(path)


class LocalNativeVideoModel(Qwen3VL):
    """Native image/video processors with explicit, pre-sampled frame metadata."""

    EXPECTED_MODEL_TYPE = None

    def __init__(
        self,
        pretrained=None,
        modality="video",
        device="cuda:0",
        device_map="auto",
        batch_size=1,
        max_frames_num=32,
        max_pixels=None,
        video_sampling_strategy="uniform",
        video_sample_fps=None,
        keyframe_mapping_path="data/keyframe_mapping.json",
        visual_input_mode="visual",
        save_sample_frames=False,
        explain_answer=False,
        explanation_max_new_tokens=2048,
        attn_implementation="sdpa",
        gpu_memory_reserve_gib=4,
        enable_thinking=False,
        **kwargs,
    ):
        lmms.__init__(self)
        if kwargs:
            raise TypeError(f"Unsupported {self.MODEL_DISPLAY_NAME} model_args: {sorted(kwargs)}")
        if int(batch_size) != 1:
            raise ValueError(f"{self.MODEL_DISPLAY_NAME} requires batch_size=1.")
        self.batch_size_per_gpu = 1
        self.answer_explanation = AnswerExplanation(explain_answer, explanation_max_new_tokens)
        self.enable_thinking = self._bool_option(enable_thinking)
        self.visual_input_mode = normalize_visual_input_mode(visual_input_mode)
        self.save_sample_frames = self._bool_option(save_sample_frames)
        self.modality = modality
        self.max_frames_num = int(max_frames_num)
        if self.max_frames_num < 1:
            raise ValueError("max_frames_num must be positive.")
        self.max_pixels = None if max_pixels in (None, "", "none", "None") else int(max_pixels)
        if self.max_pixels is not None and self.max_pixels < 1:
            raise ValueError("max_pixels must be positive.")
        self.video_sample_fps = None if video_sample_fps in (None, "", "none", "None") else float(video_sample_fps)
        self.video_sampling_strategy = str(video_sampling_strategy).lower()
        if self.video_sample_fps is not None and self.video_sampling_strategy == "uniform":
            self.video_sampling_strategy = "fps"
        if self.video_sampling_strategy in {"fps_uniform", "uniform_fps"}:
            self.video_sampling_strategy = "fps"
        if not is_blind_mode(self.visual_input_mode):
            if self.video_sampling_strategy not in {"uniform", "specific", "fps"}:
                raise ValueError(f"Unsupported video_sampling_strategy: {self.video_sampling_strategy}")
            if self.video_sampling_strategy == "fps" and (self.video_sample_fps is None or not math.isfinite(self.video_sample_fps) or self.video_sample_fps <= 0):
                raise ValueError("video_sample_fps must be a finite positive number.")
        self.keyframe_mapping = {}
        if self.video_sampling_strategy == "specific" and not is_blind_mode(self.visual_input_mode):
            with open(keyframe_mapping_path, encoding="utf-8") as handle:
                self.keyframe_mapping = json.load(handle)

        self.path = local_checkpoint(pretrained, self.DEFAULT_PRETRAINED)
        with open(Path(self.path) / "config.json", encoding="utf-8") as handle:
            model_type = json.load(handle).get("model_type")
        if model_type != self.EXPECTED_MODEL_TYPE:
            raise ValueError(f"{self.MODEL_DISPLAY_NAME} expects model_type={self.EXPECTED_MODEL_TYPE!r}, got {model_type!r} at {self.path}.")
        model_class = self._resolve_model_class()
        self.accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(weeks=52))])
        parallel = device_map in MODEL_PARALLEL_DEVICE_MAPS
        if parallel and self.accelerator.num_processes != 1:
            raise ValueError("Model parallel loading requires --num_processes 1. Select GPUs through CUDA_VISIBLE_DEVICES.")
        self._device = torch.device(device if parallel or self.accelerator.num_processes == 1 else f"cuda:{self.accelerator.local_process_index}")
        self.device_map = device_map if parallel else str(self._device)
        self._rank = self.accelerator.process_index
        self._world_size = self.accelerator.num_processes
        model_kwargs = {"torch_dtype": torch.bfloat16, "local_files_only": True, "low_cpu_mem_usage": True, "device_map": self.device_map}
        if attn_implementation not in (None, "", "none", "None"):
            model_kwargs["attn_implementation"] = str(attn_implementation)
        if parallel:
            model_kwargs["max_memory"] = inference_max_memory(gpu_memory_reserve_gib)
        self._model = model_class.from_pretrained(self.path, **model_kwargs).eval()
        self._processor = AutoProcessor.from_pretrained(self.path, local_files_only=True)
        # Use the embedding execution device when Accelerate dispatches/offloads weights.
        embedding = self.model.get_input_embeddings()
        hook = getattr(embedding, "_hf_hook", None)
        input_device = getattr(hook, "execution_device", None)
        if input_device is None:
            input_device = embedding.weight.device
        self._device = torch.device(f"cuda:{input_device}" if isinstance(input_device, int) else input_device)
        if self._device.type == "meta":
            raise RuntimeError("Could not determine the input execution device from the dispatched model.")
        eval_logger.info(f"Loaded {self.MODEL_DISPLAY_NAME} from {self.path}; input device={self._device}, device_map={getattr(self.model, 'hf_device_map', self.device_map)}")
        self.sample_frames_version = None
        self.token_usage_path = str(Path(__file__).resolve().parents[2] / "docs" / self.TOKEN_USAGE_FILENAME)
        self._last_aggregated_token_usage = None
        self._reset_token_usage()
        self.debug_prompt_dir = os.getenv("VSI_DEBUG_PROMPT_DIR")
        self.debug_prompt_limit = int(os.getenv("VSI_DEBUG_PROMPT_LIMIT", "5")) if self.debug_prompt_dir else 0
        self._debug_prompt_count = 0

    @staticmethod
    def _bool_option(value):
        value = str(value).strip().lower()
        if value not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("Boolean model_args must be true or false.")
        return value in {"true", "1", "yes"}

    def _resolve_model_class(self):
        model_class = getattr(transformers, self.MODEL_CLASS_NAME, None)
        if model_class is None:
            raise ImportError(
                f"{self.MODEL_DISPLAY_NAME} needs Transformers with {self.MODEL_CLASS_NAME}; installed version is {transformers.__version__}. Prepare a separate compatible environment; Transformers 4.57.x cannot load Qwen3.5/Gemma 4."
            )
        return model_class

    def _build_video_content(self, video_frames, effective_fps, source_fps, total_frames, frame_indices):
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Source video fps must be finite and positive.")
        return {
            "type": "video",
            "video": video_frames,
            "video_metadata": {"fps": source_fps, "total_num_frames": total_frames, "frames_indices": list(frame_indices)},
        }

    @staticmethod
    def _rgb_image(value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        with Image.open(value) as image:
            return image.convert("RGB")

    def _processor_options(self, images, videos):
        return {}

    def _sample_frame_indices(self, vr, total_frames, doc_id, task, split):
        source_fps = float(vr.get_avg_fps())
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Source video fps must be finite and positive.")
        return super()._sample_frame_indices(vr, total_frames, doc_id, task, split)

    def _prepare_inputs(self, messages):
        images, videos, metadata = [], [], []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    images.append(self._rgb_image(content["image"]))
                elif content["type"] == "video":
                    frames = [np.asarray(self._rgb_image(frame)) for frame in content["video"]]
                    videos.append(np.stack(frames))
                    metadata.append(dict(content["video_metadata"], frames_indices=list(content["video_metadata"]["frames_indices"])))
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        options = {"text": [text], "padding": True, "return_tensors": "pt"}
        if images:
            options["images"] = images
        if videos:
            options.update(videos=videos, video_metadata=metadata, do_sample_frames=False)
        options.update(self._processor_options(images, videos))
        inputs = self._processor(**options).to(self._device)
        # Preserve integral token ids/masks while matching vision inputs to bf16 weights.
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(dtype=torch.bfloat16)
        return text, inputs

    def _generate_from_messages(self, messages, gen_kwargs, media_type, debug_metadata=None):
        text, inputs = self._prepare_inputs(messages)
        self._debug_dump_prompt(messages, text, gen_kwargs, media_type, debug_metadata)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        outputs = generated[:, inputs.input_ids.shape[-1] :]
        response = self._processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self._record_token_usage(media_type, inputs.input_ids.numel(), outputs.numel())
        del inputs, generated, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return response
