import copy
import json
import logging
import math
import os
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.blind_eval import is_blind_mode, normalize_visual_input_mode, strip_visual_context
from lmms_eval.models.model_utils.load_video import read_video_pyav
from lmms_eval.models.model_utils.question_id import get_question_key

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
QWEN2VL_DEFAULT_VIDEO_MAX_PIXELS = 28 * 28 * 768
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_onevision_1_5")
class Llava_OneVision_1_5(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "vicuna_v1",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        max_pixels: Optional[int] = 602112,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        video_sampling_strategy: str = "uniform",
        video_sample_fps: Optional[float] = None,
        keyframe_mapping_path: str = "data/keyframe_mapping.json",
        processor_use_fast: Optional[bool] = False,
        visual_input_mode: str = "visual",
        save_sample_frames: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if "video_sampling_strategy" in kwargs:
            kwargs.pop("video_sampling_strategy")
        if "keyframe_mapping_path" in kwargs:
            kwargs.pop("keyframe_mapping_path")
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.visual_input_mode = normalize_visual_input_mode(visual_input_mode)
        self.save_sample_frames = str(save_sample_frames).strip().lower() not in {"0", "false", "no", "none", ""}
        self.video_sampling_strategy = video_sampling_strategy
        self.video_sample_fps = None if video_sample_fps in (None, "") else float(video_sample_fps)
        if not is_blind_mode(self.visual_input_mode):
            if self.video_sampling_strategy not in {"uniform", "specific", "fps"}:
                raise ValueError(f"Unsupported video_sampling_strategy for LLaVA-OneVision-1.5: {self.video_sampling_strategy}")
            if self.video_sampling_strategy == "fps" and (self.video_sample_fps is None or self.video_sample_fps <= 0):
                raise ValueError("video_sample_fps must be a positive number when video_sampling_strategy is 'fps'.")
        self.keyframe_mapping = {}
        self.sample_frames_version = None
        if self.video_sampling_strategy == "specific" and not is_blind_mode(self.visual_input_mode):
            import os

            if not os.path.exists(keyframe_mapping_path):
                raise ValueError(f"Keyframe mapping file not found at {keyframe_mapping_path}. Required when video_sampling_strategy is 'specific'.")
            with open(keyframe_mapping_path, "r", encoding="utf-8") as f:
                self.keyframe_mapping = json.load(f)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {}
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.max_pixels = int(max_pixels) if max_pixels is not None else None
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend
        self._logged_first_video_sample = False
        self._logged_first_model_inputs = False

        request_flash_attention = attn_implementation == "flash_attention_2" or bool(llava_model_args.get("use_flash_attention_2"))
        self._ensure_flash_attn_varlen_func_compat(request_flash_attention=request_flash_attention)
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
            **llava_model_args,
        )
        processor_kwargs = {"trust_remote_code": True, "use_fast": bool(processor_use_fast)}
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self._processor = AutoProcessor.from_pretrained(pretrained, **processor_kwargs)
        self._apply_processor_max_pixels()
        if self.max_pixels == QWEN2VL_DEFAULT_VIDEO_MAX_PIXELS:
            eval_logger.info(
                "LLaVA-OneVision-1.5 max_pixels=%s matches Qwen2-VL's default video max_pixels; lower it to reduce video visual tokens.",
                self.max_pixels,
            )
        if hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.padding_side = "left"
            self._tokenizer = self._processor.tokenizer
        else:
            raise ValueError("AutoProcessor did not provide a tokenizer for the LLaVA-OneVision model.")

        self._config = self._model.config
        self._max_length = getattr(self._tokenizer, "model_max_length", None)
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.chat_template = conv_template if conv_template and "{%" in conv_template else None
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation."

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

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def _apply_chat_template(self, messages, add_generation_prompt: bool):
        if self.chat_template is not None:
            self.tokenizer.chat_template = self.chat_template
        elif self.tokenizer.chat_template is None and getattr(self._processor, "chat_template", None) is None:
            self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
        if hasattr(self._processor, "apply_chat_template"):
            return self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def _strip_legacy_visual_tokens(self, text: str) -> str:
        return re.sub(r"(<image>|<video>)\s*", "", text).lstrip()

    def _build_user_content(self, text: str, visuals, task_type: str):
        if visuals is None:
            return text

        text = self._strip_legacy_visual_tokens(text)
        if task_type == "video":
            video_item = {"type": "video", "video": visuals}
            if self.max_pixels is not None:
                video_item["max_pixels"] = self.max_pixels
            return [video_item, {"type": "text", "text": text}]
        if task_type == "image":
            image_items = []
            for image in visuals:
                image_item = {"type": "image", "image": image}
                if self.max_pixels is not None:
                    image_item["max_pixels"] = self.max_pixels
                image_items.append(image_item)
            return image_items + [{"type": "text", "text": text}]

        return text

    def _apply_processor_max_pixels(self):
        if self.max_pixels is None:
            return

        for component_name in ("image_processor", "video_processor"):
            processor_component = getattr(self._processor, component_name, None)
            if processor_component is None:
                continue

            size = getattr(processor_component, "size", None)
            if isinstance(size, dict):
                size["longest_edge"] = self.max_pixels
            if hasattr(processor_component, "max_pixels"):
                processor_component.max_pixels = self.max_pixels

    def _vision_processor_kwargs(self, task_type: str):
        if self.max_pixels is None:
            return {}
        if task_type == "video":
            return {"videos_kwargs": {"max_pixels": self.max_pixels}}
        if task_type == "image":
            return {"images_kwargs": {"max_pixels": self.max_pixels}}
        return {}

    def _drop_unsupported_model_inputs(self, model_inputs):
        # LLaVA-OneVision-1.5 uses Qwen2.5-VL's processor, which may emit this
        # timing field, but the remote LLaVA model forward does not consume it.
        model_inputs.pop("second_per_grid_ts", None)
        return model_inputs

    def _log_model_inputs_once(self, model_inputs, task_type: str):
        if self._logged_first_model_inputs:
            return

        summary = {}
        for key, value in model_inputs.items():
            if hasattr(value, "shape"):
                summary[key] = tuple(value.shape)
            elif isinstance(value, (list, tuple)):
                summary[key] = f"{type(value).__name__}[{len(value)}]"
            else:
                summary[key] = type(value).__name__

        image_processor = getattr(self._processor, "image_processor", None)
        video_processor = getattr(self._processor, "video_processor", None)
        eval_logger.info(
            "First LLaVA-OneVision-1.5 %s model inputs: %s; processor=%s; image_processor=%s; video_processor=%s; max_pixels=%s",
            task_type,
            summary,
            type(self._processor).__name__,
            type(image_processor).__name__ if image_processor is not None else None,
            type(video_processor).__name__ if video_processor is not None else None,
            self.max_pixels,
        )
        self._logged_first_model_inputs = True

    def _flatten_visuals(self, visual):
        if visual is None or visual == []:
            return []
        if isinstance(visual, list):
            return visual
        return [visual]

    def _open_image(self, image_path):
        with PIL.Image.open(image_path) as image:
            return image.convert("RGB")

    def _load_visual_inputs(self, visual, doc_id, task, split):
        visual_items = self._flatten_visuals(visual)
        if not visual_items:
            return "text", None

        first_visual = visual_items[0]
        if isinstance(first_visual, PIL.Image.Image):
            return "image", visual_items

        if isinstance(first_visual, dict):
            media_type = str(first_visual.get("media_type", "")).lower()
            media_paths = [item.get("path") or item.get("media_path") for item in visual_items]
            if any(path is None for path in media_paths):
                raise ValueError(f"Missing media path in visual input: {visual_items}")
            if not media_type:
                media_type = "image" if os.path.splitext(str(media_paths[0]))[1].lower() in IMAGE_EXTENSIONS else "video"
        else:
            media_paths = visual_items
            media_type = "image" if isinstance(first_visual, str) and os.path.splitext(first_visual)[1].lower() in IMAGE_EXTENSIONS else "video"

        if media_type == "image":
            return "image", [self._open_image(path) if isinstance(path, str) else path for path in media_paths]
        if media_type != "video":
            raise ValueError(f"Unsupported media_type for LLaVA-OneVision-1.5: {media_type}")

        try:
            question_key = get_question_key(self.task_dict[task][split][doc_id], doc_id)

            if self.video_decode_backend == "decord" or self.video_sampling_strategy in {"specific", "fps"}:
                frames = self.load_video(media_paths, self.max_frames_num, question_key=question_key, task=task)
            else:
                frames = read_video_pyav(media_paths[0], num_frm=self.max_frames_num)
            return "video", self._ensure_pil_images(frames)
        except Exception as e:
            eval_logger.error(f"Error {e} in loading video")
            return "text", None

    def _ensure_pil_images(self, frames):
        pil_frames = []
        for frame in frames:
            if isinstance(frame, PIL.Image.Image):
                pil_frames.append(frame)
            else:
                pil_frames.append(PIL.Image.fromarray(frame))
        return pil_frames

    def _ensure_flash_attn_varlen_func_compat(self, request_flash_attention: bool):
        try:
            from transformers import modeling_flash_attention_utils
        except ImportError:
            return

        if hasattr(modeling_flash_attention_utils, "flash_attn_varlen_func"):
            return

        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func
        except ImportError as e:
            if request_flash_attention:
                raise ImportError("LLaVA-OneVision-1.5 expects flash_attn_varlen_func. " "Install a compatible flash-attn build or disable flash_attention_2.") from e
            modeling_flash_attention_utils.flash_attn_varlen_func = None
            return

        modeling_flash_attention_utils.flash_attn_varlen_func = flash_attn_varlen_func

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if is_blind_mode(self.visual_input_mode):
                contexts = strip_visual_context(contexts)
                visual = []
            else:
                visual = doc_to_visual(self.task_dict[task][split][doc_id])
            continuation = doc_to_target if isinstance(doc_to_target, str) else doc_to_target(self.task_dict[task][split][doc_id])

            task_type, visuals = self._load_visual_inputs(visual, doc_id, task, split)

            messages = [{"role": "user", "content": self._build_user_content(contexts, visuals, task_type)}, {"role": "assistant", "content": continuation}]
            prompt = self._apply_chat_template(messages[:-1], add_generation_prompt=True)
            prompt_and_continuation = self._apply_chat_template(messages, add_generation_prompt=False)

            if visuals is None:
                model_inputs = self._processor(text=[prompt_and_continuation], return_tensors="pt")
            elif task_type == "video":
                model_inputs = self._processor(text=[prompt_and_continuation], videos=[visuals], return_tensors="pt", **self._vision_processor_kwargs(task_type))
            else:
                model_inputs = self._processor(text=[prompt_and_continuation], images=visuals, return_tensors="pt", **self._vision_processor_kwargs(task_type))
            model_inputs = self._drop_unsupported_model_inputs(model_inputs)

            model_inputs = model_inputs.to(self.device, self.model.dtype)

            labels = model_inputs["input_ids"].clone()
            context_ids = self._processor(text=[prompt], return_tensors="pt")["input_ids"]
            labels[0, : context_ids.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = model_inputs["input_ids"][:, context_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, context_ids.shape[1] : model_inputs["input_ids"].shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num, question_key=None, task=None):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
            video_path = video_path[0]

        total_frame_num = len(vr)
        if total_frame_num <= 0:
            raise ValueError(f"Video has no frames: {video_path}")

        if self.video_sampling_strategy == "specific":
            found_indices = None
            if self.keyframe_mapping and question_key is not None:
                for dataset_name, questions in self.keyframe_mapping.items():
                    if question_key in questions:
                        found_indices = questions[question_key]
                        break
            if found_indices is None or len(found_indices) == 0:
                raise ValueError(f"Specific frame indices not found or empty for question ID: {question_key} in {task}")
            frame_idx = found_indices
        elif self.video_sampling_strategy == "fps":
            avg_fps = float(vr.get_avg_fps())
            if avg_fps <= 0:
                raise ValueError(f"Cannot use fps sampling because video avg_fps is invalid: {avg_fps}")
            duration = total_frame_num / avg_fps
            timestamps = np.arange(0, duration, 1.0 / self.video_sample_fps)
            if len(timestamps) == 0:
                timestamps = np.array([0.0])
            frame_idx = np.floor(timestamps * avg_fps).astype(int).tolist()
            frame_idx = list(dict.fromkeys(frame_idx))
            if max_frames_num is not None and len(frame_idx) > int(max_frames_num):
                keep_indices = np.linspace(0, len(frame_idx) - 1, int(max_frames_num), dtype=int)
                frame_idx = [frame_idx[i] for i in keep_indices]
        else:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        frame_idx = [min(max(0, int(idx)), total_frame_num - 1) for idx in frame_idx]

        if not self._logged_first_video_sample:
            eval_logger.info(
                "First LLaVA-OneVision-1.5 video sample: strategy=%s, video_sample_fps=%s, max_frames_num=%s, sampled_frames=%s, total_frames=%s, path=%s",
                self.video_sampling_strategy,
                self.video_sample_fps,
                max_frames_num,
                len(frame_idx),
                total_frame_num,
                video_path,
            )
            self._logged_first_video_sample = True

        spare_frames = vr.get_batch(frame_idx).asnumpy()

        if self.save_sample_frames:
            import re
            from pathlib import Path

            import cv2

            extracted_model_name = self.pretrained.split("/")[-1] if "/" in self.pretrained else self.pretrained
            if self.video_sampling_strategy == "fps":
                sampling_label = f"fps_{self.video_sample_fps:g}".replace(".", "p")
            elif self.video_sampling_strategy == "specific":
                sampling_label = "specific"
            else:
                sampling_label = f"{max_frames_num}f"
            base_model_dir = os.path.join("sample_frames", f"{extracted_model_name}-{sampling_label}")

            if self.sample_frames_version is None:
                env_key = f"SAMPLE_FRAMES_VERSION_{extracted_model_name}_{max_frames_num}"
                if env_key not in os.environ:
                    os.makedirs(base_model_dir, exist_ok=True)
                    existing_versions = []
                    for d in os.listdir(base_model_dir):
                        if os.path.isdir(os.path.join(base_model_dir, d)) and re.match(r"^v_\d+$", d):
                            existing_versions.append(int(d.split("_")[1]))
                    next_version = max(existing_versions) + 1 if existing_versions else 1
                    os.environ[env_key] = f"v_{next_version:02d}"
                self.sample_frames_version = os.environ[env_key]

            base_save_dir = os.path.join(base_model_dir, self.sample_frames_version)
            os.makedirs(base_save_dir, exist_ok=True)
            video_stem = Path(video_path).stem

            for i, idx in enumerate(frame_idx):
                try:
                    frame_file = f"{video_stem}_frame{idx:04d}.jpg"
                    save_path = os.path.join(base_save_dir, frame_file)
                    cv2.imwrite(save_path, cv2.cvtColor(spare_frames[i], cv2.COLOR_RGB2BGR))
                except Exception as e:
                    eval_logger.warning(f"Failed to save sampled frame {idx}: {e}")

        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            if is_blind_mode(self.visual_input_mode):
                batched_contexts = tuple(strip_visual_context(context) for context in batched_contexts)
                batched_visuals = [[] for _ in batched_doc_id]
            else:
                batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            visuals_for_batch = None
            task_type = "text"

            for visual, context in zip(batched_visuals, batched_contexts):
                task_type, visuals_for_batch = self._load_visual_inputs(visual, batched_doc_id[0], task, split)

            question = context
            if utils.is_json(question):
                question_items = json.loads(question)
                messages = []
                for idx, item in enumerate(question_items):
                    role = "user" if idx % 2 == 0 else "assistant"
                    item_value = strip_visual_context(item["value"]) if idx == 0 and role == "user" and is_blind_mode(self.visual_input_mode) else item["value"]
                    content = self._build_user_content(item_value, visuals_for_batch, task_type) if idx == 0 and role == "user" else item_value
                    messages.append({"role": role, "content": content})
                prompt_question = self._apply_chat_template(messages, add_generation_prompt=True)
            else:
                messages = [{"role": "user", "content": self._build_user_content(question, visuals_for_batch, task_type)}]
                prompt_question = self._apply_chat_template(messages, add_generation_prompt=True)

            question_input.append(prompt_question)

            if not gen_kwargs.get("do_sample", True):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)

            if visuals_for_batch is None:
                model_inputs = self._processor(text=question_input, return_tensors="pt")
            elif task_type == "video":
                # 传入 videos 参数
                model_inputs = self._processor(text=question_input, videos=[visuals_for_batch], return_tensors="pt", **self._vision_processor_kwargs(task_type))
            else:
                model_inputs = self._processor(text=question_input, images=visuals_for_batch, return_tensors="pt", **self._vision_processor_kwargs(task_type))
            model_inputs = self._drop_unsupported_model_inputs(model_inputs)
            self._log_model_inputs_once(model_inputs, task_type)

            model_inputs = model_inputs.to(self.device, self.model.dtype)

            with torch.inference_mode():
                cont = self.model.generate(**model_inputs, use_cache=self.use_cache, **gen_kwargs)

            cont = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs["input_ids"], cont)]
            decoder = self._processor if hasattr(self._processor, "batch_decode") else self.tokenizer
            text_outputs = decoder.batch_decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)

        pbar.close()
        return res
