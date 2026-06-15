import logging
import os
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.blind_eval import is_blind_mode, normalize_visual_input_mode, strip_visual_context
from lmms_eval.models.model_utils.question_id import get_question_key
from loguru import logger as eval_logger

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def build_transform(input_size):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image, input_size=448, max_num=6):
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as raw_image:
            image = raw_image.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
    else:
        start_idx = first_idx
        end_idx = max_frame

    return np.linspace(start_idx, end_idx, num_segments, dtype=int)


def load_video(
    video_path,
    bound=None,
    input_size=448,
    max_num=1,
    num_segments=32,
    model_name="unknown",
    version_dir=None,
    video_sampling_strategy="uniform",
    keyframe_mapping=None,
    question_key=None,
    task=None,
):
    vr = VideoReader(video_path, ctx=cpu(0))
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    
    if video_sampling_strategy == "specific":
        found_indices = None
        if keyframe_mapping and question_key is not None:
            for dataset_name, questions in keyframe_mapping.items():
                if question_key in questions:
                    found_indices = questions[question_key]
                    break
        if found_indices is None or len(found_indices) == 0:
            raise ValueError(f"Specific frame indices not found or empty for question ID: {question_key} in {task}")
        frame_indices = found_indices
    else:
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    # ====== 新增：持久化保存采样帧用于分析 ======
    import os
    from pathlib import Path
    import cv2
    import re
    
    # 构建保存目录: sample_frames/{model_name}-{num_segments}f/v_{xx}
    base_model_dir = os.path.join("sample_frames", f"{model_name}-{num_segments}f")
    
    # 如果没有显式传入 version_dir，则回退到原来的环境变量查找逻辑作为兼容
    if version_dir is None:
        env_key = f"SAMPLE_FRAMES_VERSION_{model_name}_{num_segments}"
        if env_key not in os.environ:
            os.makedirs(base_model_dir, exist_ok=True)
            existing_versions = []
            for d in os.listdir(base_model_dir):
                if os.path.isdir(os.path.join(base_model_dir, d)) and re.match(r"^v_\d+$", d):
                    existing_versions.append(int(d.split("_")[1]))
            
            next_version = max(existing_versions) + 1 if existing_versions else 1
            os.environ[env_key] = f"v_{next_version:02d}"
            
        version_dir = os.environ[env_key]
        
    base_save_dir = os.path.join(base_model_dir, version_dir)
    os.makedirs(base_save_dir, exist_ok=True)
    video_stem = Path(video_path).stem
    # ============================================

    for frame_index in frame_indices:
        frame_np = vr[frame_index].asnumpy()
        
        # 保存当前帧为图像
        try:
            frame_file = f"{video_stem}_frame{frame_index:04d}.jpg"
            save_path = os.path.join(base_save_dir, frame_file)
            # frame_np 是 RGB 格式，cv2 需要 BGR
            cv2.imwrite(save_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        except Exception as e:
            eval_logger.warning(f"Failed to save sampled frame {frame_index}: {e}")
            
        img = Image.fromarray(frame_np).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


@register_model("internvl3_5")
class InternVL3_5(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL-3.5-2B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        max_frames_num: int = 32,
        video_max_num: int = 1,
        video_sampling_strategy: str = "uniform",
        keyframe_mapping_path: str = "data/keyframe_mapping.json",
        visual_input_mode: str = "visual",
        **kwargs,
    ):
        super().__init__()
        self.visual_input_mode = normalize_visual_input_mode(visual_input_mode)
        self.path = pretrained
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.video_max_num = int(video_max_num)
        if self.video_max_num < 1:
            raise ValueError(f"video_max_num must be >= 1, got {video_max_num}.")
        self.sample_frames_version = None
        self.video_sampling_strategy = video_sampling_strategy
        self.keyframe_mapping = {}
        if self.video_sampling_strategy == "specific" and not is_blind_mode(self.visual_input_mode):
            import os
            if not os.path.exists(keyframe_mapping_path):
                raise ValueError(f"Keyframe mapping file not found at {keyframe_mapping_path}. Required when video_sampling_strategy is 'specific'.")
            import json
            with open(keyframe_mapping_path, "r", encoding="utf-8") as f:
                self.keyframe_mapping = json.load(f)

        if device_map == 'auto':
            self._model = AutoModel.from_pretrained(
                self.path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device_map,
            ).eval()
            try:
                from accelerate.hooks import add_hook_to_module, AlignDevicesHook
                add_hook_to_module(self._model.language_model.lm_head, AlignDevicesHook(execution_device=self._model.language_model.model.embed_tokens.weight.device))
            except Exception as e:
                eval_logger.debug(f"Could not add accelerate hook for internvl3.5: {e}")
        else:
            self._model = AutoModel.from_pretrained(
                self.path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval().cuda()

        self._config = AutoConfig.from_pretrained(self.path, trust_remote_code=True)

        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL3_5, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

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
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )

            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)

            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(
                f"Using {accelerator.num_processes} process with model parallel device_map."
            )
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

    def flatten(self, input_list):
        return [j for i in input_list for j in i]

    def _cast_and_move_pixels(self, pixel_values: torch.Tensor):
        return pixel_values.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)

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
            raise ValueError(f"Unsupported media_type for InternVL3_5: {media_type}")
        return media_type, media_inputs

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
        import gc

        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for idx, (contexts, gen_kwargs, doc_to_visual, doc_id, task, split) in enumerate(
            [reg.args for reg in requests]
        ):
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            for k, v in DEFAULT_GEN_KWARGS.items():
                gen_kwargs.setdefault(k, v)

            # Ensure pad_token_id to avoid transformers spam and potential mismatched padding behavior
            if "pad_token_id" not in gen_kwargs and self.tokenizer.eos_token_id is not None:
                gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

            pop_keys = [k for k in gen_kwargs if k not in DEFAULT_GEN_KWARGS and k != "pad_token_id"]
            for k in pop_keys:
                gen_kwargs.pop(k)

            if is_blind_mode(self.visual_input_mode):
                contexts = strip_visual_context(contexts)
                media_type, visuals = "text", []
            else:
                media_type, visuals = self._get_media_inputs(doc_to_visual(self.task_dict[task][split][doc_id]))
            if visuals:
                if media_type == "image":
                    image_tensors = [self._cast_and_move_pixels(load_image(v)) for v in visuals]
                    pixel_values = torch.cat(image_tensors, dim=0)
                    num_patches_list = [v.size(0) for v in image_tensors]
                    image_tokens = " ".join(["<image>"] * len(image_tensors))
                    question = image_tokens + "\n" + contexts

                    response, _ = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        question,
                        gen_kwargs,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=True,
                    )

                elif media_type == "video":
                    if self.sample_frames_version is None:
                        self._determine_sample_frames_version()
                    assert len(visuals) == 1, f"Only one video is supported, got {len(visuals)}."
                    video_path = visuals[0]
                    
                    # 提取模型名称用于建立对应目录，例如从 "OpenGVLab/InternVL-3.5-2B" 中提取 "InternVL-3.5-2B"
                    extracted_model_name = self.path.split("/")[-1] if "/" in self.path else self.path
                    
                    question_key = get_question_key(self.task_dict[task][split][doc_id], doc_id)
                    
                    pixel_values, num_patches_list = load_video(
                        video_path,
                        num_segments=self.max_frames_num,
                        max_num=self.video_max_num,
                        model_name=extracted_model_name,
                        version_dir=self.sample_frames_version,
                        video_sampling_strategy=self.video_sampling_strategy,
                        keyframe_mapping=self.keyframe_mapping,
                        question_key=question_key,
                        task=task,
                    )
                    pixel_values = self._cast_and_move_pixels(pixel_values)
                    video_prefix = "".join(
                        [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
                    )
                    question = video_prefix + contexts

                    response, _ = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        question,
                        gen_kwargs,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=True,
                    )
                else:
                    raise ValueError(f"Unsupported media_type: {media_type}")
            else:
                response, _ = self.model.chat(
                    self.tokenizer,
                    None,
                    contexts,
                    gen_kwargs,
                    num_patches_list=None,
                    history=None,
                    return_history=True,
                )

            res.append(response)

            if "pixel_values" in locals():
                del pixel_values
            gc.collect()
            torch.cuda.empty_cache()
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("InternVL3_5 loglikelihood is not implemented.")
