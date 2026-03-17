import math
from datetime import timedelta
from typing import List, Optional, Tuple

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
from lmms_eval.logging_utils import eval_logger

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = {
    "num_beams": 1,
    "max_new_tokens": 1024,
    "do_sample": False,
}


def build_transform(input_size: int) -> T.Compose:
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
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


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 1,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
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
    processed_images: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video(
    video_path: str,
    bound: Optional[Tuple[float, float]] = None,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 32,
) -> Tuple[torch.Tensor, List[int]]:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000.0, 100000.0

    start_idx = max(0, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )

    pixel_values_list: List[torch.Tensor] = []
    num_patches_list: List[int] = []
    transform = build_transform(input_size=input_size)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(
            img,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    return torch.cat(pixel_values_list), num_patches_list


@register_model("internvl3_78b")
class InternVL3_78B(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3-78B",
        modality: str = "video",
        device: str = "cuda:0",
        device_map: str = "auto",
        batch_size: str = "1",
        max_frames_num: int = 32,
        load_in_8bit: bool = False,
        use_flash_attn: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.path = pretrained
        self.modality = modality
        self.max_frames_num = int(max_frames_num)

        if self.modality not in {"image", "video"}:
            raise ValueError(f"Unsupported modality: {self.modality}")
        if str(device_map).lower() != "auto":
            raise ValueError("InternVL3-78B must use device_map=auto to follow official split strategy.")

        load_in_8bit = str(load_in_8bit).lower() in {"true", "1", "t", "y", "yes"}
        use_flash_attn = str(use_flash_attn).lower() in {"true", "1", "t", "y", "yes"}

        self._config = AutoConfig.from_pretrained(self.path, trust_remote_code=True)
        split_device_map = self._split_model_for_78b(self._config)

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "use_flash_attn": use_flash_attn,
            "trust_remote_code": True,
            "device_map": split_device_map,
        }
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self._model = AutoModel.from_pretrained(self.path, **model_kwargs).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=False)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        batch_size = int(batch_size)
        if batch_size != 1:
            raise ValueError(f"Batch size should be 1 for InternVL3-78B, but got {batch_size}.")
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and str(device_map).lower() == "auto":
            self._device = torch.device(device)
            self.device_map = "auto"
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if accelerator.num_processes > 1:
            if accelerator.distributed_type not in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]:
                raise ValueError("Unsupported distributed type provided.")

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True,
                    **kwargs,
                )

            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)

            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _split_model_for_78b(self, config: AutoConfig) -> dict:
        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("No CUDA devices found for InternVL3-78B.")

        llm_config = getattr(config, "llm_config", config)
        num_layers = int(getattr(llm_config, "num_hidden_layers"))

        # 官方思路是把 GPU0 视作半张卡，专门承担视觉塔和共享模块，避免跨卡错位。
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

        device_map = {}
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _ in range(num_layer):
                if layer_cnt >= num_layers:
                    break
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1

        # 这些模块固定到 GPU0 是官方多卡样例的关键，可避免推理时出现 device mismatch。
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.model.rotary_emb"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

        return device_map

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

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            gen_kwargs = dict(gen_kwargs)
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                gen_kwargs.setdefault(k, v)
            if "pad_token_id" not in gen_kwargs and self.tokenizer.eos_token_id is not None:
                gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

            pop_keys = [k for k in gen_kwargs if k not in DEFAULT_GEN_KWARGS and k != "pad_token_id"]
            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if visuals != [None]:
                visuals = self.flatten(visuals)
                if self.modality == "image":
                    visual_tensors = [self._cast_and_move_pixels(load_video(v, num_segments=1)[0]) for v in visuals]
                    pixel_values = torch.cat(visual_tensors, dim=0)
                    num_patches_list = [v.size(0) for v in visual_tensors]
                    image_tokens = " ".join(["<image>"] * len(visuals))
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
                else:
                    if len(visuals) != 1:
                        raise ValueError(f"Only one video is supported, got {len(visuals)}.")
                    pixel_values, num_patches_list = load_video(
                        visuals[0],
                        num_segments=self.max_frames_num,
                        max_num=1,
                    )
                    pixel_values = self._cast_and_move_pixels(pixel_values)
                    video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
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
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("InternVL3-78B loglikelihood is not implemented.")

