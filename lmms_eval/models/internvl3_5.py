import logging
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

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def _read_meminfo():
    """Return (mem_available_gib, swap_free_gib) from /proc/meminfo (Linux only)."""
    mem_avail_kb = None
    swap_free_kb = None
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                mem_avail_kb = int(line.split()[1])
            elif line.startswith("SwapFree:"):
                swap_free_kb = int(line.split()[1])
    mem_avail_gib = mem_avail_kb / 1024 / 1024 if mem_avail_kb is not None else -1.0
    swap_free_gib = swap_free_kb / 1024 / 1024 if swap_free_kb is not None else -1.0
    return mem_avail_gib, swap_free_gib


def _proc_rss_gib():
    """Return current process RSS in GiB from /proc/self/status (Linux only)."""
    rss_kb = None
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
                break
    return rss_kb / 1024 / 1024 if rss_kb is not None else -1.0


def _log_mem(prefix: str):
    try:
        mem_avail_gib, swap_free_gib = _read_meminfo()
        rss_gib = _proc_rss_gib()
        eval_logger.info(
            f"[{prefix}] MemAvailable={mem_avail_gib:.2f} GiB | SwapFree={swap_free_gib:.2f} GiB | ProcRSS={rss_gib:.2f} GiB"
        )
    except Exception as e:
        eval_logger.info(f"[{prefix}] (memory log failed: {e})")


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
    save_sampled_frames=False,
):
    vr = VideoReader(video_path, ctx=cpu(0))
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    sampled_frames = []

    for frame_index in frame_indices:
        frame_np = vr[frame_index].asnumpy()
        if save_sampled_frames:
            sampled_frames.append((frame_index, frame_np))
        img = Image.fromarray(frame_np).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)

    if save_sampled_frames:
        try:
            import os
            from pathlib import Path
            import cv2

            video_name = Path(video_path).stem
            model_name = "internvl3_5"
            save_dir = os.path.join(os.path.dirname(video_path), "sampled_frames")
            os.makedirs(save_dir, exist_ok=True)

            for frame_index, frame_np in sampled_frames:
                frame_file = f"{video_name}_frame{frame_index:04d}_{model_name}.jpg"
                save_path = os.path.join(save_dir, frame_file)
                cv2.imwrite(save_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        except Exception as e:
            eval_logger.warning(f"Failed to save sampled frames: {e}")

    return pixel_values, num_patches_list


@register_model("internvl3_5")
class InternVL3_5(lmms):
    def split_model_for_78b(self):
        """
        Device map aligned to the official InternVL3-78B HF model card (multi-GPU split_model example),
        adapted for 6x RTX 4090 and LMMS.

        Key differences vs previous version:
        - num_layers is read from config (no hardcoded 80)
        - explicitly pins official non-layer modules to GPU0 (vision_model/mlp1/rotary/output/etc.)
        - forces last layer onto GPU0 (official workaround for device mismatch)
        - keeps device_map[""]=0 as a LMMS safety net (avoid any CPU placement)
        """
        import math

        # Official model card reads layers from config.llm_config.num_hidden_layers
        cfg = AutoConfig.from_pretrained(self.path, trust_remote_code=True)
        num_layers = int(getattr(getattr(cfg, "llm_config", cfg), "num_hidden_layers"))

        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("No CUDA devices found for InternVL3-78B.")
        if world_size != 6:
            eval_logger.warning(f"Expected 6 GPUs (RTX 4090 x6), but detected {world_size} GPUs.")

        device_map = {}

        # Official heuristic: treat GPU0 as "half GPU" because it hosts vision + shared modules.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

        layer_cnt = 0
        for i, n in enumerate(num_layers_per_gpu):
            for _ in range(n):
                if layer_cnt < num_layers:
                    device_map[f"language_model.model.layers.{layer_cnt}"] = i
                    layer_cnt += 1

        # --- Official pins (model card) ---
        # Note: Some names differ across InternVL code variants; we include official keys + common aliases.
        pinned_to_0 = [
            "vision_model",
            "mlp1",
            "language_model.model.tok_embeddings",
            "language_model.model.embed_tokens",
            "language_model.output",
            "language_model.model.norm",
            "language_model.model.rotary_emb",
            "language_model.lm_head",
        ]
        for k in pinned_to_0:
            device_map[k] = 0

        # Official workaround: force last layer to GPU0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

        # LMMS safety net: any unmatched module stays on GPU0 (NOT CPU)
        device_map[""] = 0
        return device_map

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL-3.5-2B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        max_frames_num: int = 32,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        save_sampled_frames: bool = False,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.path = pretrained
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.save_sampled_frames = save_sampled_frames

        norm_name = self.path.lower().replace("-", "").replace("_", "")
        is_78b = "78b" in norm_name
        if is_78b and device_map != "auto":
            raise RuntimeError(
                f"InternVL3-78B must use device_map='auto'. Got {device_map!r}"
            )

        load_in_8bit = str(load_in_8bit).lower() in ["true", "1", "t", "y", "yes"]
        load_in_4bit = str(load_in_4bit).lower() in ["true", "1", "t", "y", "yes"]
        use_flash_attn = str(use_flash_attn).lower() in ["true", "1", "t", "y", "yes"]

        from transformers import BitsAndBytesConfig

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Guardrails: keep CPU budget small to avoid host swap death
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            max_memory[i] = "23GiB"
            max_memory[f"cuda:{i}"] = "23GiB"
        max_memory["cpu"] = "8GiB"

        custom_device_map = device_map
        if device_map == "auto":
            if is_78b:
                eval_logger.info(
                    f"Detected giant model {self.path}, using official-aligned 78B device_map."
                )
                custom_device_map = self.split_model_for_78b()
            else:
                custom_device_map = "auto"

        _log_mem("before_model_from_pretrained")
        self._model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=custom_device_map,
            max_memory=max_memory,
            quantization_config=quantization_config,
            use_flash_attn=use_flash_attn,
        ).eval()
        _log_mem("after_model_from_pretrained")

        self._config = AutoConfig.from_pretrained(self.path, trust_remote_code=True)

        _log_mem("before_tokenizer_from_pretrained")
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        _log_mem("after_tokenizer_from_pretrained")

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

    def generate_until(self, requests) -> List[str]:
        import gc

        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for idx, (contexts, gen_kwargs, doc_to_visual, doc_id, task, split) in enumerate(
            [reg.args for reg in requests]
        ):
            _log_mem(f"req{idx}_start")

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

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if visuals != [None]:
                visuals = self.flatten(visuals)
                if self.modality == "image":
                    if visuals:
                        visuals = [self._cast_and_move_pixels(load_image(v)) for v in visuals]
                        pixel_values = torch.cat(visuals, dim=0)
                        num_patches_list = [v.size(0) for v in visuals]
                        image_tokens = " ".join(["<image>"] * len(visuals))
                        contexts = image_tokens + "\n" + contexts
                    else:
                        pixel_values = None
                        num_patches_list = None

                    _log_mem(f"req{idx}_before_chat_image")
                    response, _ = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        contexts,
                        gen_kwargs,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=True,
                    )
                    _log_mem(f"req{idx}_after_chat_image")

                elif self.modality == "video":
                    assert len(visuals) == 1, f"Only one video is supported, got {len(visuals)}."
                    video_path = visuals[0]
                    pixel_values, num_patches_list = load_video(
                        video_path,
                        num_segments=self.max_frames_num,
                        max_num=1,
                        save_sampled_frames=self.save_sampled_frames,
                    )
                    pixel_values = self._cast_and_move_pixels(pixel_values)
                    video_prefix = "".join(
                        [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
                    )
                    question = video_prefix + contexts

                    _log_mem(f"req{idx}_before_chat_video")
                    response, _ = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        question,
                        gen_kwargs,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=True,
                    )
                    _log_mem(f"req{idx}_after_chat_video")
                else:
                    raise ValueError(f"Unsupported modality: {self.modality}")
            else:
                _log_mem(f"req{idx}_before_chat_text")
                response, _ = self.model.chat(
                    self.tokenizer,
                    None,
                    contexts,
                    gen_kwargs,
                    num_patches_list=None,
                    history=None,
                    return_history=True,
                )
                _log_mem(f"req{idx}_after_chat_text")

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