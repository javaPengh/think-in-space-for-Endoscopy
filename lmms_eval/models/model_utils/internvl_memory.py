"""Inference memory controls for InternVL's independent image/patch encoder."""

import math
from functools import wraps

import torch
from accelerate.utils import get_max_memory


def inference_max_memory(reserve_gib=4):
    """Budget weights from current free memory, leaving room on every visible GPU."""
    reserve_gib = float(reserve_gib)
    if not math.isfinite(reserve_gib) or reserve_gib <= 0:
        raise ValueError("gpu_memory_reserve_gib must be a finite positive number.")
    # Accelerate initializes CUDA contexts before querying free memory and also
    # includes available host RAM for CPU offload when the GPUs cannot fit weights.
    max_memory = get_max_memory()
    reserve_bytes = int(reserve_gib * 1024**3)
    for device, available in max_memory.items():
        if isinstance(device, int):
            if available <= reserve_bytes:
                raise ValueError(f"Visible GPU {device} has only {available / 1024**3:.2f} GiB free; cannot reserve {reserve_gib:g} GiB for inference.")
            max_memory[device] = available - reserve_bytes
    return max_memory


def enable_chunked_vision(model, batch_size):
    """Bound ViT activation memory without dropping/reordering frames or tiles."""
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("vision_batch_size must be >= 1.")
    extract_feature = model.extract_feature

    @wraps(extract_feature)
    @torch.no_grad()
    def extract_feature_chunked(pixel_values):
        if pixel_values.shape[0] <= batch_size:
            return extract_feature(pixel_values)
        # Each patch is encoded independently. Concatenate only the projected
        # features, so large intermediate ViT activations die between chunks.
        return torch.cat([extract_feature(chunk) for chunk in pixel_values.split(batch_size, dim=0)], dim=0)

    model.extract_feature = extract_feature_chunked
