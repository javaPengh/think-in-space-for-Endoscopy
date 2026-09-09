import math
from typing import Sequence


def calculate_effective_fps(frame_indices: Sequence[int], source_fps: float) -> float:
    """Return the constant FPS that preserves the sampled frames' endpoint times."""
    source_fps = float(source_fps)
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Source video FPS must be a positive finite number, got {source_fps}.")
    if len(frame_indices) == 0:
        raise ValueError("Cannot calculate effective FPS without sampled frame indices.")
    if len(frame_indices) == 1:
        return source_fps

    frame_span = int(frame_indices[-1]) - int(frame_indices[0])
    if frame_span <= 0:
        return source_fps

    sampled_time_span = frame_span / source_fps
    return (len(frame_indices) - 1) / sampled_time_span
