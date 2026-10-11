from lmms_eval.api.registry import register_model
from lmms_eval.models.internvl3_5 import InternVL3_5
from lmms_eval.models.local_native_video import local_checkpoint


@register_model("internvl3_5_30b_a3b")
class InternVL3_5_30B_A3B(InternVL3_5):
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/OpenGVLab/InternVL3_5-30B-A3B"

    def __init__(self, pretrained=None, device_map="balanced", video_sample_fps=None, video_sampling_strategy="uniform", **kwargs):
        if video_sample_fps not in (None, "", "none", "None") or video_sampling_strategy not in {"uniform", "specific"}:
            raise ValueError("InternVL3.5 supports uniform or specific sampling; fps sampling is not supported by its adapter.")
        if "local_files_only" in kwargs:
            raise TypeError("InternVL3.5-30B-A3B always uses local_files_only=True.")
        path = local_checkpoint(pretrained, self.DEFAULT_PRETRAINED)
        super().__init__(pretrained=path, device_map=device_map, video_sampling_strategy=video_sampling_strategy, local_files_only=True, **kwargs)
