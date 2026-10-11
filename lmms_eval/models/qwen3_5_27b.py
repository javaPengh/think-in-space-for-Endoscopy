from lmms_eval.api.registry import register_model
from lmms_eval.models.local_native_video import LocalNativeVideoModel


@register_model("qwen3_5_27b")
class Qwen3_5_27B(LocalNativeVideoModel):
    MODEL_CLASS_NAME = "Qwen3_5ForConditionalGeneration"
    EXPECTED_MODEL_TYPE = "qwen3_5"
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B"
    MODEL_DISPLAY_NAME = "Qwen3.5-27B"
    TOKEN_USAGE_MODEL_NAME = "qwen3_5_27b"
    TOKEN_USAGE_FILENAME = "qwen3_5_27b_token_usage.jsonl"

    def __init__(self, max_pixels=602112, **kwargs):
        super().__init__(max_pixels=max_pixels, **kwargs)

    def _processor_options(self, images, videos):
        if self.max_pixels is None:
            return {}
        # Qwen3.5 uses Qwen3VLProcessor; recent releases express pixel limits as size.
        options = {}
        if images:
            options["images_kwargs"] = {"size": {"shortest_edge": 65536, "longest_edge": self.max_pixels}}
        if videos:
            options["videos_kwargs"] = {"size": {"shortest_edge": 65536, "longest_edge": self.max_pixels}}
        return options
