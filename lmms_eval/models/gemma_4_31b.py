from lmms_eval.api.registry import register_model
from lmms_eval.models.local_native_video import LocalNativeVideoModel


@register_model("gemma_4_31b")
class Gemma4_31B(LocalNativeVideoModel):
    MODEL_CLASS_NAME = "Gemma4ForConditionalGeneration"
    EXPECTED_MODEL_TYPE = "gemma4"
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/google/gemma-4-31B-it"
    MODEL_DISPLAY_NAME = "Gemma-4-31B-it"
    TOKEN_USAGE_MODEL_NAME = "gemma_4_31b"
    TOKEN_USAGE_FILENAME = "gemma_4_31b_token_usage.jsonl"

    def __init__(self, max_soft_tokens=280, **kwargs):
        self.max_soft_tokens = int(max_soft_tokens)
        if self.max_soft_tokens not in {70, 140, 280, 560, 1120}:
            raise ValueError("Gemma 4 max_soft_tokens must be one of 70, 140, 280, 560, 1120.")
        if kwargs.get("max_pixels") is not None:
            raise ValueError("Gemma 4 uses max_soft_tokens instead of max_pixels.")
        super().__init__(**kwargs)

    def _processor_options(self, images, videos):
        options = {}
        if images:
            options["images_kwargs"] = {"max_soft_tokens": self.max_soft_tokens}
        if videos:
            options["videos_kwargs"] = {"max_soft_tokens": self.max_soft_tokens}
        return options
