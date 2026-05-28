from lmms_eval.api.registry import register_model

from .gemini3_1_pro import Gemini3_1Pro


@register_model("gemini3_1_flash_lite")
class Gemini3_1FlashLite(Gemini3_1Pro):
    def __init__(self, model_version: str = "gemini-3.1-flash-lite", **kwargs):
        super().__init__(model_version=model_version, **kwargs)
