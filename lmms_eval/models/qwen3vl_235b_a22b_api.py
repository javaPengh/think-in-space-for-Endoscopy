from lmms_eval.api.registry import register_model

from .qwen2_5vl_72b_api import Qwen2_5VL_72B_API


@register_model("qwen3vl_235b_a22b_api")
class Qwen3VL_235B_A22B_API(Qwen2_5VL_72B_API):
    def __init__(self, model_version: str = "qwen3-vl-235b-a22b-instruct", **kwargs):
        super().__init__(model_version=model_version, **kwargs)
