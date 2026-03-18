from lmms_eval.api.registry import register_model
from .qwen3vl import Qwen3VL

@register_model("qwen3vl_32b")
class Qwen3VL_32B(Qwen3VL):
    def __init__(self, pretrained: str = "Qwen/Qwen3-VL-32B-Instruct", **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
