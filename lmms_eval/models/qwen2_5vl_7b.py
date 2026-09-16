from lmms_eval.api.registry import register_model

from .qwen2_5vl_32b import Qwen2_5VL_32B


@register_model("qwen2_5vl_7b")
class Qwen2_5VL_7B(Qwen2_5VL_32B):
    DEFAULT_PRETRAINED = "Qwen/Qwen2.5-VL-7B-Instruct"
    MODEL_DISPLAY_NAME = "Qwen2.5-VL-7B-Instruct"
    TOKEN_USAGE_MODEL_NAME = "qwen2_5vl_7b"
    TOKEN_USAGE_FILENAME = "qwen2_5vl_7b_token_usage.jsonl"
