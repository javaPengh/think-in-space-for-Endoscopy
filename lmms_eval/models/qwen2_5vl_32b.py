from lmms_eval.api.registry import register_model

from .qwen3vl import Qwen3VL


@register_model("qwen2_5vl_32b")
class Qwen2_5VL_32B(Qwen3VL):
    MODEL_CLASS_NAME = "Qwen2_5_VLForConditionalGeneration"
    DEFAULT_PRETRAINED = "Qwen/Qwen2.5-VL-32B-Instruct"
    MODEL_DISPLAY_NAME = "Qwen2.5-VL-32B-Instruct"
    TOKEN_USAGE_MODEL_NAME = "qwen2_5vl_32b"
    TOKEN_USAGE_FILENAME = "qwen2_5vl_32b_token_usage.jsonl"
