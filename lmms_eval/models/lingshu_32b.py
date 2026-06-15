from lmms_eval.api.registry import register_model
from .qwen3vl import Qwen3VL


@register_model("lingshu_32b")
class Lingshu32B(Qwen3VL):
    MODEL_CLASS_NAME = "Qwen2_5_VLForConditionalGeneration"
    DEFAULT_PRETRAINED = "lingshu-medical-mllm/Lingshu-32B"
    MODEL_DISPLAY_NAME = "Lingshu-32B"
    TOKEN_USAGE_MODEL_NAME = "lingshu_32b"
    TOKEN_USAGE_FILENAME = "lingshu_32b_token_usage.jsonl"
