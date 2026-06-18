from lmms_eval.api.registry import register_model

from .qwen3vl import Qwen3VL


@register_model("medmo_8b_next")
class MedMO8BNext(Qwen3VL):
    DEFAULT_PRETRAINED = "MBZUAI/MedMO-8B-Next"
    MODEL_DISPLAY_NAME = "MedMO-8B-Next"
    TOKEN_USAGE_MODEL_NAME = "medmo_8b_next"
    TOKEN_USAGE_FILENAME = "medmo_8b_next_token_usage.jsonl"
