from lmms_eval.api.registry import register_model

from .medgemma_27b import MedGemma27B


@register_model("gemma_3_27b")
class Gemma3_27B(MedGemma27B):
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/LLM-Research/gemma-3-27b-it"
    MODEL_DISPLAY_NAME = "Gemma-3-27B-IT"
    TOKEN_USAGE_MODEL_NAME = "gemma_3_27b"
    TOKEN_USAGE_FILENAME = "gemma_3_27b_token_usage.jsonl"
    DOWNLOAD_HINT = "Download LLM-Research/gemma-3-27b-it from ModelScope to this local path, or pass a local pretrained path."
