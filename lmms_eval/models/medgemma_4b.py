from lmms_eval.api.registry import register_model

from .medgemma_27b import MedGemma27B


@register_model("medgemma_4b")
class MedGemma4B(MedGemma27B):
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/google/medgemma-4b-it"
    MODEL_DISPLAY_NAME = "MedGemma-4B-IT"
    TOKEN_USAGE_MODEL_NAME = "medgemma_4b"
    TOKEN_USAGE_FILENAME = "medgemma_4b_token_usage.jsonl"
    DOWNLOAD_HINT = "Download google/medgemma-4b-it from ModelScope to this local path, or pass a local pretrained path."

    def __init__(self, device_map: str = "cuda:0", **kwargs):
        super().__init__(device_map=device_map, **kwargs)
