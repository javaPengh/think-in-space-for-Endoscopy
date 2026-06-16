import importlib
import os
import sys

import hf_transfer
from loguru import logger

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "batch_gpt4": "BatchGPT4",
    "claude": "Claude",
    "colongpt": "ColonGPT",
    "from_log": "FromLog",
    "fuyu": "Fuyu",
    "gemini_api": "GeminiAPI",
    "gemini3_1_flash_lite": "Gemini3_1FlashLite",
    "gemini3_1_pro": "Gemini3_1Pro",
    "gpt5_4": "GPT5_4",
    "gpt4v": "GPT4V",
    "idefics2": "Idefics2",
    "instructblip": "InstructBLIP",
    "internvl": "InternVLChat",
    "internvl2": "InternVL2",
    "llama_vid": "LLaMAVid",
    "llava": "Llava",
    "llava_hf": "LlavaHf",
    "llava_onevision": "Llava_OneVision",
    "llava_onevision_1_5": "Llava_OneVision_1_5",
    "lingshu_32b": "Lingshu32B",
    "llava_sglang": "LlavaSglang",
    "llava_vid": "LlavaVid",
    "longva": "LongVA",
    "mantis": "Mantis",
    "minicpm_v": "MiniCPM_V",
    "mplug_owl_video": "mplug_Owl",
    "phi3v": "Phi3v",
    "qwen_vl": "Qwen_VL",
    "qwen_vl_api": "Qwen_VL_API",
    "reka": "Reka",
    "srt_api": "SRT_API",
    "tinyllava": "TinyLlava",
    "videoChatGPT": "VideoChatGPT",
    "video_llava": "VideoLLaVA",
    "vila": "VILA",
    "xcomposer2_4KHD": "XComposer2_4KHD",
    "xcomposer2d5": "XComposer2D5",
    "qwen2vl": "Qwen2VL",
    "qwen2_5vl_72b_api": "Qwen2_5VL_72B_API",
    "qwen3vl": "Qwen3VL",
    "qwen3vl_32b": "Qwen3VL_32B",
    "qwen3vl_235b_a22b_api": "Qwen3VL_235B_A22B_API",
    "internvl3_5": "InternVL3_5",
    "internvideo2_5_chat_8b": "InternVideo2_5_Chat",
}


def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    try:
        module = __import__(f"lmms_eval.models.{model_name}", fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise


if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            try:
                exec(f"from {plugin}.models.{model_name} import {model_class}")
            except ImportError as e:
                logger.debug(f"Failed to import {model_class} from {model_name}: {e}")
