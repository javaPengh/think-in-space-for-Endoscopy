import re


VISUAL_INPUT_MODE = "visual"
BLIND_INPUT_MODE = "none"
ALLOWED_VISUAL_INPUT_MODES = {VISUAL_INPUT_MODE, BLIND_INPUT_MODE}

_VISUAL_PREAMBLE_RE = re.compile(r"^\s*(?:This is an image\.|These are frames of a video\.)\s*", re.IGNORECASE)
_VISUAL_TOKEN_LINE_RE = re.compile(r"(?im)^\s*(?:Frame\s*\d+\s*:\s*)?(?:<image>|<video>)\s*\n?")
_VISUAL_TOKEN_RE = re.compile(r"(?:<image>|<video>)\s*", re.IGNORECASE)


def normalize_visual_input_mode(value):
    mode = str(value or VISUAL_INPUT_MODE).strip().lower()
    if mode not in ALLOWED_VISUAL_INPUT_MODES:
        raise ValueError(f"Unsupported visual_input_mode: {value}. Expected one of: {sorted(ALLOWED_VISUAL_INPUT_MODES)}")
    return mode


def is_blind_mode(value):
    return normalize_visual_input_mode(value) == BLIND_INPUT_MODE


def strip_visual_context(text):
    cleaned = "" if text is None else str(text)
    cleaned = _VISUAL_TOKEN_LINE_RE.sub("", cleaned)
    cleaned = _VISUAL_TOKEN_RE.sub("", cleaned)
    cleaned = _VISUAL_PREAMBLE_RE.sub("", cleaned)
    return cleaned.lstrip()
