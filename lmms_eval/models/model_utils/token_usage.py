class TokenUsageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.by_media = {}

    def record(self, usage, media_type=None):
        if not usage:
            return

        input_tokens = _to_int(usage.get("input_tokens"))
        output_tokens = _to_int(usage.get("output_tokens"))
        total_tokens = _to_int(usage.get("total_tokens"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        if input_tokens is None and output_tokens is None and total_tokens is None:
            return

        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        total_tokens = total_tokens if total_tokens is not None else input_tokens + output_tokens

        self.num_requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens

        if media_type:
            media_usage = self.by_media.setdefault(media_type, {"num_requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            media_usage["num_requests"] += 1
            media_usage["input_tokens"] += input_tokens
            media_usage["output_tokens"] += output_tokens
            media_usage["total_tokens"] += total_tokens

    def summary(self):
        if self.num_requests == 0:
            return None
        return {
            "num_requests": self.num_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "by_media": self.by_media,
        }


def _to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_openai_chat_usage(response_data):
    usage = response_data.get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def extract_openai_responses_usage(response_data):
    usage = response_data.get("usage") or {}
    return {
        "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
        "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def extract_gemini_usage(response_data):
    usage = response_data.get("usageMetadata") or {}
    return {
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
    }
