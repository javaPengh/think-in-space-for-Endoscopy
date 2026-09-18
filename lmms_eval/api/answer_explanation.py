"""Optional post-answer follow-up, kept separate from scored responses."""

EXPLANATION_PROMPT = "Please explain your answer step by step."


class AnswerExplanation:
    def __init__(self, enabled=False, max_new_tokens=2048):
        value = str(enabled).strip().lower()
        if value not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("explain_answer must be true or false.")
        self.enabled = value in {"true", "1", "yes"}
        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens < 1:
            raise ValueError("explanation_max_new_tokens must be positive.")

    def generation_kwargs(self, answer_kwargs):
        kwargs = dict(answer_kwargs)
        # Answer-only stopping/length controls must not truncate the explanation.
        for key in ("until", "max_length", "max_new_tokens", "min_new_tokens", "min_length", "stop_strings"):
            kwargs.pop(key, None)
        kwargs.update(max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1)
        return kwargs

    def messages(self, messages, answer):
        return messages + [
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            {"role": "user", "content": [{"type": "text", "text": EXPLANATION_PROMPT}]},
        ]

    def record(self, request, answer, explanation, generation_kwargs):
        if not hasattr(request, "answer_explanations"):
            request.answer_explanations = []
        request.answer_explanations.append({"answer": answer, "prompt": EXPLANATION_PROMPT, "explanation": explanation, "generation_kwargs": dict(generation_kwargs)})
