import re


SOURCE_ID_KEYS = ("question_id", "id", "ID", "Question_ID", "questionId")


def normalize_question_key(value):
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        return str(int(float(text)))
    return text


def get_question_key(doc_data, fallback):
    if isinstance(doc_data, dict):
        for key in SOURCE_ID_KEYS:
            value = doc_data.get(key)
            normalized = normalize_question_key(value)
            if normalized is not None:
                return normalized
    return normalize_question_key(fallback) or ""
