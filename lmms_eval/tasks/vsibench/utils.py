import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

MCA_QUESTION_TYPES = [
    "object_rel_direction",
    "fold_rel_depth",
    "route_planning",
    "object_order",
    "action_order",
]
NA_QUESTION_TYPES = [
    "object_counting",
    "action_counting",
    "polyp_size_estimation",
]

ACCURACY_METRIC = "accuracy"
MRA_METRIC = "MRA:.5:.95:.05"

METRICS_FOR_MCA = {
    ACCURACY_METRIC: "exact_match",
}

METRICS_FOR_NA = {
    MRA_METRIC: "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

AGGREGATED_RESULT_KEYS = [
    f"object_counting_{MRA_METRIC}",
    f"action_counting_{MRA_METRIC}",
    f"object_rel_direction_{ACCURACY_METRIC}",
    f"fold_rel_depth_{ACCURACY_METRIC}",
    f"route_planning_{ACCURACY_METRIC}",
    f"object_order_{ACCURACY_METRIC}",
    f"action_order_{ACCURACY_METRIC}",
    f"polyp_size_estimation_{MRA_METRIC}",
]

ANSWER_TYPE_MULTIPLE_CHOICE = "multiple_choice"
ANSWER_TYPE_NUMERIC = "numeric"
ANSWER_TYPE_ALIASES = {
    "mca": ANSWER_TYPE_MULTIPLE_CHOICE,
    "mac": ANSWER_TYPE_MULTIPLE_CHOICE,
    "mcq": ANSWER_TYPE_MULTIPLE_CHOICE,
    "multiple_choice": ANSWER_TYPE_MULTIPLE_CHOICE,
    "choice": ANSWER_TYPE_MULTIPLE_CHOICE,
    "na": ANSWER_TYPE_NUMERIC,
    "numeric": ANSWER_TYPE_NUMERIC,
    "number": ANSWER_TYPE_NUMERIC,
    "numerical": ANSWER_TYPE_NUMERIC,
}

FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*[:：]\s*([^\n\r]+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
LLM_EXTRACTION_FAILED = "EXTRACTION_FAILED"
LLM_EXTRACTION_FAILURE_LABEL = "提取失败"


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def _is_missing_value(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _doc_value(doc, key, default=None):
    value = doc[key] if key in doc else default
    return default if _is_missing_value(value) else value


def _normalize_media_type(media_type, media_path):
    media_type = str(media_type).lower().strip() if media_type is not None else ""
    if media_type in {"image", "video"}:
        return media_type

    suffix = Path(str(media_path)).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Cannot infer media_type for media path: {media_path}")


def _resolve_media_path(media_path, cache_dir):
    media_path = os.path.expanduser(str(media_path))
    if os.path.isabs(media_path):
        return media_path
    return os.path.join(cache_dir, media_path)


def _doc_media_type(doc):
    media_path = _doc_value(doc, "media_path")
    if media_path:
        return _normalize_media_type(_doc_value(doc, "media_type"), media_path)
    return "video"


def _doc_answer_type(doc):
    raw_answer_type = _doc_value(doc, "answer_type")
    if raw_answer_type is not None and str(raw_answer_type).strip():
        normalized_answer_type = ANSWER_TYPE_ALIASES.get(str(raw_answer_type).lower().strip())
        if normalized_answer_type is None:
            raise ValueError(f"Unknown answer_type: {raw_answer_type}")
        return normalized_answer_type

    question_type = doc["question_type"]
    if question_type in MCA_QUESTION_TYPES:
        return ANSWER_TYPE_MULTIPLE_CHOICE
    if question_type in NA_QUESTION_TYPES:
        return ANSWER_TYPE_NUMERIC
    raise ValueError(f"Unknown question type without answer_type: {question_type}")


def _is_blind_visual_input_mode():
    return str(os.getenv("VSI_VISUAL_INPUT_MODE", "visual")).strip().lower() == "none"


def _is_natural_answer_mode():
    return str(os.getenv("VSI_ANSWER_MODE", "restricted")).strip().lower() == "natural"


def _metrics_for_answer_type(answer_type):
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        return METRICS_FOR_MCA
    if answer_type == ANSWER_TYPE_NUMERIC:
        return METRICS_FOR_NA
    raise ValueError(f"Unknown normalized answer_type: {answer_type}")


def vsibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)

    media_path = _doc_value(doc, "media_path")
    if media_path:
        media_type = _normalize_media_type(_doc_value(doc, "media_type"), media_path)
        abs_media_path = _resolve_media_path(media_path, cache_dir)
        if not os.path.exists(abs_media_path):
            raise FileExistsError(f"{media_type} path:{abs_media_path} does not exist.")
        return [{"media_type": media_type, "path": abs_media_path}]
    else:
        abs_media_path = os.path.join(cache_dir, doc["dataset"], doc["scene_name"] + ".mp4")
        if not os.path.exists(abs_media_path):
            raise FileExistsError(f"video path:{abs_media_path} does not exist.")
        return [abs_media_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    question = doc["question"]

    if _is_blind_visual_input_mode():
        pre_prompt = ""
    else:
        default_pre_prompt = "This is an image." if _doc_media_type(doc) == "image" else "These are frames of a video."
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or default_pre_prompt

    answer_type = _doc_answer_type(doc)
    if answer_type == ANSWER_TYPE_NUMERIC:
        if _is_natural_answer_mode():
            post_prompt = lmms_eval_specific_kwargs.get("na_natural_post_prompt", "") or "You may explain your reasoning briefly. End your response with a separate line in the exact format: Final answer: <number>"
        else:
            post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return "\n".join([part for part in [pre_prompt, question, post_prompt] if part])
    elif answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        options = "Options:\n" + "\n".join(doc["options"])
        if _is_natural_answer_mode():
            post_prompt = lmms_eval_specific_kwargs.get("mca_natural_post_prompt", "") or "You may explain your reasoning briefly. End your response with a separate line in the exact format: Final answer: <option letter>"
        else:
            post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([part for part in [pre_prompt, question, options, post_prompt] if part])
    else:
        raise ValueError(f"Unknown answer type: {answer_type}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


def fuzzy_matching(pred):
    return str(pred or "").split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if str(pred or "").lower() == str(target or "").lower() else 0.0


def abs_dist_norm(pred, target):
    if pred is None or target is None:
        return float("inf")
    if target == 0:
        return 0.0 if pred == 0 else float("inf")
    return abs(pred - target) / abs(target)


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


WORST_CASE_FOR_METRICS = {
    ACCURACY_METRIC: 0.0,
    MRA_METRIC: 0.0,
}


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred


def _extract_final_answer_text(pred):
    matches = FINAL_ANSWER_RE.findall(str(pred or ""))
    if not matches:
        return None
    return matches[-1].strip().strip("`*_ ")


def _option_label(index):
    return chr(ord("A") + index)


def _parse_option_map(doc):
    option_map = {}
    for index, option in enumerate(doc.get("options") or []):
        option_text = str(option).strip()
        match = re.match(r"^\s*([A-Za-z])\s*[\.\)、\):：-]\s*(.*)$", option_text)
        if match:
            label = match.group(1).upper()
            text = match.group(2).strip()
        else:
            label = _option_label(index)
            text = option_text
        option_map[label] = text
    return option_map


def _normalize_option_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _extract_choice_prediction(pred, doc, require_final_answer=False):
    raw_text = str(pred or "").strip()
    final_answer = _extract_final_answer_text(raw_text)
    if require_final_answer and not final_answer:
        return None
    candidates = [candidate for candidate in ([final_answer] if require_final_answer else [final_answer, raw_text]) if candidate]
    option_map = _parse_option_map(doc)
    valid_labels = set(option_map.keys())

    for candidate in candidates:
        candidate_text = candidate.strip()
        line_candidates = [line.strip() for line in candidate_text.splitlines() if line.strip()]
        direct_candidates = [candidate_text]
        if line_candidates:
            direct_candidates = [line_candidates[-1], line_candidates[0], candidate_text]
        patterns = [
            r"(?:answer|option|choice|choose|select|selected|pick|picked|答案|选项)\s*(?:is|为|是)?\s*[:：]?\s*[\(\[]?([A-Za-z])[\)\].、:：]?",
            r"^[\s\(\[]*([A-Za-z])[\s\)\].、:：]*$",
            r"[\(\[]([A-Za-z])[\)\]]",
        ]
        for direct_candidate in direct_candidates:
            for pattern in patterns:
                for match in re.finditer(pattern, direct_candidate, flags=re.IGNORECASE):
                    label = match.group(1).upper()
                    if label in valid_labels:
                        return label

    for candidate in candidates:
        normalized_candidate = _normalize_option_text(candidate)
        for label, option_text in option_map.items():
            normalized_option = _normalize_option_text(option_text)
            if normalized_option and len(normalized_option) > 1 and normalized_option in normalized_candidate:
                return label

    fallback_label = fuzzy_matching(final_answer or raw_text).upper()
    if fallback_label in valid_labels:
        return fallback_label
    return None


def _extract_numeric_prediction(pred, require_final_answer=False):
    raw_text = str(pred or "").strip()
    final_answer = _extract_final_answer_text(raw_text)
    if require_final_answer and not final_answer:
        return None
    candidates = [final_answer] if require_final_answer else [final_answer, raw_text]
    for candidate in candidates:
        if not candidate:
            continue
        answer_pattern = r"(?:answer|答案|结果)\s*(?:is|为|是)?\s*[:：]?\s*(" + NUMBER_RE.pattern + r")"
        answer_match = re.search(answer_pattern, candidate, flags=re.IGNORECASE)
        if answer_match:
            return answer_match.group(1)

        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        direct_candidates = [candidate.strip()]
        if lines:
            direct_candidates = [lines[-1], lines[0], candidate.strip()]
        for direct_candidate in direct_candidates:
            numbers = NUMBER_RE.findall(direct_candidate)
            if len(numbers) == 1 and len(direct_candidate) <= 80:
                return numbers[0]
    return None


_LLM_EXTRACTION_CACHE = {}
_LLM_EXTRACTOR_MISSING_KEY_WARNED = False
_FATAL_LLM_EXTRACTOR_HTTP_STATUS = {400, 401, 403, 404}
_FATAL_LLM_EXTRACTOR_ERROR_KEYWORDS = (
    "arrearage",
    "invalidapikey",
    "invalid_api_key",
    "unauthorized",
    "forbidden",
    "accessdenied",
    "access_denied",
    "permission",
    "quota",
    "insufficient",
    "modelnotfound",
    "model_not_found",
)


def _to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _llm_extractor_api_key():
    return os.getenv("VSI_LLM_EXTRACTOR_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def _llm_extractor_enabled():
    explicit = os.getenv("VSI_LLM_EXTRACTOR_ENABLED")
    if explicit is not None:
        return _to_bool(explicit)
    return bool(_llm_extractor_api_key())


def _llm_extractor_user_prompt(doc, raw_prediction, answer_type):
    question = str(doc.get("question", "")).strip()
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        options = "\n".join(str(option) for option in (doc.get("options") or []))
        answer_instruction = (
            "The expected answer type is multiple choice. If the model output explicitly states one option, return only its option letter. "
            f"Valid option letters are: {', '.join(sorted(_parse_option_map(doc).keys()))}. "
            "If it only states option text, map it to a letter only when the output clearly selects that option."
        )
    else:
        options = ""
        answer_instruction = "The expected answer type is numeric. If the model output explicitly states one numeric answer, return only that number."

    parts = [
        answer_instruction,
        "Do not solve the question yourself. Do not infer an answer from the question, options, or world knowledge.",
        f"If the model output does not explicitly contain a final answer, return exactly {LLM_EXTRACTION_FAILED}.",
        "Original question:",
        question,
    ]
    if options:
        parts.extend(["Options:", options])
    parts.extend(["Model output to extract from:", str(raw_prediction or "").strip()])
    return "\n\n".join(parts)


def _decode_http_error_body(error):
    try:
        return error.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _llm_extractor_error_code(error_body):
    try:
        data = json.loads(error_body)
    except json.JSONDecodeError:
        return ""

    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("code") or error.get("type") or "")
    return str(data.get("code") or data.get("type") or "") if isinstance(data, dict) else ""


def _is_fatal_llm_extractor_http_error(status_code, error_body):
    if status_code in _FATAL_LLM_EXTRACTOR_HTTP_STATUS:
        return True
    haystack = f"{_llm_extractor_error_code(error_body)} {error_body}".lower().replace("-", "").replace(" ", "")
    return any(keyword in haystack for keyword in _FATAL_LLM_EXTRACTOR_ERROR_KEYWORDS)


def _raise_fatal_llm_extractor_error(status_code, error_body):
    detail = error_body.strip() or "<empty response body>"
    raise RuntimeError("VSI LLM answer extractor is unavailable due to a non-retryable DashScope/Bailian API error. " f"HTTP status: {status_code}. Response body: {detail}")


def _call_llm_extractor(prompt):
    global _LLM_EXTRACTOR_MISSING_KEY_WARNED

    api_key = _llm_extractor_api_key()
    if not api_key:
        if not _LLM_EXTRACTOR_MISSING_KEY_WARNED:
            eval_logger.warning("VSI LLM answer extractor is enabled but no API key was found. Set VSI_LLM_EXTRACTOR_API_KEY or DASHSCOPE_API_KEY.")
            _LLM_EXTRACTOR_MISSING_KEY_WARNED = True
        raise RuntimeError("VSI LLM answer extractor is enabled but no API key was found. Set VSI_LLM_EXTRACTOR_API_KEY or DASHSCOPE_API_KEY.")

    base_url = os.getenv("VSI_LLM_EXTRACTOR_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_url = os.getenv("VSI_LLM_EXTRACTOR_API_URL") or f"{base_url.rstrip('/')}/chat/completions"
    model = os.getenv("VSI_LLM_EXTRACTOR_MODEL", "qwen-plus")
    timeout = int(os.getenv("VSI_LLM_EXTRACTOR_TIMEOUT", "60"))
    max_retries = int(os.getenv("VSI_LLM_EXTRACTOR_MAX_RETRIES", "2"))
    retry_sleep = float(os.getenv("VSI_LLM_EXTRACTOR_RETRY_SLEEP", "2"))

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": ("You are a strict answer extraction tool for benchmark logs. " f"Return only the extracted answer or exactly {LLM_EXTRACTION_FAILED}. " "Never explain. Never guess."),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 32,
    }
    request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(api_url, data=request_data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            return response_data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            error_body = _decode_http_error_body(error)
            if _is_fatal_llm_extractor_http_error(error.code, error_body):
                _raise_fatal_llm_extractor_error(error.code, error_body)
            if attempt >= max_retries:
                eval_logger.warning(f"VSI LLM answer extractor failed after {attempt + 1} attempts: HTTP {error.code}: {error_body}")
                return None
            eval_logger.warning(f"VSI LLM answer extractor retrying after HTTP {error.code}: {error_body}")
            time.sleep(retry_sleep)
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as error:
            if attempt >= max_retries:
                eval_logger.warning(f"VSI LLM answer extractor failed after {attempt + 1} attempts: {error}")
                return None
            time.sleep(retry_sleep)
    return None


def _parse_llm_extracted_answer(raw_answer, doc, answer_type):
    text = str(raw_answer or "").strip().strip("`*_ ")
    if not text or LLM_EXTRACTION_FAILED in text.upper() or LLM_EXTRACTION_FAILURE_LABEL in text:
        return None

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        valid_labels = set(_parse_option_map(doc).keys())
        match = re.search(r"\b([A-Za-z])\b", first_line)
        if match:
            label = match.group(1).upper()
            if label in valid_labels:
                return label
        return None

    numbers = NUMBER_RE.findall(first_line)
    if len(numbers) == 1:
        return numbers[0]
    numbers = NUMBER_RE.findall(text)
    if len(numbers) == 1:
        return numbers[0]
    return None


def _llm_extract_prediction(raw_prediction, doc, answer_type):
    if not _llm_extractor_enabled():
        return {"attempted": False, "prediction": None, "raw": "", "status": "disabled"}

    cache_key = json.dumps(
        {
            "answer_type": answer_type,
            "question": doc.get("question", ""),
            "options": doc.get("options", []),
            "raw_prediction": raw_prediction,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if cache_key in _LLM_EXTRACTION_CACHE:
        cached = _LLM_EXTRACTION_CACHE[cache_key].copy()
        cached["attempted"] = True
        cached["status"] = f"cached_{cached['status']}"
        return cached

    prompt = _llm_extractor_user_prompt(doc, raw_prediction, answer_type)
    raw_answer = _call_llm_extractor(prompt)
    prediction = _parse_llm_extracted_answer(raw_answer, doc, answer_type)
    status = "extracted" if prediction is not None else "failed"
    result = {"attempted": True, "prediction": prediction, "raw": raw_answer or "", "status": status}
    _LLM_EXTRACTION_CACHE[cache_key] = result.copy()
    return result


def vsibench_process_results(doc, results):
    raw_prediction = results[0] if results else ""
    natural_answer_mode = _is_natural_answer_mode()
    doc["media_type"] = _doc_media_type(doc)
    doc["answer_type"] = _doc_answer_type(doc)
    doc["llm_extraction_used"] = False
    doc["llm_extraction_status"] = ""
    doc["llm_extraction_raw"] = ""
    if doc["answer_type"] == ANSWER_TYPE_MULTIPLE_CHOICE:
        restricted_prediction = _extract_choice_prediction(raw_prediction, doc, require_final_answer=natural_answer_mode)
        if restricted_prediction is None:
            llm_extraction = _llm_extract_prediction(raw_prediction, doc, doc["answer_type"])
            doc["llm_extraction_used"] = llm_extraction["attempted"]
            doc["llm_extraction_status"] = llm_extraction["status"]
            doc["llm_extraction_raw"] = llm_extraction["raw"]
            restricted_prediction = llm_extraction["prediction"]
        doc["natural_prediction"] = raw_prediction if natural_answer_mode else ""
        doc["restricted_prediction"] = restricted_prediction
        doc["prediction"] = restricted_prediction
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(doc["prediction"], doc["ground_truth"])
        doc["is_correct"] = bool(doc.get("accuracy", 0.0))
    elif doc["answer_type"] == ANSWER_TYPE_NUMERIC:
        restricted_prediction = _extract_numeric_prediction(raw_prediction, require_final_answer=natural_answer_mode)
        if restricted_prediction is None:
            llm_extraction = _llm_extract_prediction(raw_prediction, doc, doc["answer_type"])
            doc["llm_extraction_used"] = llm_extraction["attempted"]
            doc["llm_extraction_status"] = llm_extraction["status"]
            doc["llm_extraction_raw"] = llm_extraction["raw"]
            restricted_prediction = llm_extraction["prediction"]
        doc["natural_prediction"] = raw_prediction if natural_answer_mode else ""
        doc["restricted_prediction"] = restricted_prediction
        doc["prediction"] = restricted_prediction if restricted_prediction is not None else ""
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(doc["prediction"]), to_float(doc["ground_truth"]))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
        doc["is_scored"] = any(float(doc.get(key, 0.0)) > 0.0 for key in METRICS_FOR_NA.keys())
    else:
        raise ValueError(f"Unknown answer type: {doc['answer_type']}")

    return {"vsibench_score": doc}


def _ensure_answer_type_column(results):
    results = results.copy()
    results["answer_type"] = results.apply(_doc_answer_type, axis=1)
    return results


def _primary_metric_for_answer_type(answer_type):
    return next(iter(_metrics_for_answer_type(answer_type).keys()))


def _result_key_for_question_type(question_type, answer_type, answer_type_count):
    metric = _primary_metric_for_answer_type(answer_type)
    if answer_type_count > 1:
        return f"{question_type}_{answer_type}_{metric}"
    return f"{question_type}_{metric}"


def _mean_row_score(rows):
    scores = []
    for _, row in rows.iterrows():
        answer_type = row.get("answer_type") or _doc_answer_type(row)
        metric = _primary_metric_for_answer_type(answer_type)
        if metric not in row or pd.isna(row[metric]):
            continue
        scores.append(float(row[metric]))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _build_metric_outputs(results):
    results = _ensure_answer_type_column(results)
    output = {}

    for question_type, per_question_type in results.groupby("question_type").groups.items():
        per_question_type = results.loc[per_question_type]
        answer_type_count = len(per_question_type["answer_type"].dropna().unique())
        for answer_type, answer_type_indexes in per_question_type.groupby("answer_type").groups.items():
            per_answer_type = per_question_type.loc[answer_type_indexes]
            metrics = _metrics_for_answer_type(answer_type)
            for metric in metrics.keys():
                if metric in per_answer_type:
                    output[_result_key_for_question_type(question_type, answer_type, answer_type_count)] = per_answer_type[metric].mean()

    overall = _mean_row_score(results)
    if overall is not None:
        output["overall"] = overall
    return output


def _score_to_percent(score):
    return float(score) * 100.0


def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    output = _build_metric_outputs(results)
    eval_logger.info(f"Evaluation results: {output}")

    aggregated_results = OrderedDict()
    if "overall" in output:
        aggregated_results["overall"] = _score_to_percent(output["overall"])

    if "media_type" in results.columns:
        for media_type in ["image", "video"]:
            per_media = results[results["media_type"] == media_type]
            if len(per_media) == 0:
                continue
            per_media_output = _build_metric_outputs(per_media)
            if "overall" in per_media_output:
                aggregated_results[f"{media_type}_overall"] = _score_to_percent(per_media_output["overall"])

    for key in AGGREGATED_RESULT_KEYS:
        if key in output:
            aggregated_results[key] = _score_to_percent(output[key])

    for key, value in output.items():
        if key == "overall" or key in aggregated_results:
            continue
        aggregated_results[key] = _score_to_percent(value)

    tabulated_keys = ", ".join([_ for _ in aggregated_results.keys()])
    tabulated_results = ", ".join([f"{_:.3f}" for _ in aggregated_results.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")

    aggregated_results["tabulated_keys"] = tabulated_keys
    aggregated_results["tabulated_results"] = tabulated_results

    return aggregated_results
