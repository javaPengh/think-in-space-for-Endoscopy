import os
import re
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
from collections import OrderedDict

import datasets

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "fold_rel_depth",
    "route_planning",
    "temporal(object)",
    "temporal(action)",
    "polyp_size_estimation(no_ref)",
]
NA_QUESTION_TYPES = [
    "counting(object)",
    "counting(action)",
    "polyp_size_estimation(ref)",
]

ACCURACY_METRIC = "accuracy"
MRA_METRIC = "MRA:.5:.95:.05"

METRICS_FOR_MCA = {
    ACCURACY_METRIC: "exact_match",
}

METRICS_FOR_NA = {
    MRA_METRIC: "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

COMPOSITE_QUESTION_TYPES = OrderedDict(
    [
        (
            "object_rel_direction",
            {
                "members": ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"],
                "metric_key": f"object_rel_direction_{ACCURACY_METRIC}",
                "include_members": False,
            },
        ),
        (
            "counting",
            {
                "members": ["counting(object)", "counting(action)"],
                "metric_key": f"counting_{MRA_METRIC}",
                "include_members": True,
            },
        ),
        (
            "temporal",
            {
                "members": ["temporal(object)", "temporal(action)"],
                "metric_key": f"temporal_{ACCURACY_METRIC}",
                "include_members": True,
            },
        ),
        (
            "polyp_size_estimation",
            {
                "members": ["polyp_size_estimation(ref)", "polyp_size_estimation(no_ref)"],
                "metric_key": "polyp_size_estimation_overall",
                "include_members": True,
            },
        ),
    ]
)

TOP_LEVEL_RESULT_KEYS = [
    f"counting_{MRA_METRIC}",
    f"object_rel_direction_{ACCURACY_METRIC}",
    f"fold_rel_depth_{ACCURACY_METRIC}",
    f"route_planning_{ACCURACY_METRIC}",
    f"temporal_{ACCURACY_METRIC}",
    "polyp_size_estimation_overall",
]

AGGREGATED_RESULT_KEYS = [
    f"counting_{MRA_METRIC}",
    f"counting(object)_{MRA_METRIC}",
    f"counting(action)_{MRA_METRIC}",
    f"object_rel_direction_{ACCURACY_METRIC}",
    f"fold_rel_depth_{ACCURACY_METRIC}",
    f"route_planning_{ACCURACY_METRIC}",
    f"temporal_{ACCURACY_METRIC}",
    f"temporal(object)_{ACCURACY_METRIC}",
    f"temporal(action)_{ACCURACY_METRIC}",
    "polyp_size_estimation_overall",
    f"polyp_size_estimation(ref)_{MRA_METRIC}",
    f"polyp_size_estimation(no_ref)_{ACCURACY_METRIC}",
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
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    return str(pred or "").split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if str(pred or "").lower() == str(target or "").lower() else 0.

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
    ACCURACY_METRIC: 0.,
    MRA_METRIC: 0.,
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


def _extract_choice_prediction(pred, doc):
    raw_text = str(pred or "").strip()
    final_answer = _extract_final_answer_text(raw_text)
    candidates = [candidate for candidate in [final_answer, raw_text] if candidate]
    option_map = _parse_option_map(doc)
    valid_labels = set(option_map.keys())

    for candidate in candidates:
        candidate_text = candidate.strip()
        patterns = [
            r"(?:answer|option|choice|答案|选项)\s*(?:is|为|是)?\s*[:：]?\s*[\(\[]?([A-Za-z])[\)\].、:：]?",
            r"^[\s\(\[]*([A-Za-z])[\s\)\].、:：]*$",
            r"[\(\[]([A-Za-z])[\)\]]",
            r"(?:^|[\s:：])([A-Za-z])(?:[\s\.,;:：\)\]]|$)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, candidate_text, flags=re.IGNORECASE):
                label = match.group(1).upper()
                if label in valid_labels:
                    return label

    for candidate in candidates:
        normalized_candidate = _normalize_option_text(candidate)
        for label, option_text in option_map.items():
            normalized_option = _normalize_option_text(option_text)
            if normalized_option and len(normalized_option) > 1 and normalized_option in normalized_candidate:
                return label

    return fuzzy_matching(final_answer or raw_text).upper()


def _extract_numeric_prediction(pred):
    raw_text = str(pred or "").strip()
    final_answer = _extract_final_answer_text(raw_text)
    for candidate in [final_answer, raw_text]:
        if not candidate:
            continue
        match = NUMBER_RE.search(candidate)
        if match:
            return match.group(0)
    return None


def vsibench_process_results(doc, results):
    raw_prediction = results[0] if results else ""
    natural_answer_mode = _is_natural_answer_mode()
    doc['media_type'] = _doc_media_type(doc)
    doc['answer_type'] = _doc_answer_type(doc)
    if doc['answer_type'] == ANSWER_TYPE_MULTIPLE_CHOICE:
        restricted_prediction = _extract_choice_prediction(raw_prediction, doc) if natural_answer_mode else raw_prediction
        doc['natural_prediction'] = raw_prediction if natural_answer_mode else ""
        doc['restricted_prediction'] = restricted_prediction
        doc['prediction'] = restricted_prediction
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(doc['prediction'], doc['ground_truth'])
        doc["is_correct"] = bool(doc.get("accuracy", 0.0))
    elif doc['answer_type'] == ANSWER_TYPE_NUMERIC:
        restricted_prediction = _extract_numeric_prediction(raw_prediction) if natural_answer_mode else raw_prediction
        doc['natural_prediction'] = raw_prediction if natural_answer_mode else ""
        doc['restricted_prediction'] = restricted_prediction
        doc['prediction'] = restricted_prediction if restricted_prediction is not None else ""
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(doc['prediction']), to_float(doc['ground_truth']))
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


def _remove_question_type_outputs(output, question_type):
    prefix = f"{question_type}_"
    for key in list(output.keys()):
        if key.startswith(prefix):
            output.pop(key)


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


def _is_composite_member_result_key(key):
    for config in COMPOSITE_QUESTION_TYPES.values():
        for member in config["members"]:
            if key.startswith(f"{member}_"):
                return True
    return False


def _overall_keys_for_outputs(output):
    overall_keys = [key for key in TOP_LEVEL_RESULT_KEYS if key in output]
    seen = set(overall_keys)
    for key in output.keys():
        if key in seen or _is_composite_member_result_key(key):
            continue
        overall_keys.append(key)
        seen.add(key)
    return overall_keys


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

    for config in COMPOSITE_QUESTION_TYPES.values():
        per_composite = results[results["question_type"].isin(config["members"])]
        if len(per_composite) == 0:
            continue
        score = _mean_row_score(per_composite)
        if score is not None:
            output[config["metric_key"]] = score
        if not config.get("include_members", True):
            for member in config["members"]:
                _remove_question_type_outputs(output, member)

    overall_keys = _overall_keys_for_outputs(output)
    if overall_keys:
        output["overall"] = sum(float(output[key]) for key in overall_keys) / len(overall_keys)
    return output


def _score_to_percent(score):
    return float(score) * 100.


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
