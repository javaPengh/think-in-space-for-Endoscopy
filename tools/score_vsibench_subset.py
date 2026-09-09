import argparse
import csv
import glob
import json
import re
from pathlib import Path


MRA_METRIC = "MRA:.5:.95:.05"
QUESTION_TYPE_ORDER = [
    "action_order",
    "object_order",
    "route_planning",
    "object_rel_direction",
    "fold_rel_depth",
    "object_counting",
    "action_counting",
    "polyp_size_estimation",
]
ID_KEYS = ["id", "question_id", "Question_ID", "questionId", "doc_id", "ID"]
TARGET_IDS = [
    "359", "360", "361", "362", "363", "364", "365", "377", "378", "379", "380", "381", "382", "383", "384", "390", "391", "394", "399", "400", "401", "402", "403",
    "1494", "1497", "1500", "1503", "1505", "1508", "1510", "1513", "1516", "1519", "1522",
    "1846", "1847", "1848", "1849", "1850", "1851", "1852", "1853", "1854", "1855", "1857", "1858", "1859", "1868", "1869", "1870", "1871", "1872", "1873", "1876", "1877", "1878", "1879", "1880", "1881", "1882",
    "605", "606", "607", "608", "609", "610", "611", "612", "613", "614", "615", "616",
    "725", "726", "727", "728", "729", "730", "731", "732", "733", "734", "735", "736", "737", "738", "739", "740", "742", "743", "744", "745", "746", "747", "748", "749", "750", "751", "752", "753", "754", "755", "756", "757", "758", "759", "760", "761", "762", "763", "764", "765", "766", "767", "768", "769", "770", "771", "772", "773", "774", "775", "776", "777", "778", "779", "780", "781", "782", "783", "784", "785", "786", "788", "789", "790", "791", "792", "793", "794", "795", "796", "797", "798", "799", "800", "801", "802", "803", "804", "805", "806", "807", "808", "809", "810", "812", "813", "814", "815", "816", "817", "818", "819", "820", "821", "822", "823", "824",
    "2338", "2339", "2342", "2344", "2346", "2348",
    "2779", "2782", "2783", "2786", "2787", "2790", "2791", "2794", "2795", "2798", "2799", "2802", "2803", "2806", "2808", "2809", "2812", "2813", "2816", "2817", "2819", "2822", "2823", "2826", "2827", "2830", "2831", "2834", "2835",
    "3348", "3350", "3352", "3354", "3356", "3358", "3360", "3362", "3364", "3366", "3368", "3370", "3372", "3374", "3376", "3378", "3380", "3382", "3384", "3386", "3388", "3390", "3392", "3394", "3396", "3398", "3400", "3402",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "21", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "2544", "2545", "2546", "2547", "2548", "2549", "2550", "2551", "2552", "2553", "2554", "2556", "2557", "2559", "2560", "2561", "2562", "2563", "2565", "2566", "2567",
]


def main():
    parser = argparse.ArgumentParser(description="Score existing VSI-Bench per-sample result files on a selected question subset.")
    parser.add_argument("--ids", help="Optional txt/csv/json/xlsx file containing selected question ids. If omitted, the built-in 287 ids are used.")
    parser.add_argument("--id-column", default="id", help="Column/key name for ids in csv/json/xlsx when --ids is provided.")
    parser.add_argument("--results", nargs="+", help="Optional explicit vsibench.json files or glob patterns.")
    parser.add_argument("--log-root", default="logs", help="Root directory that contains dated evaluation folders.")
    parser.add_argument("--dates", nargs="+", default=["20260813", "20260814", "20260815", "20260816"], help="Date folders under --log-root to scan when --results is not provided.")
    parser.add_argument("--output", default="subset_scores.xlsx", help="Output .xlsx path.")
    parser.add_argument("--strict", action="store_true", help="Fail if any result file misses selected ids.")
    args = parser.parse_args()

    target_ids = read_ids(args.ids, args.id_column) if args.ids else TARGET_IDS
    result_files = resolve_result_files(args.results, args.log_root, args.dates)
    if not result_files:
        raise SystemExit("No result files found. Check --log-root/--dates, or pass explicit files with --results.")

    rows = []
    detail_rows = []
    all_question_types = []
    for result_file in result_files:
        row, details, missing_ids, question_types = score_result_file(result_file, target_ids)
        rows.append(row)
        detail_rows.extend(details)
        all_question_types.extend(question_types)
        if missing_ids:
            print(f"WARNING: {result_file} missing {len(missing_ids)} ids: {', '.join(missing_ids[:20])}")
            if args.strict:
                raise SystemExit("Missing selected ids. Remove --strict to write partial results.")

    score_fields = ["model"] + ordered_question_types(all_question_types) + ["overall"]
    detail_fields = ["model", "id", "question", "options", "truth", "answer"]
    write_xlsx(Path(args.output), rows, score_fields, detail_rows, detail_fields)
    print(f"Wrote {args.output}")


def score_result_file(path, target_ids):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    logs = data.get("logs")
    if not isinstance(logs, list):
        raise SystemExit(f"{path} does not have a top-level logs list.")

    logs_by_id = {}
    for log in logs:
        score_doc = log.get("vsibench_score") or {}
        doc = log.get("doc") or {}
        item_id = first_id(score_doc, doc, log)
        if item_id:
            logs_by_id[item_id] = log

    scores_by_type = {}
    missing_ids = []
    details = []
    model = model_name(data, path)
    for item_id in target_ids:
        log = logs_by_id.get(item_id)
        if log is None:
            missing_ids.append(item_id)
            continue
        score_doc = log.get("vsibench_score") or log.get("doc") or {}
        require_keys(score_doc, ["question", "options", "ground_truth", "prediction"], path, item_id)
        question_type = str(score_doc.get("question_type") or "").strip()
        score = item_score(score_doc)
        if not question_type:
            raise SystemExit(f"{path} id={item_id} has no question_type.")
        if score is None:
            raise SystemExit(f"{path} id={item_id} has no supported score field.")
        scores_by_type.setdefault(question_type, []).append(score)
        details.append(
            {
                "model": model,
                "id": item_id,
                "question": cell_value(score_doc.get("question")),
                "options": cell_value(score_doc.get("options")),
                "truth": cell_value(score_doc.get("ground_truth")),
                "answer": cell_value(score_doc.get("prediction")),
            }
        )

    row = {"model": model}
    question_type_means = []
    for question_type in sorted(scores_by_type):
        value = mean(scores_by_type[question_type])
        row[question_type] = value
        question_type_means.append(value)
    row["overall"] = mean(question_type_means)
    return row, details, missing_ids, list(scores_by_type.keys())


def item_score(score_doc):
    answer_type = str(score_doc.get("answer_type") or "").lower()
    if answer_type in {"multiple_choice", "mca", "mcq", "choice"}:
        return percent(score_doc.get("accuracy"))
    if answer_type in {"numeric", "na", "number", "numerical"}:
        return percent(score_doc.get(MRA_METRIC))
    if "accuracy" in score_doc:
        return percent(score_doc.get("accuracy"))
    if MRA_METRIC in score_doc:
        return percent(score_doc.get(MRA_METRIC))
    return None


def require_keys(data, keys, path, item_id):
    missing = [key for key in keys if key not in data]
    if missing:
        raise SystemExit(f"{path} id={item_id} missing required fields: {', '.join(missing)}")


def cell_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def percent(value):
    if value in (None, ""):
        return None
    try:
        return float(value) * 100.0
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [float(value) for value in values if value not in ("", None)]
    if not values:
        return ""
    return round(sum(values) / len(values), 6)


def read_ids(path, id_column):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        values = ids_from_json(json.loads(path.read_text(encoding="utf-8")), id_column)
    elif suffix in {".csv", ".tsv"}:
        values = ids_from_delimited(path, id_column, "\t" if suffix == ".tsv" else ",")
    elif suffix in {".xlsx", ".xls"}:
        values = ids_from_excel(path, id_column)
    else:
        values = path.read_text(encoding="utf-8-sig").splitlines()

    ids = []
    seen = set()
    for value in values:
        item_id = normalize_id(value)
        if item_id and item_id not in seen:
            ids.append(item_id)
            seen.add(item_id)
    if not ids:
        raise SystemExit(f"No ids found in {path}.")
    return ids


def ids_from_json(data, id_column):
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        column = choose_id_column(data[0].keys(), id_column)
        return [row.get(column, "") for row in data]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if id_column and isinstance(data.get(id_column), list):
            return data[id_column]
        return list(data.values())
    return []


def ids_from_delimited(path, id_column, delimiter):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=delimiter))
    if not rows:
        return []

    header = [str(cell).strip() for cell in rows[0]]
    if id_column or any(column in ID_KEYS for column in header):
        column = choose_id_column(header, id_column)
        index = header.index(column)
        return [row[index] for row in rows[1:] if len(row) > index]

    return [cell for row in rows for cell in row if str(cell).strip()]


def ids_from_excel(path, id_column):
    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit("Reading Excel id files requires pandas/openpyxl. Save ids as txt/csv, or install those dependencies.") from error
    frame = pd.read_excel(path)
    column = choose_id_column(frame.columns, id_column)
    return frame[column].tolist()


def choose_id_column(columns, requested):
    columns = [str(column) for column in columns]
    if requested:
        if requested not in columns:
            raise SystemExit(f"Column '{requested}' was not found. Available columns: {', '.join(columns)}")
        return requested
    for key in ID_KEYS:
        if key in columns:
            return key
    return columns[0]


def expand_result_paths(patterns):
    paths = []
    seen = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match).expanduser().resolve()
            if path.is_file() and str(path) not in seen:
                paths.append(path)
                seen.add(str(path))
    return sorted(paths)


def resolve_result_files(results, log_root, dates):
    if results:
        return expand_result_paths(results)
    patterns = [str(Path(log_root) / date / "vsibench" / "*" / "vsibench.json") for date in dates]
    return expand_result_paths(patterns)


def first_id(*sources):
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ID_KEYS:
            value = source.get(key)
            if value is not None and str(value).strip():
                return normalize_id(value)
    return ""


def normalize_id(value):
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        return str(int(float(text)))
    return text


def model_name(data, path):
    args = data.get("args") or {}
    suffix = str(args.get("log_samples_suffix") or "").strip()
    if suffix:
        return suffix
    model = str(args.get("model") or "").strip()
    if model:
        return model
    return Path(path).parent.name


def ordered_question_types(question_types):
    seen = set(question_types)
    ordered = [question_type for question_type in QUESTION_TYPE_ORDER if question_type in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def write_xlsx(path, score_rows, score_fields, detail_rows, detail_fields):
    if path.suffix.lower() != ".xlsx":
        raise SystemExit("--output must end with .xlsx because the result contains multiple sheets.")
    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit("Writing .xlsx requires pandas and openpyxl in the Python environment.") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(score_rows, columns=score_fields).to_excel(writer, index=False, sheet_name="Scores")
        pd.DataFrame(detail_rows, columns=detail_fields).to_excel(writer, index=False, sheet_name="Details")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
