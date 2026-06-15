import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "eval_question_matrix.json"
DEFAULT_CSV = ROOT / "docs" / "internvl3_5_8b_uniform16f_vmax1_answer_compare.csv"
DEFAULT_HTML = ROOT / "docs" / "internvl3_5_8b_uniform16f_vmax1_answer_compare.html"

TARGET_MODEL = "internvl3_5_8b"
TARGET_SAMPLING = "uniform_16f"
TARGET_NOTE = "video_max_num=1"

FIELDNAMES = [
    "doc_id",
    "question_type",
    "answer_type",
    "question",
    "ground_truth",
    "options",
    "natural_output",
    "natural_answer",
    "direct_answer",
    "answer_same",
    "natural_score",
    "direct_score",
    "score_delta",
    "natural_status",
    "direct_status",
    "comparison_bucket",
    "natural_sample_path",
    "direct_sample_path",
]

BUCKET_ORDER = {
    "natural_better": 0,
    "direct_better": 1,
    "both_wrong_different_answer": 2,
    "both_correct_different_answer": 3,
    "same_answer_different_score": 4,
    "same_answer_same_score": 5,
    "missing_pair": 6,
}


def load_rows(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f).get("rows", [])


def target_rows(rows):
    return [
        row
        for row in rows
        if row.get("model") == TARGET_MODEL
        and row.get("sampling_strategy") == TARGET_SAMPLING
        and row.get("note") == TARGET_NOTE
    ]


def path_kind(path_rows):
    sample_path = path_rows[0].get("sample_path", "")
    if "_natural_" in sample_path:
        return "natural"
    if any(row.get("natural_prediction") for row in path_rows):
        return "natural"
    return "direct"


def path_sort_key(path_rows):
    sample_path = path_rows[0].get("sample_path", "")
    timestamp = path_rows[0].get("timestamp", "")
    return (str(timestamp), str(sample_path))


def select_latest_runs(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("sample_path", "")].append(row)

    by_kind = defaultdict(list)
    for path_rows in grouped.values():
        by_kind[path_kind(path_rows)].append(path_rows)

    natural_runs = sorted(by_kind.get("natural", []), key=path_sort_key)
    direct_runs = sorted(by_kind.get("direct", []), key=path_sort_key)
    if not natural_runs:
        raise ValueError("No natural run found for target filter.")
    if not direct_runs:
        raise ValueError("No restricted/direct run found for target filter.")
    return natural_runs[-1], direct_runs[-1]


def index_by_doc_id(rows):
    return {row.get("doc_id"): row for row in rows}


def normalize_for_compare(value, answer_type):
    text = str(value if value is not None else "").strip()
    if answer_type == "multiple_choice":
        return ("choice", re.sub(r"\s+", "", text).upper())
    if answer_type == "numeric":
        try:
            return ("number", float(text))
        except ValueError:
            return ("text", text)
    return ("text", text)


def same_answer(natural_answer, direct_answer, answer_type):
    left = normalize_for_compare(natural_answer, answer_type)
    right = normalize_for_compare(direct_answer, answer_type)
    if left[0] == "number" and right[0] == "number":
        return abs(left[1] - right[1]) <= 1e-6
    return left == right


def score_value(row):
    if not row:
        return None
    value = row.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_score(value):
    if value is None:
        return ""
    return f"{value:.6g}"


def row_status(row):
    if not row:
        return "missing"
    answer_type = row.get("answer_type")
    if answer_type == "multiple_choice":
        return "correct" if row.get("is_correct") else "incorrect"
    if answer_type == "numeric":
        return "scored" if row.get("is_scored") else "not_scored"
    score = score_value(row)
    return "scored" if score and score > 0 else "not_scored"


def positive_score(row):
    score = score_value(row)
    return score is not None and score > 0


def comparison_bucket(natural_row, direct_row, answers_same):
    if not natural_row or not direct_row:
        return "missing_pair"

    natural_score = score_value(natural_row)
    direct_score = score_value(direct_row)
    if natural_score is None or direct_score is None:
        return "missing_pair"

    delta = natural_score - direct_score
    if answers_same:
        return "same_answer_same_score" if abs(delta) <= 1e-9 else "same_answer_different_score"
    if delta > 1e-9:
        return "natural_better"
    if delta < -1e-9:
        return "direct_better"
    if positive_score(natural_row) and positive_score(direct_row):
        return "both_correct_different_answer"
    return "both_wrong_different_answer"


def sortable_doc_id(doc_id):
    try:
        return int(doc_id)
    except (TypeError, ValueError):
        return str(doc_id)


def build_comparison_rows(natural_rows, direct_rows):
    natural_by_doc = index_by_doc_id(natural_rows)
    direct_by_doc = index_by_doc_id(direct_rows)
    doc_ids = sorted(set(natural_by_doc) | set(direct_by_doc), key=sortable_doc_id)

    comparison_rows = []
    for doc_id in doc_ids:
        natural_row = natural_by_doc.get(doc_id)
        direct_row = direct_by_doc.get(doc_id)
        source = natural_row or direct_row or {}
        answer_type = source.get("answer_type", "")

        natural_answer = natural_row.get("restricted_prediction", "") if natural_row else ""
        direct_answer = direct_row.get("restricted_prediction", "") if direct_row else ""
        answers_same = bool(natural_row and direct_row and same_answer(natural_answer, direct_answer, answer_type))
        bucket = comparison_bucket(natural_row, direct_row, answers_same)

        natural_score = score_value(natural_row)
        direct_score = score_value(direct_row)
        score_delta = None if natural_score is None or direct_score is None else natural_score - direct_score

        comparison_rows.append(
            {
                "doc_id": doc_id,
                "question_type": source.get("question_type", ""),
                "answer_type": answer_type,
                "question": source.get("question", ""),
                "ground_truth": source.get("ground_truth", ""),
                "options": source.get("options", ""),
                "natural_output": natural_row.get("natural_prediction", "") if natural_row else "",
                "natural_answer": natural_answer,
                "direct_answer": direct_answer,
                "answer_same": str(answers_same).lower(),
                "natural_score": fmt_score(natural_score),
                "direct_score": fmt_score(direct_score),
                "score_delta": fmt_score(score_delta),
                "natural_status": row_status(natural_row),
                "direct_status": row_status(direct_row),
                "comparison_bucket": bucket,
                "natural_sample_path": natural_row.get("sample_path", "") if natural_row else "",
                "direct_sample_path": direct_row.get("sample_path", "") if direct_row else "",
            }
        )
    return comparison_rows


def sort_comparison_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            BUCKET_ORDER.get(row["comparison_bucket"], 99),
            row.get("question_type", ""),
            sortable_doc_id(row.get("doc_id")),
        ),
    )


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def number_from_row(row, field):
    try:
        return float(row.get(field, ""))
    except ValueError:
        return None


def average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summary_stats(rows):
    natural_scores = [number_from_row(row, "natural_score") for row in rows]
    direct_scores = [number_from_row(row, "direct_score") for row in rows]
    buckets = Counter(row["comparison_bucket"] for row in rows)
    answer_same_count = sum(row["answer_same"] == "true" for row in rows)
    paired_count = sum(row["comparison_bucket"] != "missing_pair" for row in rows)
    return {
        "paired_count": paired_count,
        "row_count": len(rows),
        "natural_avg": average(natural_scores),
        "direct_avg": average(direct_scores),
        "natural_better": buckets.get("natural_better", 0),
        "direct_better": buckets.get("direct_better", 0),
        "answer_same": answer_same_count,
        "answer_different": paired_count - answer_same_count,
        "buckets": buckets,
    }


def grouped_stats(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["question_type"]].append(row)

    result = []
    for question_type, group_rows in sorted(groups.items()):
        stats = summary_stats(group_rows)
        result.append(
            {
                "question_type": question_type,
                "count": len(group_rows),
                "natural_avg": stats["natural_avg"],
                "direct_avg": stats["direct_avg"],
                "natural_better": stats["natural_better"],
                "direct_better": stats["direct_better"],
                "answer_same": stats["answer_same"],
                "answer_different": stats["answer_different"],
            }
        )
    return result


def esc(value):
    return html.escape(str(value if value is not None else ""))


def score_html(value):
    return "" if value is None else f"{value:.2f}"


def render_html(rows, natural_path, direct_path):
    stats = summary_stats(rows)
    groups = grouped_stats(rows)
    bucket_items = "".join(
        f"<li><code>{esc(bucket)}</code>: {count}</li>"
        for bucket, count in sorted(stats["buckets"].items(), key=lambda item: BUCKET_ORDER.get(item[0], 99))
    )
    group_rows = "\n".join(
        "<tr>"
        f"<td>{esc(row['question_type'])}</td>"
        f"<td>{row['count']}</td>"
        f"<td>{score_html(row['natural_avg'])}</td>"
        f"<td>{score_html(row['direct_avg'])}</td>"
        f"<td>{row['natural_better']}</td>"
        f"<td>{row['direct_better']}</td>"
        f"<td>{row['answer_same']}</td>"
        f"<td>{row['answer_different']}</td>"
        "</tr>"
        for row in groups
    )
    table_rows = "\n".join(render_table_row(row) for row in rows)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>InternVL3.5 Natural vs Direct Answer Comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #64748b; }}
    .panel {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .metric {{ background: #f1f5f9; border-radius: 6px; padding: 10px; }}
    .metric strong {{ display: block; font-size: 20px; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; }}
    th {{ background: #e2e8f0; position: sticky; top: 0; z-index: 1; }}
    .text {{ min-width: 260px; max-width: 520px; white-space: pre-wrap; }}
    .small {{ font-size: 12px; }}
    .bucket {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .natural_better {{ background: #ecfdf5; }}
    .direct_better {{ background: #fff7ed; }}
    .both_wrong_different_answer {{ background: #fef2f2; }}
    .both_correct_different_answer {{ background: #eff6ff; }}
    details summary {{ cursor: pointer; color: #2563eb; }}
  </style>
</head>
<body>
  <h1>InternVL3.5 8B uniform_16f video_max_num=1: Natural vs Direct</h1>
  <p class="muted">Natural run: {esc(natural_path)}<br>Direct run: {esc(direct_path)}</p>
  <section class="panel summary">
    <div class="metric">paired questions<strong>{stats['paired_count']} / {stats['row_count']}</strong></div>
    <div class="metric">natural avg<strong>{score_html(stats['natural_avg'])}</strong></div>
    <div class="metric">direct avg<strong>{score_html(stats['direct_avg'])}</strong></div>
    <div class="metric">natural better<strong>{stats['natural_better']}</strong></div>
    <div class="metric">direct better<strong>{stats['direct_better']}</strong></div>
    <div class="metric">same / different answers<strong>{stats['answer_same']} / {stats['answer_different']}</strong></div>
  </section>
  <section class="panel">
    <h2>Bucket Counts</h2>
    <ul>{bucket_items}</ul>
  </section>
  <section class="panel">
    <h2>By Question Type</h2>
    <table>
      <thead><tr><th>question_type</th><th>count</th><th>natural avg</th><th>direct avg</th><th>natural better</th><th>direct better</th><th>same answers</th><th>different answers</th></tr></thead>
      <tbody>{group_rows}</tbody>
    </table>
  </section>
  <section class="panel">
    <h2>Question-Level Comparison</h2>
    <table>
      <thead>
        <tr>
          <th>doc_id</th><th>question_type</th><th>answer_type</th><th>question</th><th>ground_truth</th><th>options</th>
          <th>natural_output</th><th>natural_answer</th><th>direct_answer</th><th>answer_same</th>
          <th>natural_score</th><th>direct_score</th><th>score_delta</th><th>status</th><th>bucket</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>
</body>
</html>
"""


def render_table_row(row):
    natural_output = esc(row["natural_output"])
    natural_cell = (
        f"<details><summary>{len(row['natural_output'])} chars</summary>{natural_output}</details>"
        if row["natural_output"]
        else ""
    )
    bucket = row["comparison_bucket"]
    return (
        f'<tr class="{esc(bucket)}">'
        f"<td>{esc(row['doc_id'])}</td>"
        f"<td>{esc(row['question_type'])}</td>"
        f"<td>{esc(row['answer_type'])}</td>"
        f"<td class=\"text\">{esc(row['question'])}</td>"
        f"<td>{esc(row['ground_truth'])}</td>"
        f"<td class=\"text small\">{esc(row['options'])}</td>"
        f"<td class=\"text small\">{natural_cell}</td>"
        f"<td>{esc(row['natural_answer'])}</td>"
        f"<td>{esc(row['direct_answer'])}</td>"
        f"<td>{esc(row['answer_same'])}</td>"
        f"<td>{esc(row['natural_score'])}</td>"
        f"<td>{esc(row['direct_score'])}</td>"
        f"<td>{esc(row['score_delta'])}</td>"
        f"<td class=\"small\">{esc(row['natural_status'])} / {esc(row['direct_status'])}</td>"
        f"<td class=\"bucket\">{esc(bucket)}</td>"
        "</tr>"
    )


def write_html(rows, natural_path, direct_path, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(rows, natural_path, direct_path), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare InternVL3.5 natural and direct answers for the target VSI-Bench run.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()

    rows = target_rows(load_rows(args.input))
    natural_rows, direct_rows = select_latest_runs(rows)
    comparison_rows = sort_comparison_rows(build_comparison_rows(natural_rows, direct_rows))
    natural_path = natural_rows[0].get("sample_path", "")
    direct_path = direct_rows[0].get("sample_path", "")

    write_csv(comparison_rows, args.csv)
    write_html(comparison_rows, natural_path, direct_path, args.html)

    stats = summary_stats(comparison_rows)
    print(f"selected rows: {len(rows)}")
    print(f"natural run: {natural_path} ({len(natural_rows)} rows)")
    print(f"direct run: {direct_path} ({len(direct_rows)} rows)")
    print(f"paired comparison rows: {stats['paired_count']} / {stats['row_count']}")
    print(f"wrote csv: {args.csv}")
    print(f"wrote html: {args.html}")


if __name__ == "__main__":
    main()
