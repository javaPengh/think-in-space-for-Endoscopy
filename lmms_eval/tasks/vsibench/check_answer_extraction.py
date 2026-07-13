import argparse
import importlib.util
import json
import sys
import types


class _NoOpLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _missing_module(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


def _install_lightweight_import_stubs():
    """Let this extraction helper import utils.py in local envs without full eval deps."""
    if _missing_module("loguru"):
        loguru_stub = types.ModuleType("loguru")
        loguru_stub.logger = _NoOpLogger()
        sys.modules["loguru"] = loguru_stub

    if _missing_module("datasets"):
        datasets_stub = types.ModuleType("datasets")
        datasets_stub.Dataset = object
        sys.modules["datasets"] = datasets_stub

    if _missing_module("pandas"):
        pandas_stub = types.ModuleType("pandas")
        pandas_stub.isna = lambda value: False
        sys.modules["pandas"] = pandas_stub

    if _missing_module("numpy"):
        numpy_stub = types.ModuleType("numpy")
        sys.modules["numpy"] = numpy_stub

    if _missing_module("yaml"):
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_load = lambda text: {"dataset_kwargs": {"cache_dir": "vsibench"}}
        sys.modules["yaml"] = yaml_stub


_install_lightweight_import_stubs()

try:
    from .utils import _extract_choice_prediction, _extract_final_answer_text, _extract_numeric_prediction, to_float
except ImportError:
    from utils import _extract_choice_prediction, _extract_final_answer_text, _extract_numeric_prediction, to_float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check the exact VSI-Bench answer extraction result for a model output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python lmms_eval/tasks/vsibench/check_answer_extraction.py --answer-type numeric --text "Final answer: 10."
  python lmms_eval/tasks/vsibench/check_answer_extraction.py --answer-type choice --options "A. left||B. right||C. up||D. down" --text "Final answer: C"
  python lmms_eval/tasks/vsibench/check_answer_extraction.py --answer-mode restricted --text "10."
  python lmms_eval/tasks/vsibench/check_answer_extraction.py --answer-type choice --options "A. left||B. right||C. up||D. down" --text "I choose right."

If --text is omitted, the script starts a paste-friendly interactive mode. Paste a
multi-line model output, then type a line containing only END.
""",
    )
    parser.add_argument("--answer-type", choices=["numeric", "choice", "both"], default="numeric", help="Which VSI-Bench extraction branch to run.")
    parser.add_argument("--answer-mode", choices=["natural", "restricted"], default="natural", help="natural requires a Final answer line; restricted parses raw direct output.")
    parser.add_argument("--text", help="Model output text to extract from. If omitted, stdin is used.")
    parser.add_argument("--end-marker", default="END", help="Interactive paste terminator line when --text is omitted.")
    parser.add_argument("--options", help="Multiple-choice options separated by '||' or newlines.")
    parser.add_argument("-o", "--option", action="append", default=[], help="One multiple-choice option line. Repeat as needed.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def read_text(args):
    if args.text is not None:
        return args.text
    if sys.stdin.isatty():
        print(f"Paste model output below. Finish with a line containing only {args.end_marker!r}.", file=sys.stderr)
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == args.end_marker:
                break
            lines.append(line)
        return "\n".join(lines)
    return sys.stdin.read()


def parse_options(args):
    options = []
    if args.options:
        options.extend(part.strip() for part in args.options.replace("\r\n", "\n").split("||") for part in part.splitlines() if part.strip())
    options.extend(option.strip() for option in args.option if option.strip())
    return options


def extract(text, answer_type, answer_mode, options):
    require_final_answer = answer_mode == "natural"
    result = {
        "answer_mode": answer_mode,
        "require_final_answer": require_final_answer,
        "raw_input": text,
        "final_answer_text": _extract_final_answer_text(text),
    }
    if answer_type in {"numeric", "both"}:
        numeric_prediction = _extract_numeric_prediction(text, require_final_answer=require_final_answer)
        result["numeric_prediction"] = numeric_prediction
        result["numeric_prediction_float"] = None if numeric_prediction is None else to_float(numeric_prediction)
    if answer_type in {"choice", "both"}:
        result["choice_options_used"] = options
        result["choice_prediction"] = _extract_choice_prediction(text, {"options": options}, require_final_answer=require_final_answer)
    return result


def print_result(result, as_json):
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"answer_mode: {result.get('answer_mode')}")
    print(f"require_final_answer: {result.get('require_final_answer')}")
    print(f"final_answer_text_raw: {result.get('final_answer_text')!r}")
    if "numeric_prediction" in result:
        print(f"numeric_prediction_raw: {result.get('numeric_prediction')!r}")
        print(f"numeric_prediction_for_scoring: {result.get('numeric_prediction_float')!r}")
    if "choice_prediction" in result:
        print(f"choice_options_used: {result.get('choice_options_used')!r}")
        print(f"choice_prediction: {result.get('choice_prediction')!r}")


def main():
    args = parse_args()
    text = read_text(args)
    options = parse_options(args)
    if args.answer_type in {"choice", "both"} and not options:
        raise SystemExit("Choice extraction requires --options or repeated --option values, matching the sample's real options.")
    result = extract(text, args.answer_type, args.answer_mode, options)
    print_result(result, args.json)


if __name__ == "__main__":
    import json
    import urllib.request
    import urllib.error

    api_key = "sk-68a39855d0ec4d8ea23999d4d5ccd306"
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    payload = {
        "model": "qwen-plus",
        "messages": [
            {
                "role": "system",
                "content": "You are a strict answer extraction tool. Return only the answer or EXTRACTION_FAILED.",
            },
            {
                "role": "user",
                "content": (
                    "Extract the explicit final answer from this model output.\n"
                    "If no explicit answer exists, return exactly EXTRACTION_FAILED.\n\n"
                    "Model output:\n"
                    "After comparing the options, the answer is B."
                ),
            },
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 32,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print("HTTPError:", e.code)
        print(e.read().decode("utf-8", errors="replace"))
        raise
    except Exception as e:
        print(type(e).__name__, e)
        raise

    print(json.dumps(data, ensure_ascii=False, indent=2))
    print("\nExtracted content:")
    print(data["choices"][0]["message"]["content"])
