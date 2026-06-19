from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.parse_and_validate_rationale import parse_llm_json_strict
from src.utils.artifacts import write_json


FORECAST_KEYS = ["Strong Down", "Mild Down", "Neutral", "Mild Up", "Strong Up"]
ACTION_VALUES = {"long", "short", "hold"}
LEAK_PATTERNS = [
    re.compile(r"\brealized\s+(label|return|price)\b", re.IGNORECASE),
    re.compile(r"\btrue\s+(label|return|price)\b", re.IGNORECASE),
    re.compile(r"\bground[-\s]?truth\s+(label|return|price)\b", re.IGNORECASE),
    re.compile(r"\bfuture\s+(label|return|price)\b", re.IGNORECASE),
]


def iter_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as exc:
                yield line_no, {"__row_parse_error__": f"{type(exc).__name__}: {exc}", "raw_output": line}


def rationale_object_ok(value: Any, required_text_key: str) -> bool:
    if not isinstance(value, dict):
        return False
    text = value.get(required_text_key)
    return (
        isinstance(text, str)
        and bool(text.strip())
        and value.get("direction") in {"positive", "negative", "neutral"}
        and value.get("strength") in {"weak", "medium", "strong"}
    )


def strict_schema_errors(data: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["output is not a JSON object"]
    news = data.get("news_rationale")
    if not isinstance(news, list) or not news or len(news) > 2:
        errors.append("news_rationale must contain 1-2 items")
    elif not all(rationale_object_ok(item, "factor") for item in news):
        errors.append("news_rationale items must be factor/direction/strength objects")
    tech = data.get("technical_rationale")
    if not isinstance(tech, list) or not tech or len(tech) > 2:
        errors.append("technical_rationale must contain 1-2 items")
    elif not all(rationale_object_ok(item, "signal") for item in tech):
        errors.append("technical_rationale items must be signal/direction/strength objects")
    conflict = data.get("conflict_resolution")
    if not isinstance(conflict, str) or not conflict.strip():
        errors.append("conflict_resolution must be a non-empty string")
    elif len(conflict.split()) > 35:
        errors.append("conflict_resolution must be <= 35 words")
    risk = data.get("risk_note")
    if not isinstance(risk, str) or not risk.strip():
        errors.append("risk_note must be a non-empty string")
    elif len(risk.split()) > 20:
        errors.append("risk_note must be <= 20 words")
    dist = data.get("forecast_distribution")
    if not isinstance(dist, dict):
        errors.append("forecast_distribution must be an object")
    else:
        if set(dist.keys()) != set(FORECAST_KEYS):
            errors.append("forecast_distribution must contain exactly the title-case forecast keys")
        values = []
        for key in FORECAST_KEYS:
            value = dist.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"forecast_distribution.{key} must be finite numeric")
            else:
                values.append(float(value))
        if len(values) == len(FORECAST_KEYS) and not 0.99 <= sum(values) <= 1.01:
            errors.append(f"forecast_distribution must sum to [0.99, 1.01], got {sum(values):.6f}")
    if data.get("action") not in ACTION_VALUES:
        errors.append("action must be one of long, short, hold")
    return errors


def forecast_sum_error(data: Any) -> float | None:
    if not isinstance(data, dict) or not isinstance(data.get("forecast_distribution"), dict):
        return None
    dist = data["forecast_distribution"]
    values = []
    for key in FORECAST_KEYS:
        value = dist.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            return None
        values.append(float(value))
    return abs(sum(values) - 1.0)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return float(sorted_values[idx])


def has_explicit_label_leak(record: dict[str, Any], raw_output: str) -> bool:
    if any(pattern.search(raw_output or "") for pattern in LEAK_PATTERNS):
        return True
    label = record.get("label_5")
    if not isinstance(label, str) or not label:
        return False
    # The schema itself contains title-case label names, so only snake-case label mentions are treated as leaks here.
    return label in raw_output


def validate_file(path: str | Path) -> dict[str, Any]:
    records = list(iter_jsonl(path))
    parsed_count = 0
    schema_count = 0
    action_valid = 0
    leak_count = 0
    output_tokens: list[int] = []
    sum_errors: list[float] = []
    news_counts: list[int] = []
    tech_counts: list[int] = []
    examples: list[dict[str, Any]] = []

    for line_no, record in records:
        raw_output = str(record.get("raw_output", record.get("raw_text", "")))
        output_tokens.append(int(record.get("output_tokens_est") or max(1, len(raw_output) // 4)))
        parsed = None if "__row_parse_error__" in record else parse_llm_json_strict(raw_output)
        if parsed is not None:
            parsed_count += 1
        errors = strict_schema_errors(parsed)
        if not errors:
            schema_count += 1
        elif len(examples) < 20:
            examples.append({"line": line_no, "sample_id": record.get("sample_id"), "errors": errors, "raw_output_prefix": raw_output[:300]})
        if isinstance(parsed, dict):
            if parsed.get("action") in ACTION_VALUES:
                action_valid += 1
            if isinstance(parsed.get("news_rationale"), list):
                news_counts.append(len(parsed["news_rationale"]))
            if isinstance(parsed.get("technical_rationale"), list):
                tech_counts.append(len(parsed["technical_rationale"]))
        err = forecast_sum_error(parsed)
        if err is not None:
            sum_errors.append(err)
        if has_explicit_label_leak(record, raw_output):
            leak_count += 1

    num_rows = len(records)
    raw_parse_ok_rate = parsed_count / max(1, num_rows)
    raw_schema_ok_rate = schema_count / max(1, num_rows)
    explicit_label_leak_rate = leak_count / max(1, num_rows)
    summary = {
        "num_rows": num_rows,
        "raw_parse_ok_rate": raw_parse_ok_rate,
        "raw_schema_ok_rate": raw_schema_ok_rate,
        "invalid_json_rate": 1.0 - raw_parse_ok_rate,
        "explicit_label_leak_rate": explicit_label_leak_rate,
        "avg_output_tokens_est": sum(output_tokens) / max(1, len(output_tokens)),
        "forecast_distribution_sum_error_mean": sum(sum_errors) / max(1, len(sum_errors)) if sum_errors else None,
        "forecast_distribution_sum_error_p95": percentile(sum_errors, 0.95),
        "action_valid_rate": action_valid / max(1, num_rows),
        "news_rationale_item_count_mean": sum(news_counts) / max(1, len(news_counts)) if news_counts else None,
        "technical_rationale_item_count_mean": sum(tech_counts) / max(1, len(tech_counts)) if tech_counts else None,
        "schema_error_examples": examples,
    }
    return summary


def gate_failures(summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    checks = [
        ("raw_parse_ok_rate", ">=", args.raw_parse_ok_rate_min),
        ("raw_schema_ok_rate", ">=", args.raw_schema_ok_rate_min),
        ("avg_output_tokens_est", "<=", args.avg_output_tokens_max),
        ("invalid_json_rate", "<=", args.invalid_json_rate_max),
        ("explicit_label_leak_rate", "<=", args.explicit_label_leak_rate_max),
    ]
    for key, op, threshold in checks:
        value = summary.get(key)
        if value is None:
            failures.append(f"{key} is missing")
        elif op == ">=" and value < threshold:
            failures.append(f"{key}={value:.6f} < {threshold:.6f}")
        elif op == "<=" and value > threshold:
            failures.append(f"{key}={value:.6f} > {threshold:.6f}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict-no-autofix", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument("--raw-parse-ok-rate-min", type=float, default=0.85)
    parser.add_argument("--raw-schema-ok-rate-min", type=float, default=0.80)
    parser.add_argument("--avg-output-tokens-max", type=float, default=280.0)
    parser.add_argument("--invalid-json-rate-max", type=float, default=0.15)
    parser.add_argument("--explicit-label-leak-rate-max", type=float, default=0.03)
    args = parser.parse_args()

    summary = validate_file(args.input)
    failures = gate_failures(summary, args)
    summary["strict_no_autofix"] = bool(args.strict_no_autofix)
    summary["gate_failures"] = failures
    summary["gates_passed"] = not failures
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if args.enforce_gates and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
