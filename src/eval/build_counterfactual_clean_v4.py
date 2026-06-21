from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataclean_v4_utils import clean_string
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "17_COUNTERFACTUAL_TASK_BUILD_CLEAN_V4"

POSITIVE_TERMS = ["beat", "beats", "raised", "growth", "profit", "upgrade", "strong", "surge", "outperform", "higher"]
NEGATIVE_TERMS = ["miss", "missed", "lawsuit", "decline", "loss", "warning", "downgrade", "weak", "lower", "fell", "recall"]
BEARISH_TERMS = ("BEARISH", "DOWN", "BELOW", "SELL", "DEATH_CROSS")
BULLISH_TERMS = ("BULLISH", "UP", "ABOVE", "BUY", "GOLDEN_CROSS")


def parse_pack(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(clean_string(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def has_terms(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in terms)


def neutralize_terms(text: str, terms: list[str]) -> str:
    out = text
    for term in terms:
        out = re.sub(rf"\b{re.escape(term)}\b", "neutral update", out, flags=re.IGNORECASE)
    return out


def evidence_text(items: list[dict[str, Any]]) -> tuple[str, str]:
    headlines = []
    bodies = []
    for item in items:
        headlines.append(clean_string(item.get("headline")))
        bodies.append(clean_string(item.get("body_excerpt")))
    return " | ".join(x for x in headlines if x), "\n".join(x for x in bodies if x)


def signal_text(token: Any) -> str:
    if isinstance(token, dict):
        return " ".join(clean_string(token.get(k)) for k in ["token", "direction_prior", "rule"])
    return clean_string(token)


def neutralize_signals(tokens: list[Any], terms: tuple[str, ...], neutral_token: str) -> list[Any]:
    kept = [token for token in tokens if not any(term in signal_text(token).upper() for term in terms)]
    kept.append({"token": neutral_token, "direction_prior": "neutralized", "strength": "counterfactual", "rule": "clean_v4_counterfactual"})
    return kept


def build_task(row: pd.Series, cf_type: str) -> dict[str, Any] | None:
    pack = parse_pack(row.get("evidence_pack_json"))
    company = [item for item in pack.get("company_evidence", []) if isinstance(item, dict)]
    context = [item for item in pack.get("context_evidence", []) if isinstance(item, dict)]
    tokens = [item for item in pack.get("technical_signals", []) if isinstance(item, dict)]
    original_headline, original_body = evidence_text([*company, *context])
    if not original_body:
        original_body = clean_string(row.get("clean_context_text"))
    cf_headline = original_headline
    cf_body = original_body
    cf_tokens = tokens
    expected = ""

    if cf_type == "remove_positive_evidence":
        if not has_terms(f"{original_headline} {original_body}", POSITIVE_TERMS):
            return None
        cf_headline = neutralize_terms(cf_headline, POSITIVE_TERMS)
        cf_body = neutralize_terms(cf_body, POSITIVE_TERMS)
        expected = "up_decrease"
    elif cf_type == "remove_negative_evidence":
        if not has_terms(f"{original_headline} {original_body}", NEGATIVE_TERMS):
            return None
        cf_headline = neutralize_terms(cf_headline, NEGATIVE_TERMS)
        cf_body = neutralize_terms(cf_body, NEGATIVE_TERMS)
        expected = "down_decrease"
    elif cf_type == "remove_all_company_evidence":
        if not company:
            return None
        _, context_body = evidence_text(context)
        cf_headline = "company-specific evidence removed"
        cf_body = context_body or "company-specific evidence removed"
        expected = "up_decrease" if has_terms(original_body, POSITIVE_TERMS) else "down_decrease"
    elif cf_type == "neutralize_bearish_technical":
        if not any(any(term in signal_text(token).upper() for term in BEARISH_TERMS) for token in tokens):
            return None
        cf_tokens = neutralize_signals(tokens, BEARISH_TERMS, "TECHNICAL_BEARISH_SIGNAL_NEUTRALIZED")
        expected = "down_decrease"
    elif cf_type == "neutralize_bullish_technical":
        if not any(any(term in signal_text(token).upper() for term in BULLISH_TERMS) for token in tokens):
            return None
        cf_tokens = neutralize_signals(tokens, BULLISH_TERMS, "TECHNICAL_BULLISH_SIGNAL_NEUTRALIZED")
        expected = "up_decrease"
    else:
        return None

    return {
        "sample_id": row.get("sample_id"),
        "ticker": row.get("ticker"),
        "event_date": str(row.get("event_date")),
        "split": row.get("split"),
        "counterfactual_type": cf_type,
        "expected_direction": expected,
        "original_headline": original_headline,
        "original_body": original_body,
        "original_technical_event_tokens_json": json.dumps(tokens, ensure_ascii=False),
        "counterfactual_headline": cf_headline,
        "counterfactual_body": cf_body,
        "counterfactual_technical_event_tokens_json": json.dumps(cf_tokens, ensure_ascii=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", "--output-tasks", dest="output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default="outputs/manifests/17_COUNTERFACTUAL_TASK_BUILD_CLEAN_V4.manifest.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-per-type", type=int, default=1)
    args = parser.parse_args()

    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    if args.predictions and Path(args.predictions).exists() and len(contexts):
        preds = pd.read_parquet(args.predictions)
        if "sample_id" in preds.columns:
            contexts = contexts[contexts["sample_id"].astype(str).isin(set(preds["sample_id"].astype(str)))].copy()
    df = contexts[contexts["split"].eq("test")].sort_values(["event_date", "sample_id"]).copy() if len(contexts) else pd.DataFrame()
    cf_types = [
        "remove_positive_evidence",
        "remove_negative_evidence",
        "remove_all_company_evidence",
        "neutralize_bearish_technical",
        "neutralize_bullish_technical",
    ]
    per_type = max(1, args.limit // len(cf_types))
    rows: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    for cf_type in cf_types:
        count = 0
        for _, row in df.iterrows():
            task = build_task(row, cf_type)
            if task is None:
                continue
            rows.append(task)
            count += 1
            if count >= per_type:
                break
        type_counts[cf_type] = count

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.output).suffix.lower() == ".jsonl":
        with open(args.output, "w", encoding="utf-8") as f:
            for row in out.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        out.to_parquet(args.output, index=False)
    failures: list[str] = []
    if out.empty:
        failures.append("no counterfactual tasks generated")
    if len(out) and set(out["split"].dropna()) != {"test"}:
        failures.append("counterfactual tasks contain non-test split")
    for cf_type, count in type_counts.items():
        if count < args.min_per_type:
            failures.append(f"{cf_type} tasks {count} < {args.min_per_type}")
    metrics = {
        "tasks_written": int(len(out)),
        "samples_used": int(out["sample_id"].nunique()) if len(out) else 0,
        "type_counts": type_counts,
        "limit": args.limit,
        "min_per_type": args.min_per_type,
    }
    write_json(args.metrics, metrics)
    manifest_inputs = [args.output, args.metrics]
    if args.predictions:
        manifest_inputs.append(args.predictions)
    write_manifest(args.manifest, manifest_inputs, STEP)
    status = "PASS" if not failures else "FAIL"
    inputs = [args.contexts] + ([args.predictions] if args.predictions else [])
    write_status(args.status, STEP, status, inputs, [args.output, args.metrics, args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
