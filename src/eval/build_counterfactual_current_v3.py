from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "15_COUNTERFACTUAL_TASK_BUILD_CURRENT_DATA"

POSITIVE_TERMS = [
    "beat",
    "beats",
    "raised",
    "raises",
    "growth",
    "profit",
    "profits",
    "record",
    "upgrade",
    "strong",
    "approval",
    "win",
    "surge",
    "higher",
    "outperform",
]
NEGATIVE_TERMS = [
    "miss",
    "missed",
    "lawsuit",
    "decline",
    "loss",
    "losses",
    "warning",
    "downgrade",
    "weak",
    "cut",
    "investigation",
    "bankruptcy",
    "lower",
    "fell",
    "plunge",
    "recall",
]
BEARISH_TOKEN_TERMS = ("BEARISH", "SELL", "LOWER", "DEATH_CROSS", "OVERSOLD", "DOWN", "BELOW")
BULLISH_TOKEN_TERMS = ("BULLISH", "BUY", "HIGHER", "GOLDEN_CROSS", "OVERBOUGHT", "UP", "ABOVE")


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    return value


def text_has_terms(text: Any, terms: list[str]) -> bool:
    low = str(text or "").lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in terms)


def neutralize_terms(text: Any, terms: list[str]) -> str:
    text = "" if text is None else str(text)
    for term in terms:
        text = re.sub(rf"\b{re.escape(term)}\b", "neutral update", text, flags=re.IGNORECASE)
    return text


def parse_tokens(value: Any) -> list[Any]:
    value = clean_value(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def token_name(token: Any) -> str:
    if isinstance(token, dict):
        return " ".join(str(token.get(k, "")) for k in ("token", "direction_prior", "rule"))
    return str(token)


def has_token_signal(tokens: list[Any], terms: tuple[str, ...]) -> bool:
    return any(any(term in token_name(token).upper() for term in terms) for token in tokens)


def neutralize_technical(tokens: list[Any], terms: tuple[str, ...], neutral_token: str) -> list[Any]:
    kept = [token for token in tokens if not any(term in token_name(token).upper() for term in terms)]
    kept.append({"token": neutral_token, "direction_prior": "neutralized", "strength": "counterfactual", "rule": "counterfactual signal neutralization"})
    return kept


def row_text(row: Any) -> tuple[str, str]:
    headline = clean_value(row.get("aggregated_headlines", row.get("headline", ""))) or ""
    body = clean_value(row.get("aggregated_body", row.get("body", ""))) or ""
    return str(headline), str(body)


def is_applicable(row: Any, cf_type: str) -> bool:
    headline, body = row_text(row)
    text = f"{headline} {body}"
    tokens = parse_tokens(row.get("technical_event_tokens_json"))
    if cf_type == "remove_negative_news":
        return text_has_terms(text, NEGATIVE_TERMS)
    if cf_type == "remove_positive_news":
        return text_has_terms(text, POSITIVE_TERMS)
    if cf_type == "neutralize_bearish_technical":
        return has_token_signal(tokens, BEARISH_TOKEN_TERMS)
    if cf_type == "neutralize_bullish_technical":
        return has_token_signal(tokens, BULLISH_TOKEN_TERMS)
    return False


def build_task(row: Any, cf_type: str, expected_direction: str) -> dict[str, Any]:
    tokens = parse_tokens(row.get("technical_event_tokens_json"))
    headline, body = row_text(row)
    cf_headline = str(headline)
    cf_body = str(body)
    cf_tokens = tokens

    if cf_type == "remove_negative_news":
        cf_headline = neutralize_terms(cf_headline, NEGATIVE_TERMS)
        cf_body = neutralize_terms(cf_body, NEGATIVE_TERMS)
    elif cf_type == "remove_positive_news":
        cf_headline = neutralize_terms(cf_headline, POSITIVE_TERMS)
        cf_body = neutralize_terms(cf_body, POSITIVE_TERMS)
    elif cf_type == "neutralize_bearish_technical":
        cf_tokens = neutralize_technical(tokens, BEARISH_TOKEN_TERMS, "TECHNICAL_BEARISH_SIGNAL_NEUTRALIZED")
    elif cf_type == "neutralize_bullish_technical":
        cf_tokens = neutralize_technical(tokens, BULLISH_TOKEN_TERMS, "TECHNICAL_BULLISH_SIGNAL_NEUTRALIZED")

    return {
        "sample_id": clean_value(row.get("sample_id")),
        "ticker": clean_value(row.get("ticker")),
        "event_date": str(clean_value(row.get("event_date"))),
        "split": clean_value(row.get("split", "test")),
        "counterfactual_type": cf_type,
        "expected_direction": expected_direction,
        "original_headline": str(headline),
        "original_body": str(body),
        "original_technical_event_tokens_json": json.dumps(tokens, ensure_ascii=False),
        "counterfactual_headline": cf_headline,
        "counterfactual_body": cf_body,
        "counterfactual_technical_event_tokens_json": json.dumps(cf_tokens, ensure_ascii=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="data/processed/ticker_date_contexts_h1_v2_targets.parquet")
    parser.add_argument("--tokens", default="data/indicators/technical_event_tokens_h1_v2.parquet")
    parser.add_argument("--output", default="data/counterfactual/current_cf_tasks_v3.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/counterfactual_task_build_current_v3.json")
    parser.add_argument("--status", default="outputs/status/15_COUNTERFACTUAL_TASK_BUILD_CURRENT_DATA.status.json")
    parser.add_argument("--manifest", default="outputs/manifests/15_COUNTERFACTUAL_TASK_BUILD_CURRENT_DATA.manifest.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--min-per-type", type=int, default=10)
    args = parser.parse_args()

    import pandas as pd

    contexts = pd.read_parquet(args.contexts)
    tokens = pd.read_parquet(args.tokens) if Path(args.tokens).exists() else pd.DataFrame()
    df = contexts[contexts["split"] == "test"].copy()
    if not tokens.empty and "technical_event_tokens_json" not in df.columns:
        token_cols = [col for col in ["sample_id", "technical_event_tokens_json"] if col in tokens.columns]
        if token_cols:
            df = df.merge(
                tokens[token_cols].drop_duplicates("sample_id"), on="sample_id", how="left"
            )
    df = df.sort_values(["event_date", "sample_id"]).copy()

    task_specs = [
        ("remove_negative_news", "down_decrease"),
        ("remove_positive_news", "up_decrease"),
        ("neutralize_bearish_technical", "down_decrease"),
        ("neutralize_bullish_technical", "up_decrease"),
    ]
    per_type_limit = max(1, args.limit // len(task_specs)) if args.limit else 0
    records: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    applicable_counts: dict[str, int] = {}
    for cf_type, expected_direction in task_specs:
        subset = df[df.apply(lambda row: is_applicable(row, cf_type), axis=1)].head(per_type_limit if per_type_limit else None)
        applicable_counts[cf_type] = int(len(subset))
        type_counts[cf_type] = 0
        for _, row in subset.iterrows():
            records.append(build_task(row, cf_type, expected_direction))
            type_counts[cf_type] += 1

    out_df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    metrics = {
        "tasks_written": int(len(out_df)),
        "samples_used": int(out_df["sample_id"].nunique()) if len(out_df) else 0,
        "split": "test",
        "type_counts": type_counts,
        "applicable_counts": applicable_counts,
        "limit": args.limit,
        "min_per_type": args.min_per_type,
    }
    failures: list[str] = []
    if len(out_df) == 0:
        failures.append("no counterfactual tasks generated")
    for cf_type, count in type_counts.items():
        if count < args.min_per_type:
            failures.append(f"{cf_type} tasks {count} < {args.min_per_type}")
    if len(out_df) and set(out_df["split"].dropna()) != {"test"}:
        failures.append("counterfactual tasks contain non-test split")

    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.contexts, args.tokens],
        [args.output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
