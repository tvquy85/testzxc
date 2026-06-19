from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    return value


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
        return str(token.get("token", ""))
    return str(token)


def neutralize_overbought_oversold(tokens: list[Any]) -> list[Any]:
    targets = ("BOLLINGER_UPPER", "BOLLINGER_LOWER", "RSI_OVERBOUGHT", "RSI_OVERSOLD")
    return [token for token in tokens if not any(target in token_name(token).upper() for target in targets)]


def high_vol_to_normal_vol(tokens: list[Any]) -> list[Any]:
    converted: list[Any] = []
    changed = False
    for token in tokens:
        name = token_name(token).upper()
        if "HIGH_VOL" in name or "VOLATILITY_SPIKE" in name:
            changed = True
            if isinstance(token, dict):
                new_token = dict(token)
                new_token["token"] = "NORMAL_VOL_REGIME"
                new_token["direction_prior"] = "neutral"
                new_token["strength"] = "weak"
                new_token["rule"] = "counterfactual_high_vol_to_normal"
                converted.append(new_token)
            else:
                converted.append("NORMAL_VOL_REGIME")
        else:
            converted.append(token)
    if not changed:
        converted.append(
            {
                "token": "NORMAL_VOL_REGIME",
                "direction_prior": "neutral",
                "strength": "weak",
                "rule": "counterfactual_high_vol_to_normal",
            }
        )
    return converted


def build_task(row: Any, cf_type: str, expected_direction: str) -> dict[str, Any]:
    tokens = parse_tokens(row.get("technical_event_tokens_json"))
    headline = clean_value(row.get("headline")) or ""
    body = clean_value(row.get("body")) or ""
    cf_headline = str(headline)
    cf_body = str(body)
    cf_tokens = tokens
    if cf_type == "remove_bad_news":
        cf_headline = neutralize_terms(cf_headline, NEGATIVE_TERMS)
        cf_body = neutralize_terms(cf_body, NEGATIVE_TERMS)
    elif cf_type == "remove_good_news":
        cf_headline = neutralize_terms(cf_headline, POSITIVE_TERMS)
        cf_body = neutralize_terms(cf_body, POSITIVE_TERMS)
    elif cf_type == "neutralize_overbought_oversold":
        cf_tokens = neutralize_overbought_oversold(tokens)
    elif cf_type == "high_vol_to_normal_vol":
        cf_tokens = high_vol_to_normal_vol(tokens)
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
    parser.add_argument("--samples", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--tokens", default="data/indicators/technical_event_tokens_h1_v2.parquet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", default="data/processed/counterfactual_tasks_v2.jsonl")
    args = parser.parse_args()

    import pandas as pd

    samples = pd.read_parquet(args.samples)
    tokens = pd.read_parquet(args.tokens) if Path(args.tokens).exists() else pd.DataFrame()
    df = samples[samples["split"] == args.split].copy()
    if not tokens.empty:
        token_cols = [col for col in ["sample_id", "technical_event_tokens_json"] if col in tokens.columns]
        df = df.merge(tokens[token_cols], on="sample_id", how="left")
    df = df.sort_values(["event_date", "sample_id"]).head(args.limit).copy()
    task_specs = [
        ("remove_bad_news", "less_negative"),
        ("remove_good_news", "less_positive"),
        ("neutralize_overbought_oversold", "toward_neutral"),
        ("high_vol_to_normal_vol", "lower_extreme"),
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            for cf_type, expected_direction in task_specs:
                f.write(json.dumps(build_task(row, cf_type, expected_direction), ensure_ascii=False) + "\n")
                written += 1
    print(json.dumps({"tasks_written": written, "samples_used": int(len(df)), "split": args.split}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
