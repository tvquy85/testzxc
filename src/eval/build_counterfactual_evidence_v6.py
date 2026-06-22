from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.counterfactual_direction_rules_v6 import expected_direction
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "16A_BUILD_COUNTERFACTUAL_V6"

POSITIVE_TERMS = [
    "approval",
    "beat",
    "beats",
    "bullish",
    "buy",
    "gain",
    "gains",
    "good",
    "great",
    "growth",
    "higher",
    "outperform",
    "positive",
    "profit",
    "profits",
    "raised",
    "raises",
    "record",
    "strong",
    "surge",
    "upgrade",
    "win",
    "wins",
]
NEGATIVE_TERMS = [
    "bankruptcy",
    "bearish",
    "challenging",
    "cut",
    "decline",
    "dips",
    "downgrade",
    "drop",
    "fell",
    "investigation",
    "lawsuit",
    "loss",
    "losses",
    "lower",
    "miss",
    "missed",
    "negative",
    "plunge",
    "recall",
    "warning",
    "weak",
]
BULLISH_TOKEN_TERMS = ("BULLISH", "BUY", "HIGHER", "GOLDEN_CROSS", "OVERBOUGHT", "UP", "ABOVE")
BEARISH_TOKEN_TERMS = ("BEARISH", "SELL", "LOWER", "DEATH_CROSS", "OVERSOLD", "DOWN", "BELOW")
TASK_TYPES = [
    "remove_positive_evidence",
    "remove_negative_evidence",
    "neutralize_positive_evidence",
    "neutralize_negative_evidence",
    "remove_all_company_evidence",
    "neutralize_bearish_technical",
    "neutralize_bullish_technical",
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


def parse_json(value: Any, default: Any) -> Any:
    value = clean_value(value)
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed
    except Exception:
        return default


def parse_pack(value: Any) -> dict[str, Any]:
    parsed = parse_json(value, {})
    return parsed if isinstance(parsed, dict) else {}


def parse_tokens(row: pd.Series, pack: dict[str, Any]) -> list[Any]:
    tokens = pack.get("technical_signals")
    if isinstance(tokens, list):
        return tokens
    parsed = parse_json(row.get("technical_event_tokens_json"), [])
    return parsed if isinstance(parsed, list) else []


def evidence_item_text(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item or "")
    return " ".join(str(item.get(key, "") or "") for key in ("headline", "body_excerpt", "article_type", "quality_tier"))


def evidence_text(items: list[dict[str, Any]]) -> tuple[str, str]:
    headlines = [str(item.get("headline", "") or "") for item in items if isinstance(item, dict)]
    bodies = [str(item.get("body_excerpt", "") or "") for item in items if isinstance(item, dict)]
    headline = " | ".join(text for text in headlines if text)
    body = "\n".join(text for text in bodies if text)
    return headline, body


def has_terms(text: Any, terms: list[str]) -> bool:
    low = str(text or "").lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in terms)


def neutralize_terms(text: Any, terms: list[str]) -> str:
    out = str(text or "")
    for term in terms:
        out = re.sub(rf"\b{re.escape(term)}\b", "neutral update", out, flags=re.IGNORECASE)
    return out


def evidence_polarity_score(items: list[dict[str, Any]]) -> float:
    score = 0.0
    for item in items:
        text = evidence_item_text(item)
        if has_terms(text, POSITIVE_TERMS):
            score += 1.0
        if has_terms(text, NEGATIVE_TERMS):
            score -= 1.0
    return score


def token_text(token: Any) -> str:
    if isinstance(token, dict):
        return " ".join(str(token.get(key, "") or "") for key in ("token", "direction_prior", "rule"))
    return str(token or "")


def has_token_signal(tokens: list[Any], terms: tuple[str, ...]) -> bool:
    return any(any(term in token_text(token).upper() for term in terms) for token in tokens)


def neutralize_tokens(tokens: list[Any], terms: tuple[str, ...], neutral_token: str) -> list[Any]:
    kept = [token for token in tokens if not any(term in token_text(token).upper() for term in terms)]
    kept.append(
        {
            "token": neutral_token,
            "direction_prior": "neutralized",
            "strength": "counterfactual",
            "rule": "v6_counterfactual_evidence_neutralization",
        }
    )
    return kept


def transformed_company_evidence(company: list[dict[str, Any]], cf_type: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in company:
        text = evidence_item_text(item)
        is_positive = has_terms(text, POSITIVE_TERMS)
        is_negative = has_terms(text, NEGATIVE_TERMS)
        if cf_type == "remove_positive_evidence" and is_positive:
            continue
        if cf_type == "remove_negative_evidence" and is_negative:
            continue
        new_item = dict(item)
        if cf_type == "neutralize_positive_evidence" and is_positive:
            new_item["headline"] = neutralize_terms(new_item.get("headline"), POSITIVE_TERMS)
            new_item["body_excerpt"] = neutralize_terms(new_item.get("body_excerpt"), POSITIVE_TERMS)
        if cf_type == "neutralize_negative_evidence" and is_negative:
            new_item["headline"] = neutralize_terms(new_item.get("headline"), NEGATIVE_TERMS)
            new_item["body_excerpt"] = neutralize_terms(new_item.get("body_excerpt"), NEGATIVE_TERMS)
        out.append(new_item)
    return out


def build_task(row: pd.Series, cf_type: str) -> dict[str, Any] | None:
    pack = parse_pack(row.get("evidence_pack_json"))
    company = [item for item in pack.get("company_evidence", []) if isinstance(item, dict)]
    context = [item for item in pack.get("context_evidence", []) if isinstance(item, dict)]
    tokens = parse_tokens(row, pack)
    all_evidence = [*company, *context]
    original_headline, original_body = evidence_text(all_evidence)
    if not original_body:
        original_body = str(row.get("clean_context_text", "") or "")
    cf_company = list(company)
    cf_context = list(context)
    cf_tokens = list(tokens)
    polarity = evidence_polarity_score(company)
    expected = expected_direction(cf_type)

    if cf_type in {
        "remove_positive_evidence",
        "remove_negative_evidence",
        "neutralize_positive_evidence",
        "neutralize_negative_evidence",
    }:
        terms = POSITIVE_TERMS if "positive" in cf_type else NEGATIVE_TERMS
        if not any(has_terms(evidence_item_text(item), terms) for item in company):
            return None
        cf_company = transformed_company_evidence(company, cf_type)
    elif cf_type == "remove_all_company_evidence":
        if not company or polarity == 0:
            return None
        cf_company = []
        expected = expected_direction(cf_type, polarity)
    elif cf_type == "neutralize_bearish_technical":
        if not has_token_signal(tokens, BEARISH_TOKEN_TERMS):
            return None
        cf_tokens = neutralize_tokens(tokens, BEARISH_TOKEN_TERMS, "TECHNICAL_BEARISH_SIGNAL_NEUTRALIZED")
    elif cf_type == "neutralize_bullish_technical":
        if not has_token_signal(tokens, BULLISH_TOKEN_TERMS):
            return None
        cf_tokens = neutralize_tokens(tokens, BULLISH_TOKEN_TERMS, "TECHNICAL_BULLISH_SIGNAL_NEUTRALIZED")
    else:
        return None

    cf_headline, cf_body = evidence_text([*cf_company, *cf_context])
    if cf_type == "remove_all_company_evidence" and not cf_body:
        cf_body = "company-specific evidence removed; only non-company context remains"
    if not cf_headline:
        cf_headline = f"{cf_type} applied"
    if not cf_body:
        cf_body = f"{cf_type} applied"

    return {
        "sample_id": clean_value(row.get("sample_id")),
        "ticker": clean_value(row.get("ticker")),
        "event_date": str(clean_value(row.get("event_date"))),
        "split": clean_value(row.get("split", "test")),
        "v6_track": clean_value(row.get("v6_track")),
        "counterfactual_type": cf_type,
        "expected_direction": expected,
        "evidence_polarity_score": polarity,
        "original_headline": original_headline,
        "original_body": original_body,
        "original_technical_event_tokens_json": json.dumps(tokens, ensure_ascii=False),
        "counterfactual_headline": cf_headline,
        "counterfactual_body": cf_body,
        "counterfactual_technical_event_tokens_json": json.dumps(cf_tokens, ensure_ascii=False),
    }


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="outputs/metrics/16A_v6_counterfactual_build.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--limit", type=int, default=350)
    parser.add_argument("--min-per-type", type=int, default=10)
    args = parser.parse_args()

    failures: list[str] = []
    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    if contexts.empty:
        failures.append(f"contexts missing or empty: {args.contexts}")
    if args.predictions and Path(args.predictions).exists() and len(contexts):
        preds = pd.read_parquet(args.predictions)
        if "sample_id" in preds.columns:
            contexts = contexts[contexts["sample_id"].astype(str).isin(set(preds["sample_id"].astype(str)))].copy()
    elif args.predictions:
        failures.append(f"predictions missing: {args.predictions}")

    df = contexts.copy()
    if len(df) and "split" in df.columns:
        df = df[df["split"].eq("test")].copy()
    df = df.sort_values([col for col in ["event_date", "sample_id"] if col in df.columns]).copy() if len(df) else df
    per_type_limit = max(1, args.limit // len(TASK_TYPES)) if args.limit else 0

    rows_by_type: dict[str, list[dict[str, Any]]] = {}
    type_counts: dict[str, int] = {}
    applicable_counts: dict[str, int] = {}
    for cf_type in TASK_TYPES:
        built: list[dict[str, Any]] = []
        applicable = 0
        for _, row in df.iterrows():
            task = build_task(row, cf_type)
            if task is None:
                continue
            applicable += 1
            if not per_type_limit or len(built) < per_type_limit:
                built.append(task)
        rows_by_type[cf_type] = built
        type_counts[cf_type] = len(built)
        applicable_counts[cf_type] = applicable

    rows: list[dict[str, Any]] = []
    for idx in range(max(type_counts.values(), default=0)):
        for cf_type in TASK_TYPES:
            if idx < len(rows_by_type[cf_type]):
                rows.append(rows_by_type[cf_type][idx])

    write_jsonl(args.output, rows)
    metrics = {
        "tasks_written": int(len(rows)),
        "samples_used": int(len({str(row.get("sample_id")) for row in rows})),
        "context_rows_considered": int(len(df)),
        "split": "test",
        "task_types": TASK_TYPES,
        "type_counts": type_counts,
        "applicable_counts": applicable_counts,
        "limit": args.limit,
        "per_type_limit": per_type_limit,
        "min_per_type": args.min_per_type,
        "prediction_aligned": bool(args.predictions),
    }
    if not rows:
        failures.append("no counterfactual tasks generated")
    if rows and {str(row.get("split")) for row in rows} != {"test"}:
        failures.append("counterfactual tasks contain non-test split")
    for cf_type, count in type_counts.items():
        if count < args.min_per_type:
            failures.append(f"{cf_type} tasks {count} < {args.min_per_type}")

    write_json(args.metrics, metrics)
    manifest_inputs = [args.contexts, args.output, args.metrics]
    if args.predictions:
        manifest_inputs.append(args.predictions)
    write_manifest(args.manifest, manifest_inputs, STEP)
    status = "PASS" if not failures else "FAIL"
    inputs = [args.contexts] + ([args.predictions] if args.predictions else [])
    write_status(args.status, STEP, status, inputs, [args.output, args.metrics, args.manifest, args.status], metrics, failures, status == "PASS")
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
