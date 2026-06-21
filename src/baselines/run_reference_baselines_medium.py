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

STEP = "18_MINIMUM_BASELINES_PEN_SEP_POLICY"
LABELS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]

POSITIVE = {
    "beat",
    "beats",
    "raised",
    "growth",
    "profit",
    "upgrade",
    "strong",
    "surge",
    "outperform",
    "higher",
    "buyback",
}
NEGATIVE = {
    "miss",
    "missed",
    "lawsuit",
    "decline",
    "loss",
    "warning",
    "downgrade",
    "weak",
    "lower",
    "fell",
    "recall",
}
BULLISH_TECH = ("BULLISH", "UP", "ABOVE", "BUY", "GOLDEN_CROSS", "OVERSOLD")
BEARISH_TECH = ("BEARISH", "DOWN", "BELOW", "SELL", "DEATH_CROSS", "OVERBOUGHT")


def parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(clean_string(value))
    except Exception:
        return None


def word_count(text: str, vocab: set[str]) -> int:
    low = text.lower()
    return sum(1 for word in vocab if re.search(rf"\b{re.escape(word)}\b", low))


def score_to_label(score: float) -> str:
    if score <= -2:
        return "strong_down"
    if score < 0:
        return "mild_down"
    if score >= 2:
        return "strong_up"
    if score > 0:
        return "mild_up"
    return "neutral"


def tech_score(row: pd.Series) -> float:
    pack = parse_json(row.get("evidence_pack_json")) or {}
    signals = pack.get("technical_signals") if isinstance(pack, dict) else None
    if not signals:
        signals = parse_json(row.get("technical_event_tokens_json")) or []
    text = json.dumps(signals, ensure_ascii=False).upper()
    bullish = sum(term in text for term in BULLISH_TECH)
    bearish = sum(term in text for term in BEARISH_TECH)
    return float(bullish - bearish)


def news_score(row: pd.Series) -> float:
    chunks = [clean_string(row.get("clean_context_text"))]
    pack = parse_json(row.get("evidence_pack_json")) or {}
    if isinstance(pack, dict):
        for section in ["company_evidence", "context_evidence"]:
            for item in pack.get(section, []) or []:
                if isinstance(item, dict):
                    chunks.append(clean_string(item.get("headline")))
                    chunks.append(clean_string(item.get("body_excerpt")))
    text = " ".join(chunks)
    return float(word_count(text, POSITIVE) - word_count(text, NEGATIVE))


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    scores: list[float] = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)) if y_true else 0.0


def metric_row(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    rows: int,
    notes: str = "",
    reference_only: bool = False,
) -> dict[str, Any]:
    return {
        "method": name,
        "baseline": name,
        "status": "PASS" if rows else "MISSING",
        "rows": rows,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "schema_ok_rate": None,
        "non_hold_rate": None,
        "reference_only": bool(reference_only),
        "notes": notes,
    }


def reference_row(name: str, notes: str) -> dict[str, Any]:
    return {
        "method": name,
        "baseline": name,
        "status": "PASS",
        "rows": 0,
        "accuracy": None,
        "macro_f1": None,
        "schema_ok_rate": None,
        "non_hold_rate": None,
        "reference_only": True,
        "notes": notes,
    }


def qwen_prediction_row(
    name: str,
    predictions: pd.DataFrame,
    test_contexts: pd.DataFrame,
    label_col: str,
    notes: str,
) -> dict[str, Any] | None:
    if predictions.empty or "sample_id" not in predictions.columns:
        return None
    merged = predictions.merge(test_contexts, on="sample_id", suffixes=("_pred", ""), how="inner")
    if "schema_ok" in merged.columns:
        merged = merged[merged["schema_ok"].astype(bool)].copy()
    if merged.empty:
        return None
    y_true = merged[label_col].astype(str).str.lower().str.replace(" ", "_").tolist()
    y_pred = merged.get("pred_label", pd.Series("neutral", index=merged.index)).astype(str).str.lower().str.replace(" ", "_").tolist()
    row = metric_row(name, y_true, y_pred, len(merged), notes)
    action = merged.get("action", pd.Series("hold", index=merged.index)).astype(str)
    schema = merged["schema_ok"].astype(bool) if "schema_ok" in merged.columns else pd.Series(True, index=merged.index)
    row["schema_ok_rate"] = float(schema.mean())
    row["non_hold_rate"] = float((action[schema] != "hold").mean()) if int(schema.sum()) else 0.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--rwsft-predictions", default="outputs/predictions/medium_clean_v4_rwsft_test_predictions.parquet")
    parser.add_argument("--output", default="outputs/tables/medium_baseline_comparison.csv")
    parser.add_argument("--metrics", default="outputs/metrics/18_baseline_comparison_medium.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--label-col", default="target_label_5")
    args = parser.parse_args()

    failures: list[str] = []
    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    preds = pd.read_parquet(args.predictions) if Path(args.predictions).exists() else pd.DataFrame()
    rwsft_preds = pd.read_parquet(args.rwsft_predictions) if args.rwsft_predictions and Path(args.rwsft_predictions).exists() else pd.DataFrame()
    if contexts.empty:
        failures.append(f"contexts missing or empty: {args.contexts}")
    if preds.empty:
        failures.append(f"predictions missing or empty: {args.predictions}")
    if args.label_col not in contexts.columns:
        failures.append(f"contexts missing label column: {args.label_col}")

    rows: list[dict[str, Any]] = []
    if not failures:
        test_contexts = contexts[contexts["split"].eq("test")].copy()
        dpo_merged = preds.merge(test_contexts, on="sample_id", suffixes=("_pred", ""), how="inner")
        if "schema_ok" in dpo_merged.columns:
            dpo_merged = dpo_merged[dpo_merged["schema_ok"].astype(bool)].copy()
        if dpo_merged.empty:
            failures.append("no schema-valid prediction rows joined to test contexts")
        else:
            y_true = dpo_merged[args.label_col].astype(str).str.lower().str.replace(" ", "_").tolist()
            tech_pred = [score_to_label(tech_score(row)) for _, row in dpo_merged.iterrows()]
            news_pred = [score_to_label(news_score(row)) for _, row in dpo_merged.iterrows()]
            combo_pred = [score_to_label(tech_score(row) + news_score(row)) for _, row in dpo_merged.iterrows()]
            neutral_pred = ["neutral"] * len(dpo_merged)
            rows.extend(
                [
                    metric_row("Technical_Rule", y_true, tech_pred, len(dpo_merged), "deterministic technical-token sign rule"),
                    metric_row("Text_News_Heuristic", y_true, news_pred, len(dpo_merged), "deterministic finance keyword text baseline"),
                    metric_row("News_Technical_Heuristic", y_true, combo_pred, len(dpo_merged), "combined text and technical sign rule"),
                    metric_row("Neutral_Hold_Majority", y_true, neutral_pred, len(dpo_merged), "always neutral sanity baseline"),
                ]
            )
            dpo_row = qwen_prediction_row("Qwen_DPO_Medium", preds, test_contexts, args.label_col, "official medium DPO prediction artifact")
            if dpo_row:
                rows.append(dpo_row)
            rwsft_row = qwen_prediction_row("Qwen_RWSFT_Medium", rwsft_preds, test_contexts, args.label_col, "same medium adapter family before DPO preference optimization")
            if rwsft_row:
                rows.append(rwsft_row)
            else:
                rows.append(reference_row("Qwen_RWSFT_Medium", "RWSFT prediction artifact missing; row is reference-only and not evidence"))
            rows.extend(
                [
                    reference_row("PEN_Reference_Only", "PEN code exists in repo, but no comparable current-data medium run was executed; do not claim outperforming PEN"),
                    reference_row("SEP_Reference_Only", "sep code exists in repo, but no comparable current-data medium run was executed; do not claim outperforming SEP"),
                    reference_row("Policy_Reference_Only", "policy code exists in repo, but no comparable current-data medium run was executed; do not claim outperforming Policy"),
                ]
            )

    table = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    if len(table) < 4:
        failures.append(f"baseline rows {len(table)} < 4")
    if len(table) and "NOT_RUN" in set(table.get("status", [])):
        failures.append("baseline table contains NOT_RUN")
    metrics = {
        "baseline_count": int(len(table)),
        "rows_per_baseline": int(table["rows"].max()) if len(table) and "rows" in table else 0,
        "best_macro_f1": float(pd.to_numeric(table["macro_f1"], errors="coerce").max()) if len(table) and "macro_f1" in table else 0.0,
        "best_accuracy": float(pd.to_numeric(table["accuracy"], errors="coerce").max()) if len(table) and "accuracy" in table else 0.0,
        "best_method_by_macro_f1": str(
            table.assign(_macro_f1=pd.to_numeric(table["macro_f1"], errors="coerce"))
            .sort_values("_macro_f1", ascending=False)
            .iloc[0]["method"]
        ) if len(table) and "macro_f1" in table else None,
        "baselines": table["baseline"].tolist() if len(table) else [],
        "reference_only_count": int(table["reference_only"].astype(bool).sum()) if len(table) and "reference_only" in table else 0,
        "comparable_baseline_count": int((~table["reference_only"].astype(bool)).sum()) if len(table) and "reference_only" in table else int(len(table)),
        "claim_allowed": False,
        "pipeline_pass": not failures,
    }
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.contexts, args.predictions, args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.contexts, args.predictions],
        [args.output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
