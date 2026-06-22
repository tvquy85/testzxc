from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "17_BASELINES_SEP_POLICY_TECH_RULE"
LABELS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]

POSITIVE_TERMS = {
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
}
NEGATIVE_TERMS = {
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
}


def normalize_label(value: Any) -> str:
    label = str(value or "neutral").strip().lower().replace(" ", "_")
    return label if label in LABELS else "neutral"


def parse_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def token_strength(token: dict[str, Any]) -> float:
    strength = str(token.get("strength", "medium")).lower()
    return {"weak": 0.5, "low": 0.5, "medium": 1.0, "high": 1.5, "strong": 1.5}.get(strength, 1.0)


def technical_score(row: pd.Series) -> float:
    pack = parse_json(row.get("evidence_pack_json"), {})
    tokens = pack.get("technical_signals") if isinstance(pack, dict) else None
    if not isinstance(tokens, list):
        tokens = parse_json(row.get("technical_event_tokens_json"), [])
    score = 0.0
    for token in tokens if isinstance(tokens, list) else []:
        if not isinstance(token, dict):
            continue
        text = " ".join(str(token.get(key, "") or "") for key in ("token", "direction_prior", "rule")).lower()
        weight = token_strength(token)
        if "bullish" in text or "positive" in text or "outperformance" in text or "above" in text:
            score += weight
        if "bearish" in text or "negative" in text or "underperformance" in text or "below" in text:
            score -= weight
    return float(score)


def evidence_text(row: pd.Series) -> str:
    chunks = [str(row.get("clean_context_text", "") or "")]
    pack = parse_json(row.get("evidence_pack_json"), {})
    if isinstance(pack, dict):
        for section in ["company_evidence", "context_evidence"]:
            for item in pack.get(section, []) or []:
                if isinstance(item, dict):
                    chunks.append(str(item.get("headline", "") or ""))
                    chunks.append(str(item.get("body_excerpt", "") or ""))
    return " ".join(chunks)


def word_count(text: str, vocab: set[str]) -> int:
    low = str(text or "").lower()
    return sum(1 for word in vocab if re.search(rf"\b{re.escape(word)}\b", low))


def news_score(row: pd.Series) -> float:
    text = evidence_text(row)
    return float(word_count(text, POSITIVE_TERMS) - word_count(text, NEGATIVE_TERMS))


def label_from_score(score: float) -> str:
    score = float(score)
    if score <= -2.0:
        return "strong_down"
    if score < -0.25:
        return "mild_down"
    if score >= 2.0:
        return "strong_up"
    if score > 0.25:
        return "mild_up"
    return "neutral"


def metric_row(
    method: str,
    y_true: list[str],
    y_pred: list[str],
    notes: str,
    status: str = "PASS",
    reference_only: bool = False,
    schema_ok_rate: float | None = None,
    non_hold_rate: float | None = None,
) -> dict[str, Any]:
    rows = len(y_true)
    if rows:
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    else:
        accuracy = macro_f1 = mcc = bal_acc = np.nan
    return {
        "method": method,
        "status": status,
        "rows": int(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "balanced_accuracy": bal_acc,
        "schema_ok_rate": schema_ok_rate,
        "non_hold_rate": non_hold_rate,
        "reference_only": bool(reference_only),
        "comparable_current_data": bool(not reference_only and status == "PASS" and rows > 0),
        "notes": notes,
    }


def reference_row(method: str, notes: str) -> dict[str, Any]:
    return metric_row(method, [], [], notes=notes, status="REFERENCE_ONLY", reference_only=True)


def prediction_metric_row(method: str, predictions: pd.DataFrame, contexts: pd.DataFrame, label_col: str, notes: str) -> dict[str, Any]:
    if predictions.empty or "sample_id" not in predictions.columns:
        return reference_row(method, f"{notes}; prediction artifact missing")
    merged = predictions.merge(contexts[["sample_id", label_col]], on="sample_id", how="inner")
    if merged.empty:
        return reference_row(method, f"{notes}; no joined current-data rows")
    schema = merged["schema_ok"].astype(bool) if "schema_ok" in merged.columns else pd.Series(True, index=merged.index)
    valid = merged[schema].copy()
    y_true = valid[label_col].map(normalize_label).tolist()
    y_pred = valid.get("pred_label", pd.Series("neutral", index=valid.index)).map(normalize_label).tolist()
    actions = valid.get("action", pd.Series("hold", index=valid.index)).astype(str).str.lower()
    return metric_row(
        method,
        y_true,
        y_pred,
        notes=notes,
        schema_ok_rate=float(schema.mean()) if len(schema) else 0.0,
        non_hold_rate=float((actions != "hold").mean()) if len(actions) else 0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--dpo-predictions", required=True)
    parser.add_argument("--rwsft-predictions", required=True)
    parser.add_argument("--base-predictions", default=None)
    parser.add_argument("--output", default="outputs/tables/17_v6_comparable_baselines.csv")
    parser.add_argument("--metrics", default="outputs/metrics/17_v6_baseline_comparison.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--label-col", default="target_label_5")
    args = parser.parse_args()

    failures: list[str] = []
    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    dpo = pd.read_parquet(args.dpo_predictions) if Path(args.dpo_predictions).exists() else pd.DataFrame()
    rwsft = pd.read_parquet(args.rwsft_predictions) if Path(args.rwsft_predictions).exists() else pd.DataFrame()
    base = pd.read_parquet(args.base_predictions) if args.base_predictions and Path(args.base_predictions).exists() else pd.DataFrame()
    if contexts.empty:
        failures.append(f"contexts missing or empty: {args.contexts}")
    if dpo.empty:
        failures.append(f"dpo predictions missing or empty: {args.dpo_predictions}")
    if rwsft.empty:
        failures.append(f"rwsft predictions missing or empty: {args.rwsft_predictions}")
    if args.label_col not in contexts.columns:
        failures.append(f"contexts missing label column: {args.label_col}")

    rows: list[dict[str, Any]] = []
    if not failures:
        test_contexts = contexts[contexts["split"].eq("test")].copy() if "split" in contexts.columns else contexts.copy()
        dpo_ids = set(dpo["sample_id"].astype(str)) if "sample_id" in dpo.columns else set()
        rwsft_ids = set(rwsft["sample_id"].astype(str)) if "sample_id" in rwsft.columns else set()
        eval_contexts = test_contexts[
            test_contexts["sample_id"].astype(str).isin(dpo_ids & rwsft_ids)
        ].drop_duplicates("sample_id").copy()
        if eval_contexts.empty:
            failures.append("no shared current-data rows across contexts, DPO, and RWSFT predictions")
        else:
            y_true = eval_contexts[args.label_col].map(normalize_label).tolist()
            tech_scores = eval_contexts.apply(technical_score, axis=1)
            news_scores = eval_contexts.apply(news_score, axis=1)
            clipped_news = news_scores.clip(-2.0, 2.0)
            clipped_tech = tech_scores.clip(-2.0, 2.0)
            rows.extend(
                [
                    metric_row(
                        "Technical_Rule",
                        y_true,
                        [label_from_score(score) for score in tech_scores],
                        "deterministic technical-token sign/strength rule on the V6 prediction contexts",
                    ),
                    prediction_metric_row(
                        "Qwen_DPO_V6",
                        dpo,
                        eval_contexts,
                        args.label_col,
                        "official Step 14 DPO prediction artifact",
                    ),
                    prediction_metric_row(
                        "Qwen_RWSFT_V6",
                        rwsft,
                        eval_contexts,
                        args.label_col,
                        "official Step 14 RWSFT prediction artifact",
                    ),
                    prediction_metric_row(
                        "Qwen_Base_NoAlign",
                        base,
                        eval_contexts,
                        args.label_col,
                        "no current-data base no-align prediction artifact was produced in this run",
                    ),
                    metric_row(
                        "SEP_Style_Summarize_Explain",
                        y_true,
                        [label_from_score(score) for score in clipped_news],
                        "comparable current-data evidence-text polarity proxy for summarize-explain style reasoning; not an external SEP checkpoint",
                    ),
                    metric_row(
                        "Policy_Style_Scalar_Proxy",
                        y_true,
                        [label_from_score(0.55 * t + 0.45 * n) for t, n in zip(clipped_tech, clipped_news)],
                        "comparable deterministic scalar-policy proxy using technical and news evidence scores; Flow claim remains controlled by Step 11",
                    ),
                    metric_row(
                        "Neutral_Hold_Majority",
                        y_true,
                        ["neutral"] * len(y_true),
                        "always-neutral sanity baseline",
                    ),
                    reference_row(
                        "PEN_Reference_Only",
                        "no current-data PEN adapter or prediction artifact was implemented; reference only, excluded from outperform claims",
                    ),
                ]
            )

    table = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    required = {
        "Technical_Rule",
        "Qwen_Base_NoAlign",
        "Qwen_RWSFT_V6",
        "Qwen_DPO_V6",
        "SEP_Style_Summarize_Explain",
        "Policy_Style_Scalar_Proxy",
        "PEN_Reference_Only",
    }
    methods = set(table["method"].astype(str)) if len(table) and "method" in table.columns else set()
    missing = sorted(required - methods)
    if missing:
        failures.append(f"missing required baseline rows: {missing}")
    if len(table) < 6:
        failures.append(f"baseline rows {len(table)} < 6")

    comparable = table[~table["reference_only"].astype(bool)].copy() if len(table) else pd.DataFrame()
    reference_only_count = int(table["reference_only"].astype(bool).sum()) if len(table) else 0
    tech = table[table["method"].eq("Technical_Rule")].iloc[0].to_dict() if "Technical_Rule" in methods else {}
    dpo_row = table[table["method"].eq("Qwen_DPO_V6")].iloc[0].to_dict() if "Qwen_DPO_V6" in methods else {}
    dpo_beats_tech_macro = bool(pd.notna(dpo_row.get("macro_f1")) and pd.notna(tech.get("macro_f1")) and float(dpo_row["macro_f1"]) > float(tech["macro_f1"]))
    dpo_beats_tech_mcc = bool(pd.notna(dpo_row.get("mcc")) and pd.notna(tech.get("mcc")) and float(dpo_row["mcc"]) > float(tech["mcc"]))
    forecast_claim_allowed = bool(dpo_beats_tech_macro and dpo_beats_tech_mcc)
    metrics = {
        "baseline_count": int(len(table)),
        "comparable_baseline_count": int(len(comparable)),
        "reference_only_count": reference_only_count,
        "required_methods_present": not missing,
        "evaluation_rows": int(comparable["rows"].max()) if len(comparable) and "rows" in comparable else 0,
        "best_method_by_macro_f1": None,
        "best_macro_f1": None,
        "best_mcc": None,
        "technical_rule_macro_f1": float(tech["macro_f1"]) if tech else None,
        "technical_rule_mcc": float(tech["mcc"]) if tech else None,
        "qwen_dpo_macro_f1": float(dpo_row["macro_f1"]) if dpo_row else None,
        "qwen_dpo_mcc": float(dpo_row["mcc"]) if dpo_row else None,
        "dpo_beats_technical_rule_macro_f1": dpo_beats_tech_macro,
        "dpo_beats_technical_rule_mcc": dpo_beats_tech_mcc,
        "forecast_claim_allowed": forecast_claim_allowed,
        "claim_allowed": forecast_claim_allowed,
    }
    if len(comparable):
        ranked = comparable.assign(_macro_f1=pd.to_numeric(comparable["macro_f1"], errors="coerce")).sort_values("_macro_f1", ascending=False)
        metrics["best_method_by_macro_f1"] = str(ranked.iloc[0]["method"])
        metrics["best_macro_f1"] = float(ranked.iloc[0]["_macro_f1"])
        metrics["best_mcc"] = float(pd.to_numeric(comparable["mcc"], errors="coerce").max())
    metrics["pipeline_pass"] = not failures

    write_json(args.metrics, metrics)
    manifest_inputs = [args.contexts, args.dpo_predictions, args.rwsft_predictions, args.output, args.metrics]
    if args.base_predictions:
        manifest_inputs.append(args.base_predictions)
    write_manifest(args.manifest, manifest_inputs, STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.contexts, args.dpo_predictions, args.rwsft_predictions],
        [args.output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
