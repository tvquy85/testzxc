from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.forecast_prediction import (
    FORECAST_KEYS,
    LABEL_FROM_KEY,
    expected_action_for_distribution,
)
from src.llm.parse_and_validate_rationale import canonical_forecast_key, parse_llm_json_strict
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "14_6_FORECAST_DISTRIBUTION_REPAIR_V6"
LABELS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]


def normalize_label(value: Any) -> str:
    label = str(value or "neutral").strip().lower().replace(" ", "_")
    return label if label in LABELS else "neutral"


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def raw_distribution_from_output(raw_output: Any) -> tuple[dict[str, float] | None, str]:
    parsed = parse_llm_json_strict(str(raw_output or ""))
    if not isinstance(parsed, dict):
        return None, "invalid_json"
    dist = parsed.get("forecast_distribution")
    if not isinstance(dist, dict):
        return None, "missing_forecast_distribution"
    canonical: dict[str, float] = {}
    extras: list[str] = []
    for key, value in dist.items():
        canonical_key = canonical_forecast_key(str(key))
        if canonical_key not in FORECAST_KEYS:
            extras.append(str(key))
            continue
        numeric = finite_float(value)
        if numeric is None:
            return None, f"non_finite_value:{canonical_key}"
        canonical[canonical_key] = numeric
    missing = [key for key in FORECAST_KEYS if key not in canonical]
    if missing:
        return None, f"missing_keys:{','.join(missing)}"
    if extras:
        return None, f"extra_keys:{','.join(sorted(extras))}"
    if any(value < 0.0 for value in canonical.values()):
        return None, "negative_probability"
    total = sum(canonical.values())
    if total <= 0.0:
        return None, "non_positive_probability_sum"
    return {key: float(canonical[key]) / total for key in FORECAST_KEYS}, "normalized_probability_sum"


def current_distribution(row: pd.Series) -> dict[str, float]:
    return {
        "strong_down": float(row.get("p_strong_down", 0.0)),
        "mild_down": float(row.get("p_mild_down", 0.0)),
        "neutral": float(row.get("p_neutral", 0.0)),
        "mild_up": float(row.get("p_mild_up", 0.0)),
        "strong_up": float(row.get("p_strong_up", 0.0)),
    }


def apply_distribution(row: dict[str, Any], dist: dict[str, float]) -> dict[str, Any]:
    for key in FORECAST_KEYS:
        row[f"p_{key}"] = float(dist[key])
    pred_key = max(dist, key=dist.get)
    row["pred_label"] = LABEL_FROM_KEY[pred_key]
    row["action"] = expected_action_for_distribution(dist)
    row["expected_action"] = row["action"]
    row["action_consistency_ok"] = True
    return row


def serialize_parse_errors(value: Any) -> str:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def repair_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repaired_rows = 0
    repairable_invalid_rows = 0
    unrepaired_invalid_rows = 0
    repair_reasons: dict[str, int] = {}
    original_schema_ok = predictions["schema_ok"].astype(bool) if "schema_ok" in predictions.columns else pd.Series(False, index=predictions.index)
    for _, series in predictions.iterrows():
        row = series.to_dict()
        row["forecast_repair_applied"] = False
        row["forecast_repair_reason"] = ""
        if bool(series.get("schema_ok", False)):
            row = apply_distribution(row, current_distribution(series))
            row["schema_ok"] = True
            rows.append(row)
            continue
        dist, reason = raw_distribution_from_output(series.get("raw_output", ""))
        repair_reasons[reason] = repair_reasons.get(reason, 0) + 1
        if dist is None:
            unrepaired_invalid_rows += 1
            row["schema_ok"] = False
            row["forecast_repair_reason"] = reason
            rows.append(row)
            continue
        repairable_invalid_rows += 1
        repaired_rows += 1
        row = apply_distribution(row, dist)
        row["parse_ok"] = True
        row["schema_ok"] = True
        row["parse_errors"] = []
        row["forecast_repair_applied"] = True
        row["forecast_repair_reason"] = reason
        rows.append(row)

    repaired = pd.DataFrame(rows)
    if "parse_errors" in repaired.columns:
        repaired["parse_errors"] = repaired["parse_errors"].map(serialize_parse_errors)
    metrics = {
        "rows": int(len(predictions)),
        "original_schema_ok_rows": int(original_schema_ok.sum()),
        "original_schema_ok_rate": float(original_schema_ok.mean()) if len(original_schema_ok) else 0.0,
        "repairable_invalid_rows": int(repairable_invalid_rows),
        "repaired_rows": int(repaired_rows),
        "unrepaired_invalid_rows": int(unrepaired_invalid_rows),
        "repaired_schema_ok_rows": int(repaired["schema_ok"].astype(bool).sum()) if "schema_ok" in repaired.columns else 0,
        "repaired_schema_ok_rate": float(repaired["schema_ok"].astype(bool).mean()) if len(repaired) and "schema_ok" in repaired.columns else 0.0,
        "repair_reason_counts": repair_reasons,
    }
    return repaired, metrics


def classification_metrics(predictions: pd.DataFrame, contexts: pd.DataFrame, label_col: str) -> dict[str, Any]:
    if predictions.empty or contexts.empty or label_col not in contexts.columns:
        return {}
    valid = predictions[predictions["schema_ok"].astype(bool)].copy() if "schema_ok" in predictions.columns else predictions.iloc[0:0].copy()
    merged = valid.merge(contexts[["sample_id", label_col]], on="sample_id", how="inner")
    if merged.empty:
        return {"valid_rows": 0}
    y_true = merged[label_col].map(normalize_label)
    y_pred = merged["pred_label"].map(normalize_label)
    return {
        "valid_rows": int(len(merged)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pred_label_distribution": y_pred.value_counts(dropna=False).to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--contexts", default="data/processed/current_v6_prediction_contexts.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--label-col", default="target_label_5")
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.99)
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.predictions).exists():
        failures.append(f"predictions missing: {args.predictions}")
        predictions = pd.DataFrame()
    else:
        predictions = pd.read_parquet(args.predictions)
    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    if contexts.empty:
        failures.append(f"contexts missing or empty: {args.contexts}")
    missing = sorted({"sample_id", "raw_output", "schema_ok", "pred_label", "action", *[f"p_{key}" for key in FORECAST_KEYS]} - set(predictions.columns))
    if missing:
        failures.append(f"predictions missing columns: {missing}")

    repaired = pd.DataFrame()
    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    if not failures:
        repaired, repair_metrics = repair_predictions(predictions)
        original_eval = classification_metrics(predictions, contexts, args.label_col)
        repaired_eval = classification_metrics(repaired, contexts, args.label_col)
        metrics.update(
            {
                "pipeline_pass": True,
                "claim_allowed": False,
                "diagnostic_only": True,
                **repair_metrics,
                "original_eval": original_eval,
                "repaired_eval": repaired_eval,
                "macro_f1_delta": float(repaired_eval.get("macro_f1", 0.0) - original_eval.get("macro_f1", 0.0)),
                "mcc_delta": float(repaired_eval.get("mcc", 0.0) - original_eval.get("mcc", 0.0)),
                "repair_boundary": (
                    "repair normalizes parse-ok forecast distributions with all five non-negative probabilities and positive total; "
                    "it does not use target labels and does not claim forecast superiority"
                ),
            }
        )
        if metrics["repaired_schema_ok_rate"] < args.min_schema_ok_rate:
            failures.append(
                f"repaired_schema_ok_rate {metrics['repaired_schema_ok_rate']:.4f} < {args.min_schema_ok_rate:.4f}"
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(args.output, index=False)
    metrics["pipeline_pass"] = not failures and not repaired.empty
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.predictions, args.contexts, args.output, args.metrics], STEP)
    status = "PASS" if metrics["pipeline_pass"] else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.predictions, args.contexts],
        [args.output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
