from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baselines.run_v6_comparable_baselines import (
    LABELS,
    label_from_score,
    normalize_label,
    technical_score,
)
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "17_5_VALIDATION_CALIBRATED_FORECAST_HYBRID_V6"


def side_confidence(df: pd.DataFrame) -> pd.Series:
    up = df["p_mild_up"].astype(float) + df["p_strong_up"].astype(float)
    down = df["p_mild_down"].astype(float) + df["p_strong_down"].astype(float)
    neutral = df["p_neutral"].astype(float)
    return pd.Series(np.maximum.reduce([up.to_numpy(), down.to_numpy(), neutral.to_numpy()]), index=df.index)


def metric_dict(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)) if y_true else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if y_true else 0.0,
    }


def percentile_interval(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.quantile(arr, alpha / 2.0)), float(np.quantile(arr, 1.0 - alpha / 2.0))


def paired_bootstrap_delta(
    y_true: list[str],
    candidate: list[str],
    benchmark: list[str],
    metric: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=object)
    cand = np.asarray(candidate, dtype=object)
    bench = np.asarray(benchmark, dtype=object)
    n = len(y)
    if metric == "macro_f1":
        fn = lambda yy, pp: float(f1_score(yy.tolist(), pp.tolist(), labels=LABELS, average="macro", zero_division=0))
    elif metric == "mcc":
        fn = lambda yy, pp: float(matthews_corrcoef(yy.tolist(), pp.tolist()))
    elif metric == "accuracy":
        fn = lambda yy, pp: float(accuracy_score(yy.tolist(), pp.tolist()))
    else:
        raise ValueError(f"unknown metric: {metric}")
    estimate = fn(y, cand) - fn(y, bench)
    boot: list[float] = []
    for _ in range(max(1, n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boot.append(fn(y[idx], cand[idx]) - fn(y[idx], bench[idx]))
    low, high = percentile_interval(boot)
    return float(estimate), low, high


def prepare_frame(contexts: pd.DataFrame, predictions: pd.DataFrame, label_col: str) -> pd.DataFrame:
    ctx = contexts.copy().drop_duplicates("sample_id")
    pred = predictions.copy().drop_duplicates("sample_id")
    pred_cols = [
        "sample_id",
        "schema_ok",
        "pred_label",
        "action",
        "p_strong_down",
        "p_mild_down",
        "p_neutral",
        "p_mild_up",
        "p_strong_up",
    ]
    missing_pred = sorted(set(pred_cols) - set(pred.columns))
    if missing_pred:
        raise ValueError(f"predictions missing columns: {missing_pred}")
    missing_ctx = sorted({"sample_id", label_col} - set(ctx.columns))
    if missing_ctx:
        raise ValueError(f"contexts missing columns: {missing_ctx}")
    out = ctx.merge(pred[pred_cols], on="sample_id", how="left")
    out["schema_ok_bool"] = out["schema_ok"].fillna(False).astype(bool)
    for col in ["p_strong_down", "p_mild_down", "p_neutral", "p_mild_up", "p_strong_up"]:
        out[col] = out[col].fillna(0.0).astype(float)
    out["technical_rule_pred"] = [label_from_score(score) for score in out.apply(technical_score, axis=1)]
    out["dpo_pred"] = out["pred_label"].map(normalize_label).fillna("neutral")
    out["true_label"] = out[label_col].map(normalize_label)
    out["dpo_confidence"] = side_confidence(out)
    return out


def apply_gate(frame: pd.DataFrame, threshold: float) -> pd.Series:
    use_dpo = frame["schema_ok_bool"].astype(bool) & (frame["dpo_confidence"].astype(float) >= threshold)
    return pd.Series(np.where(use_dpo, frame["dpo_pred"], frame["technical_rule_pred"]), index=frame.index)


def tune_threshold(frame: pd.DataFrame, grid: list[float]) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    y_true = frame["true_label"].tolist()
    for threshold in grid:
        pred = apply_gate(frame, threshold).tolist()
        metrics = metric_dict(y_true, pred)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
                "dpo_use_rate": float((frame["schema_ok_bool"] & (frame["dpo_confidence"] >= threshold)).mean()),
            }
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(["macro_f1", "mcc", "accuracy", "threshold"], ascending=[False, False, False, True]).reset_index(drop=True)
    return float(table.iloc[0]["threshold"]), table


def method_row(split: str, method: str, frame: pd.DataFrame, pred_col: str, notes: str) -> dict[str, Any]:
    y_true = frame["true_label"].tolist()
    y_pred = frame[pred_col].map(normalize_label).tolist()
    metrics = metric_dict(y_true, y_pred)
    return {
        "split": split,
        "method": method,
        "rows": int(len(frame)),
        **metrics,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-contexts", required=True)
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-contexts", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--label-col", default="target_label_5")
    parser.add_argument("--threshold-grid", default="0.00:1.00:0.01")
    parser.add_argument("--output", default="outputs/tables/17_5_v6_validation_calibrated_hybrid.csv")
    parser.add_argument("--threshold-table", default="outputs/tables/17_5_v6_validation_threshold_search.csv")
    parser.add_argument("--predictions-output", default="outputs/predictions/current_v6_validation_calibrated_hybrid_predictions.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/17_5_v6_validation_calibrated_hybrid.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    failures: list[str] = []
    required_inputs = [args.val_contexts, args.val_predictions, args.test_contexts, args.test_predictions]
    for path in required_inputs:
        if not Path(path).exists():
            failures.append(f"missing input: {path}")

    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    result_table = pd.DataFrame()
    threshold_table = pd.DataFrame()
    pred_out = pd.DataFrame()
    if not failures:
        try:
            val_frame = prepare_frame(pd.read_parquet(args.val_contexts), pd.read_parquet(args.val_predictions), args.label_col)
            test_frame = prepare_frame(pd.read_parquet(args.test_contexts), pd.read_parquet(args.test_predictions), args.label_col)
            start, stop, step = [float(x) for x in args.threshold_grid.split(":")]
            grid = [round(float(x), 10) for x in np.arange(start, stop + step / 2.0, step)]
            best_threshold, threshold_table = tune_threshold(val_frame, grid)
            val_frame["hybrid_pred"] = apply_gate(val_frame, best_threshold)
            test_frame["hybrid_pred"] = apply_gate(test_frame, best_threshold)
            val_frame["dpo_use_for_hybrid"] = val_frame["schema_ok_bool"] & (val_frame["dpo_confidence"] >= best_threshold)
            test_frame["dpo_use_for_hybrid"] = test_frame["schema_ok_bool"] & (test_frame["dpo_confidence"] >= best_threshold)

            result_rows = [
                method_row("val", "Technical_Rule", val_frame, "technical_rule_pred", "deterministic technical-token baseline"),
                method_row("val", "Qwen_DPO_V6", val_frame[val_frame["schema_ok_bool"]].copy(), "dpo_pred", "schema-valid DPO validation predictions"),
                method_row("val", "Qwen_DPO_Technical_Gated_V6", val_frame, "hybrid_pred", "validation-selected confidence gate: DPO if confident, else Technical_Rule"),
                method_row("test", "Technical_Rule", test_frame, "technical_rule_pred", "deterministic technical-token baseline"),
                method_row("test", "Qwen_DPO_V6", test_frame[test_frame["schema_ok_bool"]].copy(), "dpo_pred", "schema-valid DPO test predictions"),
                method_row("test", "Qwen_DPO_Technical_Gated_V6", test_frame, "hybrid_pred", "validation-selected confidence gate: DPO if confident, else Technical_Rule"),
            ]
            result_table = pd.DataFrame(result_rows)
            pred_cols = [
                "sample_id",
                "split",
                "ticker",
                "event_date",
                "true_label",
                "technical_rule_pred",
                "dpo_pred",
                "schema_ok_bool",
                "dpo_confidence",
                "dpo_use_for_hybrid",
                "hybrid_pred",
            ]
            pred_out = test_frame[[col for col in pred_cols if col in test_frame.columns]].copy()

            test_lookup = {row["method"]: row for _, row in result_table[result_table["split"].eq("test")].iterrows()}
            tech = test_lookup["Technical_Rule"]
            hybrid = test_lookup["Qwen_DPO_Technical_Gated_V6"]
            val_lookup = {row["method"]: row for _, row in result_table[result_table["split"].eq("val")].iterrows()}
            val_tech = val_lookup["Technical_Rule"]
            val_hybrid = val_lookup["Qwen_DPO_Technical_Gated_V6"]
            beats_technical = bool(
                float(hybrid["macro_f1"]) > float(tech["macro_f1"])
                and float(hybrid["mcc"]) > float(tech["mcc"])
            )
            validation_beats_technical = bool(
                float(val_hybrid["macro_f1"]) > float(val_tech["macro_f1"])
                and float(val_hybrid["mcc"]) > float(val_tech["mcc"])
            )
            metrics = {
                "pipeline_pass": True,
                "claim_allowed": False,
                "diagnostic_positive": beats_technical,
                "validation_positive": validation_beats_technical,
                "best_threshold": best_threshold,
                "threshold_selected_on": "val",
                "threshold_grid": args.threshold_grid,
                "val_rows": int(len(val_frame)),
                "test_rows": int(len(test_frame)),
                "val_dpo_schema_ok_rate": float(val_frame["schema_ok_bool"].mean()) if len(val_frame) else 0.0,
                "test_dpo_schema_ok_rate": float(test_frame["schema_ok_bool"].mean()) if len(test_frame) else 0.0,
                "val_dpo_use_rate": float(val_frame["dpo_use_for_hybrid"].mean()) if len(val_frame) else 0.0,
                "test_dpo_use_rate": float(test_frame["dpo_use_for_hybrid"].mean()) if len(test_frame) else 0.0,
                "test_hybrid_macro_f1": float(hybrid["macro_f1"]),
                "test_hybrid_mcc": float(hybrid["mcc"]),
                "test_hybrid_accuracy": float(hybrid["accuracy"]),
                "test_technical_macro_f1": float(tech["macro_f1"]),
                "test_technical_mcc": float(tech["mcc"]),
                "test_technical_accuracy": float(tech["accuracy"]),
                "delta_macro_f1_vs_technical": float(hybrid["macro_f1"]) - float(tech["macro_f1"]),
                "delta_mcc_vs_technical": float(hybrid["mcc"]) - float(tech["mcc"]),
                "claim_boundary": "diagnostic only until paired CI/statistical tests support the hybrid improvement",
            }
            y_test = test_frame["true_label"].tolist()
            hybrid_pred = test_frame["hybrid_pred"].map(normalize_label).tolist()
            tech_pred = test_frame["technical_rule_pred"].map(normalize_label).tolist()
            for metric_name in ["accuracy", "macro_f1", "mcc"]:
                estimate, low, high = paired_bootstrap_delta(
                    y_test,
                    hybrid_pred,
                    tech_pred,
                    metric_name,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + len(metric_name),
                )
                metrics[f"delta_{metric_name}_vs_technical_bootstrap"] = estimate
                metrics[f"delta_{metric_name}_vs_technical_ci95_low"] = low
                metrics[f"delta_{metric_name}_vs_technical_ci95_high"] = high
            metrics["paired_ci_support"] = bool(
                metrics["delta_macro_f1_vs_technical_ci95_low"] > 0
                and metrics["delta_mcc_vs_technical_ci95_low"] > 0
            )
            metrics["bootstrap_repetitions"] = int(args.n_bootstrap)
            metrics["seed"] = int(args.seed)
        except Exception as exc:
            failures.append(f"hybrid calibration failed: {type(exc).__name__}: {str(exc)[:500]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(args.output, index=False)
    Path(args.threshold_table).parent.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(args.threshold_table, index=False)
    Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    pred_out.to_parquet(args.predictions_output, index=False)
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [*required_inputs, args.output, args.threshold_table, args.predictions_output, args.metrics],
        STEP,
        extra={
            "references": [
                "Guo et al. 2017 validation-set post-hoc calibration",
                "Scikit-learn decision-threshold tuning on held-out validation data",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        required_inputs,
        [args.output, args.threshold_table, args.predictions_output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
