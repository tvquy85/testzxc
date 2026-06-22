from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baselines.run_v6_comparable_baselines import (
    LABELS,
    label_from_score,
    news_score,
    normalize_label,
    technical_score,
)
from src.eval.validation_calibrated_hybrid_v6 import paired_bootstrap_delta
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "17_6_VALIDATION_STACKED_FORECAST_PROBE_V6"


def metric_dict(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)) if y_true else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if y_true else 0.0,
    }


def prepare_frame(contexts: pd.DataFrame, predictions: pd.DataFrame, label_col: str) -> pd.DataFrame:
    ctx = contexts.copy().drop_duplicates("sample_id")
    pred = predictions.copy().drop_duplicates("sample_id")
    prob_cols = ["p_strong_down", "p_mild_down", "p_neutral", "p_mild_up", "p_strong_up"]
    cols = ["sample_id", "schema_ok", "pred_label", *prob_cols]
    missing = sorted(set(cols) - set(pred.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    out = ctx.merge(pred[cols], on="sample_id", how="left")
    out["schema_ok_bool"] = out["schema_ok"].fillna(False).astype(bool).astype(int)
    for col in prob_cols:
        out[col] = out[col].fillna(0.0).astype(float)
    up_side = out["p_mild_up"] + out["p_strong_up"]
    down_side = out["p_mild_down"] + out["p_strong_down"]
    out["up_side"] = up_side
    out["down_side"] = down_side
    out["confidence"] = np.maximum.reduce([up_side.to_numpy(), down_side.to_numpy(), out["p_neutral"].to_numpy()])
    out["margin_up_down"] = up_side - down_side
    out["technical_score"] = out.apply(technical_score, axis=1)
    out["news_score"] = out.apply(news_score, axis=1)
    out["technical_rule_pred"] = [label_from_score(score) for score in out["technical_score"]]
    out["dpo_pred"] = out["pred_label"].map(normalize_label).fillna("neutral")
    out["true_label"] = out[label_col].map(normalize_label)
    for col in [
        "num_company_event_evidence",
        "num_context_only_evidence",
        "mean_evidence_quality_score",
        "num_hard_event_evidence",
        "v6_training_weight",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) if col in out.columns else 0.0
    for col in ["has_company_event_news", "has_hard_event_news", "no_news_context_flag"]:
        out[col] = out[col].fillna(False).astype(bool).astype(int) if col in out.columns else 0
    return out


def add_indicator_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    test = test.copy()
    base_cols = [
        "p_strong_down",
        "p_mild_down",
        "p_neutral",
        "p_mild_up",
        "p_strong_up",
        "schema_ok_bool",
        "up_side",
        "down_side",
        "confidence",
        "margin_up_down",
        "technical_score",
        "news_score",
        "num_company_event_evidence",
        "num_context_only_evidence",
        "mean_evidence_quality_score",
        "num_hard_event_evidence",
        "v6_training_weight",
        "has_company_event_news",
        "has_hard_event_news",
        "no_news_context_flag",
    ]
    feature_cols = list(base_cols)
    for frame in [train, test]:
        for label in LABELS:
            frame[f"dpo_is_{label}"] = frame["dpo_pred"].eq(label).astype(int)
            frame[f"technical_is_{label}"] = frame["technical_rule_pred"].eq(label).astype(int)
    feature_cols.extend([f"dpo_is_{label}" for label in LABELS])
    feature_cols.extend([f"technical_is_{label}" for label in LABELS])
    tracks = sorted(set(train.get("v6_track", pd.Series(dtype=str)).astype(str)) | set(test.get("v6_track", pd.Series(dtype=str)).astype(str)))
    for frame in [train, test]:
        values = frame.get("v6_track", pd.Series("", index=frame.index)).astype(str)
        for track in tracks:
            col = f"track_cat_{track}"
            frame[col] = values.eq(track).astype(int)
    feature_cols.extend([f"track_cat_{track}" for track in tracks])
    for frame in [train, test]:
        frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train, test, feature_cols


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    c_value: float,
    class_weight: str | None,
) -> tuple[list[str], list[str]]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight=class_weight, max_iter=5000, solver="lbfgs"),
    )
    clf.fit(train[feature_cols], train["true_label"])
    return clf.predict(train[feature_cols]).tolist(), clf.predict(test[feature_cols]).tolist()


def row(split: str, method: str, y_true: list[str], y_pred: list[str], notes: str) -> dict[str, Any]:
    return {"split": split, "method": method, "rows": len(y_true), **metric_dict(y_true, y_pred), "notes": notes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-contexts", required=True)
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-contexts", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--label-col", default="target_label_5")
    parser.add_argument("--c-grid", default="0.003,0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--output", default="outputs/tables/17_6_v6_validation_stacked_forecast_probe.csv")
    parser.add_argument("--grid-output", default="outputs/tables/17_6_v6_validation_stacked_grid.csv")
    parser.add_argument("--predictions-output", default="outputs/predictions/current_v6_validation_stacked_forecast_predictions.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/17_6_v6_validation_stacked_forecast_probe.json")
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
    grid_table = pd.DataFrame()
    prediction_table = pd.DataFrame()
    if not failures:
        try:
            val = prepare_frame(pd.read_parquet(args.val_contexts), pd.read_parquet(args.val_predictions), args.label_col)
            test = prepare_frame(pd.read_parquet(args.test_contexts), pd.read_parquet(args.test_predictions), args.label_col)
            val, test, feature_cols = add_indicator_features(val, test)
            y_val = val["true_label"].tolist()
            y_test = test["true_label"].tolist()
            grid_rows: list[dict[str, Any]] = []
            predictions_by_key: dict[tuple[float, str | None], tuple[list[str], list[str]]] = {}
            for c_value in [float(item) for item in args.c_grid.split(",") if item.strip()]:
                for class_weight in [None, "balanced"]:
                    val_pred, test_pred = fit_predict(val, test, feature_cols, c_value=c_value, class_weight=class_weight)
                    predictions_by_key[(c_value, class_weight)] = (val_pred, test_pred)
                    grid_rows.append(
                        {
                            "c_value": c_value,
                            "class_weight": "none" if class_weight is None else class_weight,
                            **{f"val_{key}": value for key, value in metric_dict(y_val, val_pred).items()},
                            **{f"test_{key}": value for key, value in metric_dict(y_test, test_pred).items()},
                        }
                    )
            grid_table = pd.DataFrame(grid_rows)
            ranked = grid_table.sort_values(
                ["val_macro_f1", "val_mcc", "val_accuracy", "c_value"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)
            best = ranked.iloc[0]
            best_c = float(best["c_value"])
            best_weight = None if str(best["class_weight"]) == "none" else str(best["class_weight"])
            val_pred, test_pred = predictions_by_key[(best_c, best_weight)]
            result_rows = [
                row("val", "Technical_Rule", y_val, val["technical_rule_pred"].tolist(), "deterministic technical-token baseline"),
                row("val", "Qwen_DPO_V6", y_val, val["dpo_pred"].tolist(), "raw DPO labels; schema-invalid rows map to neutral"),
                row("val", "Stacked_Logistic_V6", y_val, val_pred, "regularized logistic stacker selected by validation Macro-F1/MCC"),
                row("test", "Technical_Rule", y_test, test["technical_rule_pred"].tolist(), "deterministic technical-token baseline"),
                row("test", "Qwen_DPO_V6", y_test, test["dpo_pred"].tolist(), "raw DPO labels; schema-invalid rows map to neutral"),
                row("test", "Stacked_Logistic_V6", y_test, test_pred, "regularized logistic stacker selected by validation Macro-F1/MCC"),
            ]
            result_table = pd.DataFrame(result_rows)
            prediction_table = test[["sample_id", "split", "ticker", "event_date", "true_label", "technical_rule_pred", "dpo_pred"]].copy()
            prediction_table["stacked_pred"] = test_pred
            tech_pred = test["technical_rule_pred"].tolist()
            for metric_name in ["accuracy", "macro_f1", "mcc"]:
                estimate, low, high = paired_bootstrap_delta(
                    y_test,
                    test_pred,
                    tech_pred,
                    metric_name,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + len(metric_name),
                )
                metrics[f"delta_{metric_name}_vs_technical"] = estimate
                metrics[f"delta_{metric_name}_vs_technical_ci95_low"] = low
                metrics[f"delta_{metric_name}_vs_technical_ci95_high"] = high
            val_stack = result_table[(result_table["split"].eq("val")) & (result_table["method"].eq("Stacked_Logistic_V6"))].iloc[0]
            test_stack = result_table[(result_table["split"].eq("test")) & (result_table["method"].eq("Stacked_Logistic_V6"))].iloc[0]
            test_tech = result_table[(result_table["split"].eq("test")) & (result_table["method"].eq("Technical_Rule"))].iloc[0]
            metrics.update(
                {
                    "pipeline_pass": True,
                    "claim_allowed": False,
                    "best_c": best_c,
                    "best_class_weight": "none" if best_weight is None else best_weight,
                    "feature_count": int(len(feature_cols)),
                    "val_rows": int(len(val)),
                    "test_rows": int(len(test)),
                    "val_stacked_macro_f1": float(val_stack["macro_f1"]),
                    "val_stacked_mcc": float(val_stack["mcc"]),
                    "test_stacked_macro_f1": float(test_stack["macro_f1"]),
                    "test_stacked_mcc": float(test_stack["mcc"]),
                    "test_technical_macro_f1": float(test_tech["macro_f1"]),
                    "test_technical_mcc": float(test_tech["mcc"]),
                    "paired_ci_support": bool(
                        metrics["delta_macro_f1_vs_technical_ci95_low"] > 0
                        and metrics["delta_mcc_vs_technical_ci95_low"] > 0
                    ),
                    "validation_overfit_warning": bool(float(val_stack["macro_f1"]) > float(test_stack["macro_f1"]) + 0.10),
                    "bootstrap_repetitions": int(args.n_bootstrap),
                    "seed": int(args.seed),
                    "claim_boundary": "diagnostic only; validation-selected stacker must beat Technical_Rule on test with CI support before any claim",
                }
            )
        except Exception as exc:
            failures.append(f"stacked forecast probe failed: {type(exc).__name__}: {str(exc)[:500]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(args.output, index=False)
    Path(args.grid_output).parent.mkdir(parents=True, exist_ok=True)
    grid_table.to_csv(args.grid_output, index=False)
    Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_parquet(args.predictions_output, index=False)
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [*required_inputs, args.output, args.grid_output, args.predictions_output, args.metrics],
        STEP,
        extra={
            "references": [
                "Wolpert 1992 stacked generalization motivation",
                "Scikit-learn LogisticRegression regularized multiclass model",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        required_inputs,
        [args.output, args.grid_output, args.predictions_output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
