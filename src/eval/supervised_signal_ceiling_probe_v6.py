from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baselines.run_v6_comparable_baselines import (
    LABELS,
    evidence_text,
    label_from_score,
    news_score,
    normalize_label,
    technical_score,
)
from src.eval.validation_calibrated_hybrid_v6 import paired_bootstrap_delta
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "17_7_SUPERVISED_SIGNAL_CEILING_PROBE_V6"


def metric_dict(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "mcc": 0.0, "balanced_accuracy": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    return out


def key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["ticker"].astype(str), frame["event_date"].astype(str)))


def select_train_frame(train_contexts: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_contexts = parse_dates(train_contexts)
    val = parse_dates(val)
    test = parse_dates(test)
    cutoff = min(val["event_date"].min(), test["event_date"].min())
    heldout_ids = set(val["sample_id"].astype(str)) | set(test["sample_id"].astype(str))
    heldout_keys = key_set(pd.concat([val, test], ignore_index=True))
    train = train_contexts[
        train_contexts["split"].astype(str).eq("train")
        & train_contexts["event_date"].notna()
        & (train_contexts["event_date"] < cutoff)
        & ~train_contexts["sample_id"].astype(str).isin(heldout_ids)
    ].copy()
    train = train[~train.apply(lambda row: (str(row["ticker"]), str(row["event_date"])) in heldout_keys, axis=1)]
    return train.drop_duplicates("sample_id").reset_index(drop=True)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    out = pd.DataFrame(index=frame.index)
    clean_text = frame.get("clean_context_text", pd.Series("", index=frame.index)).fillna("").astype(str)
    out["text"] = frame.apply(evidence_text, axis=1).fillna("").astype(str) + " " + clean_text
    out["technical_tokens_text"] = frame.get("technical_event_tokens_json", pd.Series("", index=frame.index)).fillna("").astype(str)
    out["technical_score"] = frame.apply(technical_score, axis=1)
    out["news_score"] = frame.apply(news_score, axis=1)
    for col in [
        "num_company_event_evidence",
        "num_context_only_evidence",
        "mean_evidence_quality_score",
        "num_hard_event_evidence",
        "v6_training_weight",
    ]:
        out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0) if col in frame.columns else 0.0
    for col in ["has_company_event_news", "has_hard_event_news", "no_news_context_flag"]:
        out[col] = frame[col].fillna(False).astype(bool).astype(int) if col in frame.columns else 0
    out["ticker"] = frame.get("ticker", pd.Series("", index=frame.index)).fillna("").astype(str)
    out["v6_track"] = frame.get("v6_track", pd.Series("", index=frame.index)).fillna("").astype(str)
    return out


def make_model(c_value: float, min_df: int, max_text_features: int, max_token_features: int) -> Pipeline:
    numeric_cols = [
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
    features = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(max_features=max_text_features, ngram_range=(1, 2), min_df=min_df),
                "text",
            ),
            (
                "technical_tokens",
                TfidfVectorizer(max_features=max_token_features, ngram_range=(1, 2), min_df=min_df),
                "technical_tokens_text",
            ),
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=min_df), ["ticker", "v6_track"]),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, solver="lbfgs"),
            ),
        ]
    )


def result_row(split: str, method: str, y_true: list[str], y_pred: list[str], notes: str) -> dict[str, Any]:
    return {"split": split, "method": method, "rows": len(y_true), **metric_dict(y_true, y_pred), "notes": notes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-contexts", required=True)
    parser.add_argument("--val-contexts", required=True)
    parser.add_argument("--test-contexts", required=True)
    parser.add_argument("--label-col", default="target_label_5")
    parser.add_argument("--c-grid", default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-text-features", type=int, default=2000)
    parser.add_argument("--max-token-features", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/tables/17_7_v6_supervised_signal_ceiling_probe.csv")
    parser.add_argument("--grid-output", default="outputs/tables/17_7_v6_supervised_signal_ceiling_grid.csv")
    parser.add_argument("--predictions-output", default="outputs/predictions/current_v6_supervised_signal_ceiling_predictions.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/17_7_v6_supervised_signal_ceiling_probe.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    required_inputs = [args.train_contexts, args.val_contexts, args.test_contexts]
    for path in required_inputs:
        if not Path(path).exists():
            failures.append(f"missing input: {path}")

    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    result_table = pd.DataFrame()
    grid_table = pd.DataFrame()
    predictions_table = pd.DataFrame()
    if not failures:
        try:
            train_contexts = pd.read_parquet(args.train_contexts)
            val = pd.read_parquet(args.val_contexts).drop_duplicates("sample_id").reset_index(drop=True)
            test = pd.read_parquet(args.test_contexts).drop_duplicates("sample_id").reset_index(drop=True)
            for name, frame in [("train_contexts", train_contexts), ("val_contexts", val), ("test_contexts", test)]:
                missing_cols = sorted({args.label_col, "sample_id", "ticker", "event_date", "split"} - set(frame.columns))
                if missing_cols:
                    failures.append(f"{name} missing columns: {missing_cols}")
            if not failures:
                train = select_train_frame(train_contexts, val, test)
                if len(train) < args.min_train_rows:
                    failures.append(f"selected train rows {len(train)} < {args.min_train_rows}")
            if not failures:
                X_train = build_features(train)
                X_val = build_features(val)
                X_test = build_features(test)
                y_train = train[args.label_col].map(normalize_label).tolist()
                y_val = val[args.label_col].map(normalize_label).tolist()
                y_test = test[args.label_col].map(normalize_label).tolist()
                val_technical = [label_from_score(score) for score in val.apply(technical_score, axis=1)]
                test_technical = [label_from_score(score) for score in test.apply(technical_score, axis=1)]
                grid_rows: list[dict[str, Any]] = []
                predictions_by_c: dict[float, tuple[list[str], list[str]]] = {}
                for c_value in [float(item) for item in args.c_grid.split(",") if item.strip()]:
                    model = make_model(c_value, args.min_df, args.max_text_features, args.max_token_features)
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val).tolist()
                    test_pred = model.predict(X_test).tolist()
                    predictions_by_c[c_value] = (val_pred, test_pred)
                    grid_rows.append(
                        {
                            "c_value": c_value,
                            **{f"val_{key}": value for key, value in metric_dict(y_val, val_pred).items()},
                            **{f"test_{key}": value for key, value in metric_dict(y_test, test_pred).items()},
                        }
                    )
                grid_table = pd.DataFrame(grid_rows)
                ranked = grid_table.sort_values(
                    ["val_macro_f1", "val_mcc", "val_accuracy", "c_value"],
                    ascending=[False, False, False, True],
                ).reset_index(drop=True)
                best_c = float(ranked.iloc[0]["c_value"])
                val_pred, test_pred = predictions_by_c[best_c]
                result_table = pd.DataFrame(
                    [
                        result_row("val", "Technical_Rule", y_val, val_technical, "deterministic technical-token baseline"),
                        result_row(
                            "val",
                            "Supervised_LogReg_TFIDF_V6",
                            y_val,
                            val_pred,
                            "train-only regularized text/technical supervised probe selected by validation Macro-F1/MCC",
                        ),
                        result_row("test", "Technical_Rule", y_test, test_technical, "deterministic technical-token baseline"),
                        result_row(
                            "test",
                            "Supervised_LogReg_TFIDF_V6",
                            y_test,
                            test_pred,
                            "held-out test evaluation of validation-selected supervised probe",
                        ),
                    ]
                )
                predictions_table = test[["sample_id", "split", "ticker", "event_date", args.label_col]].copy()
                predictions_table["technical_rule_pred"] = test_technical
                predictions_table["supervised_logreg_pred"] = test_pred
                for metric_name in ["accuracy", "macro_f1", "mcc"]:
                    estimate, low, high = paired_bootstrap_delta(
                        y_test,
                        test_pred,
                        test_technical,
                        metric_name,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + len(metric_name),
                    )
                    metrics[f"delta_{metric_name}_vs_technical"] = estimate
                    metrics[f"delta_{metric_name}_vs_technical_ci95_low"] = low
                    metrics[f"delta_{metric_name}_vs_technical_ci95_high"] = high
                val_supervised = result_table[
                    result_table["split"].eq("val") & result_table["method"].eq("Supervised_LogReg_TFIDF_V6")
                ].iloc[0]
                val_technical_row = result_table[
                    result_table["split"].eq("val") & result_table["method"].eq("Technical_Rule")
                ].iloc[0]
                test_supervised = result_table[
                    result_table["split"].eq("test") & result_table["method"].eq("Supervised_LogReg_TFIDF_V6")
                ].iloc[0]
                test_technical_row = result_table[
                    result_table["split"].eq("test") & result_table["method"].eq("Technical_Rule")
                ].iloc[0]
                metrics.update(
                    {
                        "pipeline_pass": True,
                        "claim_allowed": False,
                        "best_c": best_c,
                        "train_rows": int(len(train)),
                        "val_rows": int(len(val)),
                        "test_rows": int(len(test)),
                        "train_min_date": str(train["event_date"].min()),
                        "train_max_date": str(train["event_date"].max()),
                        "val_min_date": str(pd.to_datetime(val["event_date"]).min()),
                        "test_min_date": str(pd.to_datetime(test["event_date"]).min()),
                        "val_supervised_macro_f1": float(val_supervised["macro_f1"]),
                        "val_supervised_mcc": float(val_supervised["mcc"]),
                        "val_technical_macro_f1": float(val_technical_row["macro_f1"]),
                        "val_technical_mcc": float(val_technical_row["mcc"]),
                        "test_supervised_macro_f1": float(test_supervised["macro_f1"]),
                        "test_supervised_mcc": float(test_supervised["mcc"]),
                        "test_technical_macro_f1": float(test_technical_row["macro_f1"]),
                        "test_technical_mcc": float(test_technical_row["mcc"]),
                        "val_beats_technical_macro_f1": bool(float(val_supervised["macro_f1"]) > float(val_technical_row["macro_f1"])),
                        "val_beats_technical_mcc": bool(float(val_supervised["mcc"]) > float(val_technical_row["mcc"])),
                        "test_beats_technical_macro_f1": bool(float(test_supervised["macro_f1"]) > float(test_technical_row["macro_f1"])),
                        "test_beats_technical_mcc": bool(float(test_supervised["mcc"]) > float(test_technical_row["mcc"])),
                        "paired_ci_support": bool(
                            metrics["delta_macro_f1_vs_technical_ci95_low"] > 0
                            and metrics["delta_mcc_vs_technical_ci95_low"] > 0
                        ),
                        "signal_ceiling_warning": bool(
                            not (float(val_supervised["macro_f1"]) > float(val_technical_row["macro_f1"]))
                            or not (float(test_supervised["macro_f1"]) > float(test_technical_row["macro_f1"]))
                            or metrics["delta_macro_f1_vs_technical_ci95_low"] <= 0
                        ),
                        "bootstrap_repetitions": int(args.n_bootstrap),
                        "seed": int(args.seed),
                        "claim_boundary": (
                            "diagnostic only; a supervised non-LLM probe cannot open the aligned-model forecast claim, "
                            "and any ceiling claim requires validation and test superiority with CI support"
                        ),
                    }
                )
        except Exception as exc:
            failures.append(f"supervised signal ceiling probe failed: {type(exc).__name__}: {str(exc)[:500]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(args.output, index=False)
    Path(args.grid_output).parent.mkdir(parents=True, exist_ok=True)
    grid_table.to_csv(args.grid_output, index=False)
    Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    predictions_table.to_parquet(args.predictions_output, index=False)
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [*required_inputs, args.output, args.grid_output, args.predictions_output, args.metrics],
        STEP,
        extra={
            "references": [
                "scikit-learn cross-validation/model selection user guide",
                "scikit-learn TimeSeriesSplit documentation",
                "Lopez de Prado purged and embargoed validation principle for financial ML",
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
