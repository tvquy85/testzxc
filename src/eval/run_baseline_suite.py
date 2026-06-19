from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.classification_metrics import compute_metrics
from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "18_BASELINES_EXPANSION_AND_SEEDS"
LABELS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
REQUIRED_BASELINES = ["B1_FinBERT_LR", "B2_Technical_LightGBM", "B3_News_Technical_Late_Fusion", "B4_DLinear"]
ALL_BASELINES = REQUIRED_BASELINES + [
    "B5_Chronos",
    "B6_TTM",
    "B7_MOMENT",
    "B8_LLM_ZeroShot",
    "B9_LLM_SFT",
    "B10_LLM_RWSFT",
    "B11_LLM_DPO_PROXY",
    "B12_FIRE_Fin_FlowV2",
]
ID_COLS = {
    "sample_id",
    "ticker",
    "event_date",
    "timestamp_utc",
    "window_start_date",
    "window_end_date",
    "headline",
    "body",
    "horizon",
    "label_5",
    "split",
    "stock_return_h1",
    "market_return_h1",
    "abnormal_return_h1",
}


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numeric_feature_cols(df: Any) -> list[str]:
    import numpy as np

    cols = []
    for col in df.columns:
        if col in ID_COLS or col.endswith("_x") or col.endswith("_y"):
            continue
        if df[col].dtype in [np.float32, np.float64, np.int32, np.int64, np.bool_]:
            cols.append(col)
    return cols


def sample_split(df: Any, seed: int, max_train_rows: int | None, max_test_rows: int | None) -> Any:
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    if max_train_rows and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=seed)
    if max_test_rows and len(test) > max_test_rows:
        test = test.sample(n=max_test_rows, random_state=seed)
    return train, test


def metric_row(baseline: str, seed: int, y_true: Any, y_pred: Any, y_prob: Any, classes: list[str], train_rows: int, test_rows: int, feature_dim: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = compute_metrics(y_true, y_pred, y_prob, classes)
    row = {
        "baseline": baseline,
        "seed": seed,
        "status": "PASS",
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "mcc": metrics.get("mcc"),
        "brier": metrics.get("brier_score_multiclass"),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "feature_dim": int(feature_dim),
    }
    if extra:
        row.update(extra)
    return row


def label_to_action(label: str) -> str:
    label = str(label).strip().lower()
    if label in {"strong_down", "mild_down"}:
        return "short"
    if label in {"strong_up", "mild_up"}:
        return "long"
    return "hold"


def prediction_rows(baseline: str, seed: int, test: Any, y_pred: Any, y_prob: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_list = list(y_pred)
    for idx, (_, row) in enumerate(test.reset_index(drop=True).iterrows()):
        pred_label = str(pred_list[idx])
        out = {
            "sample_id": row["sample_id"],
            "split": "test",
            "baseline": baseline,
            "seed": int(seed),
            "pred_label": pred_label,
            "action": label_to_action(pred_label),
            "schema_ok": True,
            "parse_ok": True,
            "model_checkpoint": baseline,
            "run_id": f"{baseline}_seed_{seed}",
        }
        for label_idx, label in enumerate(LABELS):
            out[f"p_{label}"] = float(y_prob[idx, label_idx])
        rows.append(out)
    return rows


def align_probs(classes_seen: Any, probs: Any) -> Any:
    import numpy as np

    aligned = np.zeros((probs.shape[0], len(LABELS)), dtype=float)
    for idx, cls in enumerate(classes_seen):
        if cls in LABELS:
            aligned[:, LABELS.index(cls)] = probs[:, idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(aligned, row_sum, out=np.full_like(aligned, 1.0 / len(LABELS)), where=row_sum > 0)
    return aligned


def run_logreg_baseline(baseline: str, df: Any, feature_cols: list[str], seed: int, max_train_rows: int | None, max_test_rows: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, test = sample_split(df.dropna(subset=feature_cols + ["label_5"]), seed, max_train_rows, max_test_rows)
    if train.empty or test.empty:
        raise RuntimeError(f"{baseline} has empty train/test after feature merge")
    x_train = np.nan_to_num(train[feature_cols].astype(float).values)
    x_test = np.nan_to_num(test[feature_cols].astype(float).values)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, multi_class="auto"),
    )
    clf.fit(x_train, train["label_5"].values)
    y_pred = clf.predict(x_test)
    y_prob = align_probs(clf.named_steps["logisticregression"].classes_, clf.predict_proba(x_test))
    return (
        metric_row(baseline, seed, test["label_5"].values, y_pred, y_prob, LABELS, len(train), len(test), len(feature_cols)),
        prediction_rows(baseline, seed, test, y_pred, y_prob),
    )


def run_lgbm_baseline(baseline: str, df: Any, feature_cols: list[str], seed: int, max_train_rows: int | None, max_test_rows: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    from lightgbm import LGBMClassifier

    train, test = sample_split(df.dropna(subset=["label_5"]), seed, max_train_rows, max_test_rows)
    if train.empty or test.empty:
        raise RuntimeError(f"{baseline} has empty train/test")
    x_train = np.nan_to_num(train[feature_cols].astype(float).values)
    x_test = np.nan_to_num(test[feature_cols].astype(float).values)
    clf = LGBMClassifier(
        objective="multiclass",
        num_class=len(LABELS),
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(x_train, train["label_5"].values)
    y_pred = clf.predict(x_test)
    y_prob = align_probs(clf.classes_, clf.predict_proba(x_test))
    return (
        metric_row(baseline, seed, test["label_5"].values, y_pred, y_prob, LABELS, len(train), len(test), len(feature_cols)),
        prediction_rows(baseline, seed, test, y_pred, y_prob),
    )


def run_dlinear_baseline(df: Any, feature_cols: list[str], seed: int, max_train_rows: int | None, max_test_rows: int | None, epochs: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    train, test = sample_split(df.dropna(subset=["label_5"]), seed, max_train_rows, max_test_rows)
    if train.empty or test.empty:
        raise RuntimeError("B4_DLinear has empty train/test")
    x_train_raw = np.nan_to_num(train[feature_cols].astype(float).values.astype("float32"))
    x_test_raw = np.nan_to_num(test[feature_cols].astype(float).values.astype("float32"))
    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0) + 1e-6
    x_train = (x_train_raw - mean) / std
    x_test = (x_test_raw - mean) / std
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    y_train = np.array([label_to_idx[label] for label in train["label_5"].values], dtype="int64")
    y_test = test["label_5"].values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(x_train.shape[1], len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=512, shuffle=True)
    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    y_pred = [LABELS[idx] for idx in pred_idx]
    return (
        metric_row("B4_DLinear", seed, y_test, y_pred, probs, LABELS, len(train), len(test), len(feature_cols), {"device": str(device), "epochs": epochs}),
        prediction_rows("B4_DLinear", seed, test, y_pred, probs),
    )


def load_inputs(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    labels = pd.read_parquet(args.labels)
    technical = pd.read_parquet(args.technical_features)
    finbert = pd.read_parquet(args.finbert_features)
    if "split" not in labels.columns:
        raise RuntimeError("labels input must contain split")
    tech_df = labels.merge(technical, on="sample_id", how="inner")
    finbert_df = labels.merge(finbert, on="sample_id", how="inner")
    late_df = tech_df.merge(finbert, on="sample_id", how="inner", suffixes=("", "_finbert"))
    frames = {"technical": tech_df, "finbert": finbert_df, "late_fusion": late_df}
    return labels, frames


def aggregate(rows: list[dict[str, Any]]) -> Any:
    import pandas as pd

    df = pd.DataFrame(rows)
    passed = df[df["status"] == "PASS"].copy()
    if passed.empty:
        return pd.DataFrame()
    return (
        passed.groupby("baseline", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
            brier_mean=("brier", "mean"),
            train_rows_min=("train_rows", "min"),
            test_rows_min=("test_rows", "min"),
            feature_dim=("feature_dim", "max"),
        )
        .reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--labels", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--technical-features", default="data/indicators/technical_features_h1_v2.parquet")
    parser.add_argument("--finbert-features", default="data/indicators/finbert_features_h1.parquet")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--dlinear-epochs", type=int, default=5)
    parser.add_argument("--write-not-run-only", action="store_true")
    parser.add_argument("--output", default="outputs/tables/baseline_suite_by_seed.csv")
    parser.add_argument("--aggregate", default="outputs/tables/baseline_suite_aggregate.csv")
    parser.add_argument("--predictions-output", default="outputs/predictions/baseline_suite_predictions.parquet")
    parser.add_argument("--summary", default="outputs/metrics/baseline_suite_summary.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    if args.write_not_run_only:
        rows = [{"baseline": baseline, "seed": seed, "status": "NOT_RUN"} for baseline in ALL_BASELINES for seed in args.seeds]
        failures.append("write-not-run-only requested")
        agg = pd.DataFrame()
    else:
        try:
            _, frames = load_inputs(args)
            tech_cols = numeric_feature_cols(frames["technical"])
            finbert_cols = ["finbert_0", "finbert_1", "finbert_2"]
            late_cols = numeric_feature_cols(frames["late_fusion"])
            for seed in args.seeds:
                set_seed(seed)
                metric, preds = run_logreg_baseline("B1_FinBERT_LR", frames["finbert"], finbert_cols, seed, args.max_train_rows, args.max_test_rows)
                rows.append(metric)
                pred_rows.extend(preds)
                metric, preds = run_lgbm_baseline("B2_Technical_LightGBM", frames["technical"], tech_cols, seed, args.max_train_rows, args.max_test_rows)
                rows.append(metric)
                pred_rows.extend(preds)
                metric, preds = run_lgbm_baseline("B3_News_Technical_Late_Fusion", frames["late_fusion"], late_cols, seed, args.max_train_rows, args.max_test_rows)
                rows.append(metric)
                pred_rows.extend(preds)
                metric, preds = run_dlinear_baseline(frames["technical"], tech_cols, seed, args.max_train_rows, args.max_test_rows, args.dlinear_epochs)
                rows.append(metric)
                pred_rows.extend(preds)
            for baseline in ALL_BASELINES:
                if baseline not in REQUIRED_BASELINES:
                    for seed in args.seeds:
                        rows.append({"baseline": baseline, "seed": seed, "status": "NOT_RUN", "multi_seed_missing": True})
            agg = aggregate(rows)
        except Exception as exc:
            failures.append(f"baseline run failed: {type(exc).__name__}: {str(exc)[:500]}")
            agg = pd.DataFrame()

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    Path(args.aggregate).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.aggregate, index=False)
    pred_df = pd.DataFrame(pred_rows)
    Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(args.predictions_output, index=False)
    required = df[df["baseline"].isin(REQUIRED_BASELINES)] if not df.empty and "baseline" in df.columns else pd.DataFrame()
    required_pass = (
        not required.empty
        and set(required["baseline"]) == set(REQUIRED_BASELINES)
        and set(required["seed"]) == set(args.seeds)
        and set(required["status"]) == {"PASS"}
    )
    if not required_pass:
        failures.append("B1-B4 have not been run successfully for all required seeds")
    summary = {
        "baseline_count": len(ALL_BASELINES),
        "required_minimum": REQUIRED_BASELINES,
        "seeds": args.seeds,
        "required_minimum_pass": bool(required_pass),
        "multi_seed_missing": bool((df.get("status") == "NOT_RUN").any()) if not df.empty else True,
        "run_counts": df.groupby("status").size().to_dict() if not df.empty and "status" in df.columns else {},
        "prediction_rows": int(len(pred_df)),
        "max_train_rows": args.max_train_rows,
        "max_test_rows": args.max_test_rows,
        "dlinear_epochs": args.dlinear_epochs,
    }
    write_json(args.summary, summary)
    write_manifest(args.manifest, [args.output, args.aggregate, args.predictions_output, args.summary], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.config, args.labels, args.technical_features, args.finbert_features], [args.output, args.aggregate, args.predictions_output, args.summary, args.manifest, args.status], summary, failures, status == "PASS")
    print(json.dumps({"status": status, "summary": summary, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
