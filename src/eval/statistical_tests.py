from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "19_ABLATION_AND_STATISTICAL_TESTS"
LABELS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
REQUIRED_BASELINES = ["B1_FinBERT_LR", "B2_Technical_LightGBM", "B3_News_Technical_Late_Fusion", "B4_DLinear"]
METRIC_KEYS = ["accuracy", "macro_f1", "mcc"]


def action_to_position(label: str) -> int:
    label = str(label).strip().lower().replace(" ", "_")
    if label in {"strong_up", "mild_up", "long"}:
        return 1
    if label in {"strong_down", "mild_down", "short"}:
        return -1
    return 0


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def ci(values: list[float]) -> dict[str, float | None]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def sharpe_daily(values: Any) -> float | None:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return None
    std = arr.std(ddof=1)
    if std <= 0:
        return None
    return float(np.sqrt(252.0) * arr.mean() / std)


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def prepare_predictions(preds: Any, labels: Any, failures: list[str], source: str) -> Any:
    import pandas as pd

    required = {"sample_id", "split", "pred_label"}
    missing = required - set(preds.columns)
    if missing:
        failures.append(f"{source} predictions missing columns: {sorted(missing)}")
        return pd.DataFrame()
    df = preds.copy()
    if "schema_ok" not in df.columns:
        failures.append(f"{source} predictions missing schema_ok")
        df["schema_ok"] = False
    df = df[(df["split"] == "test") & (df["schema_ok"].astype(bool))].copy()
    if df.empty:
        failures.append(f"{source} predictions have no schema-ok test rows")
        return df
    label_cols = ["sample_id", "ticker", "event_date", "stock_return_h1", "label_5", "split"]
    merged = df.merge(labels[label_cols].rename(columns={"split": "label_split"}), on="sample_id", how="inner")
    merged = merged[merged["label_split"] == "test"].copy()
    if merged.empty:
        failures.append(f"{source} predictions have no merged test labels")
    return merged


def paired_prediction_bootstrap(ours: Any, baseline: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    import numpy as np

    failures: list[str] = []
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for baseline_name in REQUIRED_BASELINES:
        for seed in args.seeds:
            base_slice = baseline[(baseline["baseline"] == baseline_name) & (baseline["seed"].astype(int) == seed)].copy()
            if base_slice.empty:
                failures.append(f"missing baseline predictions for {baseline_name} seed {seed}")
                continue
            paired = ours[["sample_id", "label_5", "pred_label"]].merge(
                base_slice[["sample_id", "pred_label"]],
                on="sample_id",
                how="inner",
                suffixes=("_ours", "_baseline"),
            )
            paired = paired.dropna(subset=["label_5", "pred_label_ours", "pred_label_baseline"])
            if len(paired) < args.min_paired_rows:
                failures.append(f"{baseline_name} seed {seed} paired rows {len(paired)} < {args.min_paired_rows}")
                continue
            y_true = paired["label_5"].to_numpy()
            ours_pred = paired["pred_label_ours"].to_numpy()
            base_pred = paired["pred_label_baseline"].to_numpy()
            ours_metrics = classification_metrics(y_true, ours_pred)
            base_metrics = classification_metrics(y_true, base_pred)
            boot: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
            n = len(paired)
            for _ in range(args.bootstrap_samples):
                idx = rng.integers(0, n, size=n)
                sample_true = y_true[idx]
                sample_ours = ours_pred[idx]
                sample_base = base_pred[idx]
                sample_ours_metrics = classification_metrics(sample_true, sample_ours)
                sample_base_metrics = classification_metrics(sample_true, sample_base)
                for key in METRIC_KEYS:
                    boot[key].append(sample_ours_metrics[key] - sample_base_metrics[key])
            row: dict[str, Any] = {
                "baseline": baseline_name,
                "seed": int(seed),
                "n_paired_rows": int(n),
            }
            for key in METRIC_KEYS:
                interval = ci(boot[key])
                row[f"ours_{key}"] = ours_metrics[key]
                row[f"baseline_{key}"] = base_metrics[key]
                row[f"delta_{key}"] = ours_metrics[key] - base_metrics[key]
                row[f"delta_{key}_ci_low"] = interval["ci_low"]
                row[f"delta_{key}_ci_high"] = interval["ci_high"]
                row[f"delta_{key}_p_le_0"] = float(np.mean(np.asarray(boot[key]) <= 0.0))
            rows.append(row)
    return rows, failures


def daily_returns_from_predictions(preds: Any, labels: Any, cost_bps: float) -> Any:
    import pandas as pd

    if {"ticker", "event_date", "stock_return_h1"}.issubset(preds.columns):
        df = preds.copy()
        if "label_split" in df.columns:
            df = df[df["label_split"] == "test"].copy()
        elif "split" in df.columns:
            df = df[df["split"] == "test"].copy()
    else:
        label_cols = ["sample_id", "ticker", "event_date", "stock_return_h1", "split"]
        df = preds.merge(labels[label_cols].rename(columns={"split": "label_split"}), on="sample_id", how="inner")
        df = df[df["label_split"] == "test"].copy()
    if "schema_ok" in df.columns:
        df = df[df["schema_ok"].astype(bool)].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "portfolio_return", "gross_return", "cost", "turnover"])
    signal = df["action"] if "action" in df.columns else df["pred_label"]
    df["position"] = signal.apply(action_to_position)
    df["date"] = pd.to_datetime(df["event_date"]).dt.date
    agg = df.groupby(["date", "ticker"]).agg(position=("position", "mean"), ret=("stock_return_h1", "first")).reset_index()
    agg["position"] = agg["position"].clip(-1, 1)
    agg["gross"] = agg["position"] * agg["ret"]
    agg = agg.sort_values(["ticker", "date"])
    agg["prev_position"] = agg.groupby("ticker")["position"].shift(1).fillna(0.0)
    agg["turnover"] = (agg["position"] - agg["prev_position"]).abs()
    out = agg.groupby("date").agg(gross_return=("gross", "mean"), turnover=("turnover", "sum")).reset_index()
    out["cost"] = cost_bps / 10000.0 * out["turnover"].clip(lower=0)
    out["portfolio_return"] = out["gross_return"] - out["cost"]
    return out.sort_values("date").reset_index(drop=True)


def paired_block_bootstrap(ours: Any, baseline: Any, labels: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    import numpy as np

    failures: list[str] = []
    rng = np.random.default_rng(args.seed + 17)
    rows: list[dict[str, Any]] = []
    for baseline_name in REQUIRED_BASELINES:
        for seed in args.seeds:
            base_slice = baseline[(baseline["baseline"] == baseline_name) & (baseline["seed"].astype(int) == seed)].copy()
            if base_slice.empty:
                failures.append(f"missing baseline predictions for daily return test: {baseline_name} seed {seed}")
                continue
            overlap = sorted(set(ours["sample_id"]) & set(base_slice["sample_id"]))
            if len(overlap) < args.min_paired_rows:
                failures.append(f"{baseline_name} seed {seed} daily overlap {len(overlap)} < {args.min_paired_rows}")
                continue
            ours_daily = daily_returns_from_predictions(ours[ours["sample_id"].isin(overlap)].copy(), labels, args.cost_bps)
            base_daily = daily_returns_from_predictions(base_slice[base_slice["sample_id"].isin(overlap)].copy(), labels, args.cost_bps)
            paired = ours_daily[["date", "portfolio_return"]].merge(
                base_daily[["date", "portfolio_return"]],
                on="date",
                how="inner",
                suffixes=("_ours", "_baseline"),
            )
            if len(paired) < args.min_block_days:
                failures.append(f"{baseline_name} seed {seed} paired trading days {len(paired)} < {args.min_block_days}")
                continue
            ours_ret = paired["portfolio_return_ours"].astype(float).to_numpy()
            base_ret = paired["portfolio_return_baseline"].astype(float).to_numpy()
            n = len(paired)
            block_len = min(max(1, args.block_length), n)
            deltas_mean: list[float] = []
            deltas_sharpe: list[float] = []
            for _ in range(args.bootstrap_samples):
                starts = rng.integers(0, n, size=math.ceil(n / block_len))
                idx = np.concatenate([(np.arange(start, start + block_len) % n) for start in starts])[:n]
                sample_ours = ours_ret[idx]
                sample_base = base_ret[idx]
                deltas_mean.append(float(sample_ours.mean() - sample_base.mean()))
                ours_sharpe = sharpe_daily(sample_ours)
                base_sharpe = sharpe_daily(sample_base)
                if ours_sharpe is not None and base_sharpe is not None:
                    deltas_sharpe.append(ours_sharpe - base_sharpe)
            mean_interval = ci(deltas_mean)
            sharpe_interval = ci(deltas_sharpe)
            row = {
                "baseline": baseline_name,
                "seed": int(seed),
                "n_overlap_rows": int(len(overlap)),
                "n_paired_days": int(n),
                "ours_mean_daily_return": float(ours_ret.mean()),
                "baseline_mean_daily_return": float(base_ret.mean()),
                "delta_mean_daily_return": float(ours_ret.mean() - base_ret.mean()),
                "delta_mean_daily_return_ci_low": mean_interval["ci_low"],
                "delta_mean_daily_return_ci_high": mean_interval["ci_high"],
                "delta_mean_daily_return_p_le_0": float(np.mean(np.asarray(deltas_mean) <= 0.0)),
                "ours_sharpe": finite_float(sharpe_daily(ours_ret)),
                "baseline_sharpe": finite_float(sharpe_daily(base_ret)),
                "delta_sharpe": None,
                "delta_sharpe_ci_low": sharpe_interval["ci_low"],
                "delta_sharpe_ci_high": sharpe_interval["ci_high"],
                "delta_sharpe_p_le_0": float(np.mean(np.asarray(deltas_sharpe) <= 0.0)) if deltas_sharpe else None,
            }
            if row["ours_sharpe"] is not None and row["baseline_sharpe"] is not None:
                row["delta_sharpe"] = row["ours_sharpe"] - row["baseline_sharpe"]
            rows.append(row)
    return rows, failures


def ablation_summary(path: str) -> dict[str, Any]:
    import pandas as pd

    p = Path(path)
    if not p.exists():
        return {"path": path, "status": "MISSING", "not_used_as_evidence": True}
    df = pd.read_csv(p)
    status_counts = df["status"].value_counts(dropna=False).to_dict() if "status" in df.columns else {}
    all_not_run = bool(len(df) and set(df["status"].astype(str)) == {"NOT_RUN"}) if "status" in df.columns else False
    return {
        "path": path,
        "row_count": int(len(df)),
        "status_counts": status_counts,
        "not_used_as_evidence": all_not_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-predictions", default="outputs/predictions/test_predictions_qwen3_dpo_flash_medium.parquet")
    parser.add_argument("--baseline-predictions", default="outputs/predictions/baseline_suite_predictions.parquet")
    parser.add_argument("--labels", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--daily-returns", default="outputs/tables/backtest_daily_returns_v2.csv")
    parser.add_argument("--ablation-results", default="outputs/tables/ablation_suite_results.csv")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--min-paired-rows", type=int, default=100)
    parser.add_argument("--min-block-days", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--output", default="outputs/metrics/statistical_tests.json")
    parser.add_argument("--prediction-table", default="outputs/tables/prediction_bootstrap_comparisons.csv")
    parser.add_argument("--backtest-table", default="outputs/tables/backtest_block_bootstrap_comparisons.csv")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    failures: list[str] = []
    for name, path in {
        "ours predictions": args.ours_predictions,
        "baseline predictions": args.baseline_predictions,
        "labels": args.labels,
        "daily returns": args.daily_returns,
    }.items():
        if not Path(path).exists():
            failures.append(f"{name} missing: {path}")
    if failures:
        prediction_rows: list[dict[str, Any]] = []
        backtest_rows: list[dict[str, Any]] = []
        report = {
            "prediction_bootstrap": {"comparisons": 0},
            "backtest_block_bootstrap": {"comparisons": 0},
            "ablations": ablation_summary(args.ablation_results),
            "failures": failures,
        }
    else:
        labels = pd.read_parquet(args.labels)
        ours_raw = pd.read_parquet(args.ours_predictions)
        baseline_raw = pd.read_parquet(args.baseline_predictions)
        daily = pd.read_csv(args.daily_returns)
        if len(daily) < args.min_block_days:
            failures.append(f"official daily returns has {len(daily)} days < {args.min_block_days}")
        ours = prepare_predictions(ours_raw, labels, failures, "ours")
        baseline = prepare_predictions(baseline_raw, labels, failures, "baseline")
        if "baseline" not in baseline.columns or "seed" not in baseline.columns:
            failures.append("baseline predictions must contain baseline and seed columns")
        prediction_rows, prediction_failures = paired_prediction_bootstrap(ours, baseline, args) if not baseline.empty else ([], [])
        backtest_rows, backtest_failures = paired_block_bootstrap(ours, baseline, labels, args) if not baseline.empty else ([], [])
        failures.extend(prediction_failures)
        failures.extend(backtest_failures)
        if not prediction_rows:
            failures.append("no paired prediction bootstrap comparisons were computed")
        if not backtest_rows:
            failures.append("no paired block bootstrap comparisons were computed")
        report = {
            "prediction_bootstrap": {
                "comparisons": len(prediction_rows),
                "bootstrap_samples": args.bootstrap_samples,
                "metrics": METRIC_KEYS,
                "required_baselines": REQUIRED_BASELINES,
                "seeds": args.seeds,
            },
            "backtest_block_bootstrap": {
                "comparisons": len(backtest_rows),
                "bootstrap_samples": args.bootstrap_samples,
                "block_length": args.block_length,
                "min_block_days": args.min_block_days,
                "cost_bps": args.cost_bps,
            },
            "ablations": ablation_summary(args.ablation_results),
        }

    pred_df = pd.DataFrame(prediction_rows)
    backtest_df = pd.DataFrame(backtest_rows)
    Path(args.prediction_table).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.prediction_table, index=False)
    Path(args.backtest_table).parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(args.backtest_table, index=False)
    write_json(args.output, report)
    write_manifest(args.manifest, [args.output, args.prediction_table, args.backtest_table], STEP)
    status = "PASS" if not failures else "FAIL"
    metrics = {
        "prediction_bootstrap_comparisons": int(len(prediction_rows)),
        "backtest_block_bootstrap_comparisons": int(len(backtest_rows)),
        "bootstrap_samples": args.bootstrap_samples,
        "block_length": args.block_length,
        "ablation_not_used_as_evidence": report["ablations"].get("not_used_as_evidence"),
    }
    write_status(
        args.status,
        STEP,
        status,
        [args.ours_predictions, args.baseline_predictions, args.labels, args.daily_returns, args.ablation_results],
        [args.output, args.prediction_table, args.backtest_table, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
