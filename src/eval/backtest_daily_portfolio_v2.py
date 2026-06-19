from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "16_DAILY_PORTFOLIO_BACKTEST_V2"


def action_to_position(label: str) -> int:
    label = str(label).strip().lower().replace(" ", "_")
    if label in {"strong_up", "mild_up", "long"}:
        return 1
    if label in {"strong_down", "mild_down", "short"}:
        return -1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/predictions/test_predictions.parquet")
    parser.add_argument("--labels", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--short-borrow-bps", type=float, default=0.0)
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.80)
    parser.add_argument("--min-trading-days", type=int, default=1)
    parser.add_argument("--output-json", default="outputs/metrics/backtest_daily_portfolio_v2.json")
    parser.add_argument("--daily-returns", default="outputs/tables/backtest_daily_returns_v2.csv")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    failures: list[str] = []
    if args.split != "test":
        failures.append("backtest must use test split only")
    if not Path(args.predictions).exists():
        failures.append(f"predictions missing: {args.predictions}")
        daily = pd.DataFrame(columns=["date", "portfolio_return", "gross_return", "cost"])
    else:
        preds = pd.read_parquet(args.predictions)
        labels = pd.read_parquet(args.labels)
        if "split" not in preds.columns:
            failures.append("predictions must contain split")
        elif set(preds["split"].dropna()) - {args.split}:
            failures.append("predictions contain non-test rows")
        if "schema_ok" not in preds.columns:
            failures.append("predictions must contain schema_ok for v2 backtest")
        else:
            schema_ok_rate = float(preds["schema_ok"].astype(bool).mean()) if len(preds) else 0.0
            if schema_ok_rate < args.min_schema_ok_rate:
                failures.append(f"prediction schema_ok_rate {schema_ok_rate:.4f} < {args.min_schema_ok_rate:.4f}")
        labels_subset = labels[["sample_id", "ticker", "event_date", "split", "stock_return_h1"]].rename(columns={"split": "label_split"})
        df = preds.merge(labels_subset, on="sample_id", how="inner")
        df = df[df["label_split"] == args.split].copy()
        if "schema_ok" in df.columns:
            df = df[df["schema_ok"].astype(bool)].copy()
        pred_col = "pred_label" if "pred_label" in df.columns else "prediction"
        if pred_col not in df.columns:
            failures.append("predictions must contain pred_label or prediction")
            daily = pd.DataFrame(columns=["date", "portfolio_return", "gross_return", "cost"])
        else:
            if len(df) == 0:
                failures.append("no merged test prediction rows")
            df["position"] = df["action"].apply(action_to_position) if "action" in df.columns else df[pred_col].apply(action_to_position)
            if df["position"].abs().sum() <= 0:
                failures.append("all prediction positions are zero")
            df["date"] = pd.to_datetime(df["event_date"]).dt.date
            agg = df.groupby(["date", "ticker"]).agg(position=("position", "mean"), ret=("stock_return_h1", "first")).reset_index()
            agg["position"] = agg["position"].clip(-1, 1)
            agg["gross"] = agg["position"] * agg["ret"]
            agg["abs_position"] = agg["position"].abs()
            agg = agg.sort_values(["ticker", "date"])
            agg["prev_position"] = agg.groupby("ticker")["position"].shift(1).fillna(0.0)
            agg["turnover"] = (agg["position"] - agg["prev_position"]).abs()
            agg["short_position"] = agg["position"].clip(upper=0).abs()
            by_day = agg.groupby("date").agg(
                gross_return=("gross", "mean"),
                positions=("abs_position", "sum"),
                turnover=("turnover", "sum"),
                short_exposure=("short_position", "sum"),
            ).reset_index()
            by_day["cost"] = (args.cost_bps + args.slippage_bps) / 10000.0 * by_day["turnover"].clip(lower=0)
            by_day["borrow_cost"] = args.short_borrow_bps / 10000.0 * by_day["short_exposure"].clip(lower=0)
            by_day["cost"] = by_day["cost"] + by_day["borrow_cost"]
            by_day["portfolio_return"] = by_day["gross_return"] - by_day["cost"]
            daily = by_day
    Path(args.daily_returns).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.daily_returns, index=False)
    ret = daily["portfolio_return"].astype(float) if len(daily) and "portfolio_return" in daily else []
    sharpe = float(np.sqrt(252) * np.mean(ret) / np.std(ret, ddof=1)) if len(daily) > 1 and np.std(ret, ddof=1) > 0 else None
    metrics = {
        "num_trading_days": int(len(daily)),
        "sharpe_daily_annualized": sharpe,
        "avg_positions_per_day": float(daily["positions"].mean()) if len(daily) and "positions" in daily else 0.0,
        "avg_turnover_per_day": float(daily["turnover"].mean()) if len(daily) and "turnover" in daily else 0.0,
        "total_turnover": float(daily["turnover"].sum()) if len(daily) and "turnover" in daily else 0.0,
        "mean_daily_return": float(daily["portfolio_return"].mean()) if len(daily) and "portfolio_return" in daily else 0.0,
        "nonzero_daily_return_rate": float((daily["portfolio_return"].abs() > 0).mean()) if len(daily) and "portfolio_return" in daily else 0.0,
        "coverage": float((daily["positions"] > 0).mean()) if len(daily) and "positions" in daily else 0.0,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "short_borrow_bps": args.short_borrow_bps,
        "schema_ok_rate_required": args.min_schema_ok_rate,
        "min_trading_days_required": args.min_trading_days,
    }
    if len(daily) == 0:
        failures.append("daily returns output has zero trading days")
    elif len(daily) < args.min_trading_days:
        failures.append(f"trading days {len(daily)} < required {args.min_trading_days}")
    elif metrics["nonzero_daily_return_rate"] <= 0:
        failures.append("daily returns are all zero")
    write_json(args.output_json, metrics)
    write_manifest(args.manifest, [args.output_json, args.daily_returns], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.predictions, args.labels], [args.output_json, args.daily_returns, args.manifest, args.status], metrics, failures, status == "PASS")
    print(json.dumps(metrics, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
