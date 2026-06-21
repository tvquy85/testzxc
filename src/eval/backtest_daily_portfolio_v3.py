from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "14_DAILY_BACKTEST_ABNORMAL_AND_COSTS"

def action_to_position(label: str) -> int:
    label = str(label).strip().lower().replace(" ", "_")
    if label in {"strong_up", "mild_up", "long"}:
        return 1
    if label in {"strong_down", "mild_down", "short"}:
        return -1
    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/predictions/current_v3_test_predictions.parquet")
    parser.add_argument("--contexts", default="data/processed/ticker_date_contexts_h1_v2_targets.parquet")
    parser.add_argument("--return-col", default="target_return")
    parser.add_argument("--max-positions-per-day", "--position-cap", dest="max_positions_per_day", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--min-trading-days", type=int, default=20)
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.80)
    parser.add_argument("--cost-bps", "--transaction-cost-bps", dest="cost_bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--short-borrow-bps", "--borrow-cost-bps", dest="short_borrow_bps", type=float, default=1.0)
    parser.add_argument("--metrics", "--output", dest="metrics", default="outputs/metrics/backtest_daily_portfolio_current_v3.json")
    parser.add_argument("--daily-output", default="outputs/tables/backtest_daily_returns_current_v3.csv")
    parser.add_argument("--track-output", default=None)
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    failures: list[str] = []
    schema_ok_rate = 0.0
    track_daily = pd.DataFrame(columns=["track", "date", "daily_return_net", "gross_return", "cost", "positions", "turnover"])
    if not Path(args.predictions).exists():
        failures.append(f"predictions missing: {args.predictions}")
        daily = pd.DataFrame(columns=["date", "daily_return_net", "gross_return", "cost"])
        metrics = {}
    else:
        preds = pd.read_parquet(args.predictions)
        contexts = pd.read_parquet(args.contexts)
        
        # Merge predictions with contexts to get returns
        # Contexts already has 'split' and 'target_return'
        if "sample_id" not in preds.columns:
            failures.append("predictions must contain sample_id")
        if "split" in preds.columns and set(preds["split"].dropna()) != {"test"}:
            failures.append("predictions contain non-test rows")
        if "schema_ok" in preds.columns:
            schema_ok_rate = float(preds["schema_ok"].astype(bool).mean()) if len(preds) else 0.0
            if schema_ok_rate < args.min_schema_ok_rate:
                failures.append(f"schema_ok_rate {schema_ok_rate:.4f} < {args.min_schema_ok_rate:.4f}")
            preds = preds[preds["schema_ok"].astype(bool)].copy()
        else:
            schema_ok_rate = 0.0
            failures.append("predictions missing schema_ok")
            
        # Drop ticker and split from preds to avoid merge conflicts
        if "ticker" in preds.columns:
            preds = preds.drop(columns=["ticker"])
        if "event_date" in preds.columns:
            preds = preds.drop(columns=["event_date"])
        if "split" in preds.columns:
            preds = preds.drop(columns=["split"])
            
        context_cols = ["sample_id", "ticker", "event_date", args.return_col, "split"]
        for optional in ["track", "news_reasoning_track", "has_company_event_news"]:
            if optional in contexts.columns:
                context_cols.append(optional)
        df = preds.merge(contexts[context_cols], on="sample_id", how="inner")
        
        # Only evaluate on test split
        df = df[df["split"] == "test"].copy()
        
        if len(df) == 0:
            failures.append("no merged test prediction rows")
            daily = pd.DataFrame(columns=["date", "daily_return_net", "gross_return", "cost"])
        else:
            def extract_action(row):
                try:
                    import json
                    parsed = json.loads(row)
                    return parsed.get("action", "hold")
                except:
                    return "hold"
                    
            if "action" not in df.columns and "parsed_json" in df.columns:
                df["action"] = df["parsed_json"].apply(extract_action)
                
            pred_col = "pred_label" if "pred_label" in df.columns else "prediction"
            df["position"] = df["action"].apply(action_to_position) if "action" in df.columns else df[pred_col].apply(action_to_position)
            if {"p_mild_up", "p_strong_up", "p_mild_down", "p_strong_down", "p_neutral"} <= set(df.columns):
                up_prob = df["p_mild_up"].astype(float) + df["p_strong_up"].astype(float)
                down_prob = df["p_mild_down"].astype(float) + df["p_strong_down"].astype(float)
                neutral_prob = df["p_neutral"].astype(float)
                df["signal_confidence"] = np.maximum.reduce([up_prob.to_numpy(), down_prob.to_numpy(), neutral_prob.to_numpy()])
                df.loc[(df["position"] > 0) & (up_prob < args.threshold), "position"] = 0
                df.loc[(df["position"] < 0) & (down_prob < args.threshold), "position"] = 0
            else:
                df["signal_confidence"] = 1.0
            
            # Use threshold if predicted probabilities exist
            df = df[df["position"] != 0].copy()
            df["date"] = pd.to_datetime(df["event_date"]).dt.date
            
            # Aggregate by date x ticker
            if "track" not in df.columns:
                if "news_reasoning_track" in df.columns:
                    df["track"] = df["news_reasoning_track"]
                elif "has_company_event_news" in df.columns:
                    df["track"] = df["has_company_event_news"].map(lambda x: "news_technical" if bool(x) else "technical_only")
                else:
                    df["track"] = "unknown"
            agg = df.groupby(["date", "ticker"]).agg(
                position=("position", "mean"),
                ret=(args.return_col, "first"),
                signal_confidence=("signal_confidence", "max"),
                track=("track", "first"),
            ).reset_index()
            agg["position"] = agg["position"].clip(-1, 1)
            
            # Cap top positions per day deterministically by confidence, never randomly.
            def cap_positions(group):
                if len(group) > args.max_positions_per_day:
                    return group.sort_values(["signal_confidence", "ticker"], ascending=[False, True]).head(args.max_positions_per_day)
                return group
            
            agg = agg.groupby("date", group_keys=False).apply(cap_positions)
            
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
            by_day["daily_return_net"] = by_day["gross_return"] - by_day["cost"]
            daily = by_day
            track_daily = agg.groupby(["track", "date"]).agg(
                gross_return=("gross", "mean"),
                positions=("abs_position", "sum"),
                turnover=("turnover", "sum"),
                short_exposure=("short_position", "sum"),
            ).reset_index()
            if len(track_daily):
                track_daily["cost"] = (args.cost_bps + args.slippage_bps) / 10000.0 * track_daily["turnover"].clip(lower=0)
                track_daily["borrow_cost"] = args.short_borrow_bps / 10000.0 * track_daily["short_exposure"].clip(lower=0)
                track_daily["cost"] = track_daily["cost"] + track_daily["borrow_cost"]
                track_daily["daily_return_net"] = track_daily["gross_return"] - track_daily["cost"]
            
    Path(args.daily_output).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.daily_output, index=False)
    if args.track_output:
        Path(args.track_output).parent.mkdir(parents=True, exist_ok=True)
        track_daily.to_csv(args.track_output, index=False)
    
    ret = daily["daily_return_net"].astype(float).values if len(daily) else np.array([])
    sharpe = float(np.sqrt(252) * np.mean(ret) / np.std(ret, ddof=1)) if len(ret) > 1 and np.std(ret, ddof=1) > 0 else 0.0
    
    # Sortino ratio
    downside_returns = ret[ret < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
    sortino = float(np.sqrt(252) * np.mean(ret) / downside_std) if downside_std > 0 else 0.0
    
    # Max Drawdown
    cumulative = np.cumprod(1 + ret) if len(ret) else np.array([])
    running_max = np.maximum.accumulate(cumulative) if len(cumulative) else np.array([])
    drawdown = (running_max - cumulative) / running_max if len(running_max) else np.array([])
    max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0
    
    num_trading_days = int(len(daily))
    alpha_claim_allowed = True
    if num_trading_days < 60:
        alpha_claim_allowed = False
    if sharpe < 0:
        alpha_claim_allowed = False

    metrics = {
        "num_trading_days": num_trading_days,
        "sharpe_daily_annualized": sharpe,
        "sharpe_annualized": sharpe,
        "sortino_daily_annualized": sortino,
        "max_drawdown": max_drawdown,
        "avg_positions_per_day": float(daily["positions"].mean()) if len(daily) and "positions" in daily else 0.0,
        "avg_turnover_per_day": float(daily["turnover"].mean()) if len(daily) and "turnover" in daily else 0.0,
        "total_turnover": float(daily["turnover"].sum()) if len(daily) and "turnover" in daily else 0.0,
        "mean_daily_return": float(daily["daily_return_net"].mean()) if len(daily) and "daily_return_net" in daily else 0.0,
        "coverage": float((daily["positions"] > 0).mean()) if len(daily) and "positions" in daily else 0.0,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "short_borrow_bps": args.short_borrow_bps,
        "return_column": args.return_col,
        "schema_ok_rate": schema_ok_rate,
        "schema_ok_rate_required": args.min_schema_ok_rate,
        "min_trading_days_required": args.min_trading_days,
        "alpha_claim_allowed": alpha_claim_allowed,
        "pipeline_pass": not bool(failures)
    }

    if len(daily) == 0:
        failures.append("daily returns output has zero trading days")
        metrics["pipeline_pass"] = False
    if num_trading_days < args.min_trading_days:
        failures.append(f"trading days {num_trading_days} < {args.min_trading_days}")
        metrics["pipeline_pass"] = False
    if metrics["total_turnover"] <= 0:
        failures.append("total turnover is zero")
        metrics["pipeline_pass"] = False
    if len(daily) and float((daily["daily_return_net"].abs() > 0).mean()) == 0.0:
        failures.append("all daily returns are zero")
        metrics["pipeline_pass"] = False

    write_json(args.metrics, metrics)
    manifest_outputs = [args.predictions, args.contexts, args.metrics, args.daily_output]
    if args.track_output:
        manifest_outputs.append(args.track_output)
    write_manifest(args.manifest, manifest_outputs, STEP)
    status = "PASS" if metrics["pipeline_pass"] else "FAIL"
    write_status(
        args.status, 
        STEP, 
        status, 
        [args.predictions, args.contexts], 
        [args.metrics, args.daily_output, *([args.track_output] if args.track_output else []), args.manifest, args.status], 
        metrics, 
        failures, 
        status == "PASS"
    )
    print(json.dumps(metrics, indent=2))
    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
