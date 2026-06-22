from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "15_BACKTEST_TRACK_BASELINE_V6"


def compute_sharpe(returns) -> float:
    import numpy as np

    ret = np.asarray(list(returns), dtype=float)
    if len(ret) <= 1:
        return 0.0
    std = float(np.std(ret, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.sqrt(252) * np.mean(ret) / std)


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
    parser.add_argument("--daily-output", "--output-daily", dest="daily_output", default="outputs/tables/backtest_daily_returns_current_v3.csv")
    parser.add_argument("--track-output", "--output-track", dest="track_output", default=None)
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
        for optional in [
            "track",
            "v6_track",
            "news_reasoning_track",
            "has_company_event_news",
            "has_hard_event_news",
            "no_news_context_flag",
            "num_company_event_evidence",
            "num_context_only_evidence",
            "technical_event_tokens_json",
        ]:
            if optional in contexts.columns:
                context_cols.append(optional)
        context_test = contexts[context_cols].copy()
        if "split" in context_test.columns:
            context_test = context_test[context_test["split"] == "test"].copy()
        context_test["date"] = pd.to_datetime(context_test["event_date"]).dt.date
        calendar_dates = pd.DataFrame({"date": sorted(context_test["date"].dropna().unique())})

        def add_calendar_zeros(day_df: pd.DataFrame, calendar: pd.DataFrame | None) -> pd.DataFrame:
            if calendar is None or not len(calendar):
                return day_df.sort_values("date").reset_index(drop=True) if len(day_df) and "date" in day_df else day_df
            out = calendar.merge(day_df, on="date", how="left")
            for col in ["gross_return", "positions", "turnover", "short_exposure", "cost", "borrow_cost", "daily_return_net"]:
                if col not in out.columns:
                    out[col] = 0.0
                else:
                    out[col] = out[col].fillna(0.0)
            return out.sort_values("date").reset_index(drop=True)

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
            if "technical_event_tokens_json" in df.columns:
                from src.reward.build_flow_decision_dataset_v6 import technical_rule_action

                df["technical_rule_action"] = df["technical_event_tokens_json"].apply(technical_rule_action)
                df["technical_rule_position"] = df["technical_rule_action"].apply(action_to_position)
            else:
                df["technical_rule_position"] = 0
            if {"p_mild_up", "p_strong_up", "p_mild_down", "p_strong_down", "p_neutral"} <= set(df.columns):
                up_prob = df["p_mild_up"].astype(float) + df["p_strong_up"].astype(float)
                down_prob = df["p_mild_down"].astype(float) + df["p_strong_down"].astype(float)
                neutral_prob = df["p_neutral"].astype(float)
                df["signal_confidence"] = np.maximum.reduce([up_prob.to_numpy(), down_prob.to_numpy(), neutral_prob.to_numpy()])
                df.loc[(df["position"] > 0) & (up_prob < args.threshold), "position"] = 0
                df.loc[(df["position"] < 0) & (down_prob < args.threshold), "position"] = 0
            else:
                df["signal_confidence"] = 1.0
            
            df = df[df["position"] != 0].copy()
            df["date"] = pd.to_datetime(df["event_date"]).dt.date
            
            # Aggregate by date x ticker
            if "track" not in df.columns:
                if "v6_track" in df.columns:
                    df["track"] = df["v6_track"]
                elif "news_reasoning_track" in df.columns:
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
            daily = add_calendar_zeros(by_day, calendar_dates)
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

            def build_rule_daily(
                source: pd.DataFrame,
                position_col: str,
                calendar: pd.DataFrame | None = None,
            ) -> pd.DataFrame:
                rule = source.copy()
                rule = rule[rule[position_col] != 0].copy()
                if not len(rule):
                    empty = pd.DataFrame(columns=["date", "daily_return_net", "gross_return", "cost", "positions", "turnover"])
                    return add_calendar_zeros(empty, calendar)
                rule["date"] = pd.to_datetime(rule["event_date"]).dt.date
                rule["position"] = rule[position_col].astype(float)
                rule["signal_confidence"] = 1.0
                rule_agg = rule.groupby(["date", "ticker"]).agg(
                    position=("position", "mean"),
                    ret=(args.return_col, "first"),
                    signal_confidence=("signal_confidence", "max"),
                ).reset_index()
                rule_agg["position"] = rule_agg["position"].clip(-1, 1)
                rule_agg = rule_agg.groupby("date", group_keys=False).apply(cap_positions)
                rule_agg["gross"] = rule_agg["position"] * rule_agg["ret"]
                rule_agg["abs_position"] = rule_agg["position"].abs()
                rule_agg = rule_agg.sort_values(["ticker", "date"])
                rule_agg["prev_position"] = rule_agg.groupby("ticker")["position"].shift(1).fillna(0.0)
                rule_agg["turnover"] = (rule_agg["position"] - rule_agg["prev_position"]).abs()
                rule_agg["short_position"] = rule_agg["position"].clip(upper=0).abs()
                rule_daily = rule_agg.groupby("date").agg(
                    gross_return=("gross", "mean"),
                    positions=("abs_position", "sum"),
                    turnover=("turnover", "sum"),
                    short_exposure=("short_position", "sum"),
                ).reset_index()
                rule_daily["cost"] = (args.cost_bps + args.slippage_bps) / 10000.0 * rule_daily["turnover"].clip(lower=0)
                rule_daily["borrow_cost"] = args.short_borrow_bps / 10000.0 * rule_daily["short_exposure"].clip(lower=0)
                rule_daily["cost"] = rule_daily["cost"] + rule_daily["borrow_cost"]
                rule_daily["daily_return_net"] = rule_daily["gross_return"] - rule_daily["cost"]
                return add_calendar_zeros(rule_daily, calendar)

            technical_source = context_test.copy()
            if "technical_event_tokens_json" in technical_source.columns:
                from src.reward.build_flow_decision_dataset_v6 import technical_rule_action

                technical_source["technical_rule_action"] = technical_source["technical_event_tokens_json"].apply(technical_rule_action)
                technical_source["technical_rule_position"] = technical_source["technical_rule_action"].apply(action_to_position)
            else:
                technical_source["technical_rule_position"] = 0
            technical_daily = build_rule_daily(technical_source, "technical_rule_position", calendar_dates)
            if "no_news_context_flag" in technical_source.columns:
                no_news_mask = technical_source["no_news_context_flag"].fillna(False).astype(bool)
            elif "has_company_event_news" in technical_source.columns:
                no_news_mask = ~technical_source["has_company_event_news"].fillna(False).astype(bool)
            else:
                no_news_mask = pd.Series(False, index=technical_source.index)
            no_news_context_rows = int(no_news_mask.sum())
            no_news_calendar = (
                pd.DataFrame({"date": sorted(technical_source.loc[no_news_mask, "date"].dropna().unique())})
                if no_news_context_rows
                else None
            )
            no_news_technical_daily = build_rule_daily(
                technical_source[no_news_mask].copy(),
                "technical_rule_position",
                no_news_calendar,
            )
            
    Path(args.daily_output).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.daily_output, index=False)
    if args.track_output:
        Path(args.track_output).parent.mkdir(parents=True, exist_ok=True)
        track_daily.to_csv(args.track_output, index=False)
    
    ret = daily["daily_return_net"].astype(float).values if len(daily) else np.array([])
    sharpe = compute_sharpe(ret)
    tech_ret = technical_daily["daily_return_net"].astype(float).values if "technical_daily" in locals() and len(technical_daily) else np.array([])
    technical_sharpe = compute_sharpe(tech_ret)
    no_news_ret = (
        no_news_technical_daily["daily_return_net"].astype(float).values
        if "no_news_technical_daily" in locals() and len(no_news_technical_daily)
        else np.array([])
    )
    no_news_technical_sharpe = compute_sharpe(no_news_ret)
    
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
        "technical_rule_num_trading_days": int(len(technical_daily)) if "technical_daily" in locals() else 0,
        "technical_rule_sharpe_annualized": technical_sharpe,
        "technical_rule_mean_daily_return": float(technical_daily["daily_return_net"].mean()) if "technical_daily" in locals() and len(technical_daily) else 0.0,
        "technical_rule_total_turnover": float(technical_daily["turnover"].sum()) if "technical_daily" in locals() and len(technical_daily) and "turnover" in technical_daily else 0.0,
        "technical_only_baseline_name": "Technical_Rule",
        "technical_only_baseline_num_trading_days": int(len(technical_daily)) if "technical_daily" in locals() else 0,
        "technical_only_baseline_sharpe_annualized": technical_sharpe,
        "technical_only_baseline_mean_daily_return": float(technical_daily["daily_return_net"].mean()) if "technical_daily" in locals() and len(technical_daily) else 0.0,
        "no_news_baseline_context_rows": int(no_news_context_rows) if "no_news_context_rows" in locals() else 0,
        "no_news_technical_rule_num_trading_days": int(len(no_news_technical_daily)) if "no_news_technical_daily" in locals() else 0,
        "no_news_technical_rule_sharpe_annualized": no_news_technical_sharpe,
        "no_news_baseline_available": bool("no_news_context_rows" in locals() and no_news_context_rows > 0),
        "delta_sharpe_vs_technical_rule": float(sharpe - technical_sharpe),
        "delta_mean_daily_return_vs_technical_rule": (
            float(daily["daily_return_net"].mean()) if len(daily) and "daily_return_net" in daily else 0.0
        ) - (float(technical_daily["daily_return_net"].mean()) if "technical_daily" in locals() and len(technical_daily) else 0.0),
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
