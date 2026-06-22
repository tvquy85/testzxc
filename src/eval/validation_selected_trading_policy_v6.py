from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.backtest_daily_portfolio_v3 import action_to_position, compute_sharpe
from src.eval.trading_policy_variant_probe_v6 import max_drawdown, sortino
from src.repro.run_v6_statistical_tests import bootstrap_series_stat, paired_block_bootstrap_delta
from src.reward.build_flow_decision_dataset_v6 import technical_rule_action
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "15_6_VALIDATION_SELECTED_TRADING_POLICY_V6"
SELECTED_STRATEGY = "Validation_Selected_DPO_Policy"
DEFAULT_STRATEGY = "Qwen_DPO_V6_Default_Threshold_0p20_Cap20"
TECHNICAL_STRATEGY = "Technical_Rule"
PROB_COLS = ["p_strong_down", "p_mild_down", "p_neutral", "p_mild_up", "p_strong_up"]


def parse_grid(spec: str, *, cast=float) -> list[Any]:
    text = str(spec).strip()
    if not text:
        return []
    if "," in text:
        return [cast(part.strip()) for part in text.split(",") if part.strip()]
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        if len(parts) != 3:
            raise ValueError(f"grid range must be start:stop:step, got {spec!r}")
        start, stop, step = parts
        if step <= 0:
            raise ValueError("grid step must be positive")
        values = []
        current = start
        while current <= stop + step / 10.0:
            values.append(cast(round(current, 10)))
            current += step
        return values
    return [cast(text)]


def calendar_from_contexts(contexts: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(contexts["event_date"]).dt.date.dropna().unique())
    return pd.DataFrame({"date": dates})


def add_calendar_zeros(day_df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    out = calendar.merge(day_df, on="date", how="left")
    for col in ["gross_return", "positions", "turnover", "short_exposure", "cost", "borrow_cost", "daily_return_net"]:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0)
    return out.sort_values("date").reset_index(drop=True)


def normalized_action(value: Any) -> str:
    action = str(value).strip().lower().replace(" ", "_")
    if action in {"long", "short", "hold"}:
        return action
    if action in {"strong_up", "mild_up"}:
        return "long"
    if action in {"strong_down", "mild_down"}:
        return "short"
    return "hold"


def merge_predictions(contexts: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    ctx = contexts.drop_duplicates("sample_id").copy()
    pred = predictions.drop_duplicates("sample_id").copy()
    keep = ["sample_id"]
    for col in ["action", "pred_label", "schema_ok", *PROB_COLS]:
        if col in pred.columns:
            keep.append(col)
    df = ctx.merge(pred[keep], on="sample_id", how="left")
    if "schema_ok" not in df.columns:
        df["schema_ok"] = False
    df["schema_ok"] = df["schema_ok"].fillna(False).astype(bool)
    if "action" in df.columns:
        raw_action = df["action"]
    elif "pred_label" in df.columns:
        raw_action = df["pred_label"]
    else:
        raw_action = pd.Series("hold", index=df.index)
    df["policy_action"] = [normalized_action(value) for value in raw_action]
    df.loc[~df["schema_ok"], "policy_action"] = "hold"
    for col in PROB_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df


def side_confidence(frame: pd.DataFrame, position: pd.Series) -> pd.Series:
    up_prob = frame["p_mild_up"].astype(float) + frame["p_strong_up"].astype(float)
    down_prob = frame["p_mild_down"].astype(float) + frame["p_strong_down"].astype(float)
    return pd.Series(
        np.where(position.astype(float) > 0, up_prob, np.where(position.astype(float) < 0, down_prob, 0.0)),
        index=frame.index,
    )


def build_policy_daily(
    contexts: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    strategy: str,
    threshold: float,
    position_cap: int,
    return_col: str,
    cost_bps: float,
    slippage_bps: float,
    short_borrow_bps: float,
) -> pd.DataFrame:
    ctx = contexts.drop_duplicates("sample_id").copy()
    calendar = calendar_from_contexts(ctx)
    df = merge_predictions(ctx, predictions)
    df["position"] = df["policy_action"].apply(action_to_position).astype(float)
    df["signal_confidence"] = side_confidence(df, df["position"])
    df.loc[(df["position"] > 0) & (df["signal_confidence"] < threshold), "position"] = 0.0
    df.loc[(df["position"] < 0) & (df["signal_confidence"] < threshold), "position"] = 0.0
    return build_daily_from_positions(
        df,
        calendar=calendar,
        strategy=strategy,
        position_cap=position_cap,
        return_col=return_col,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        short_borrow_bps=short_borrow_bps,
    )


def build_technical_daily(
    contexts: pd.DataFrame,
    *,
    strategy: str,
    position_cap: int,
    return_col: str,
    cost_bps: float,
    slippage_bps: float,
    short_borrow_bps: float,
) -> pd.DataFrame:
    ctx = contexts.drop_duplicates("sample_id").copy()
    calendar = calendar_from_contexts(ctx)
    df = ctx.copy()
    if "technical_event_tokens_json" in df.columns:
        df["policy_action"] = df["technical_event_tokens_json"].apply(technical_rule_action)
    else:
        df["policy_action"] = "hold"
    df["position"] = df["policy_action"].apply(action_to_position).astype(float)
    df["signal_confidence"] = 1.0
    return build_daily_from_positions(
        df,
        calendar=calendar,
        strategy=strategy,
        position_cap=position_cap,
        return_col=return_col,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        short_borrow_bps=short_borrow_bps,
    )


def build_daily_from_positions(
    frame: pd.DataFrame,
    *,
    calendar: pd.DataFrame,
    strategy: str,
    position_cap: int,
    return_col: str,
    cost_bps: float,
    slippage_bps: float,
    short_borrow_bps: float,
) -> pd.DataFrame:
    df = frame.copy()
    df = df[df["position"] != 0].copy()
    if df.empty:
        out = add_calendar_zeros(pd.DataFrame(columns=["date"]), calendar)
        out["strategy"] = strategy
        return out
    df["date"] = pd.to_datetime(df["event_date"]).dt.date
    agg = df.groupby(["date", "ticker"]).agg(
        position=("position", "mean"),
        ret=(return_col, "first"),
        signal_confidence=("signal_confidence", "max"),
    ).reset_index()
    agg["position"] = agg["position"].clip(-1, 1)

    def cap_positions(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) > position_cap:
            return group.sort_values(["signal_confidence", "ticker"], ascending=[False, True]).head(position_cap)
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
    by_day["cost"] = (cost_bps + slippage_bps) / 10000.0 * by_day["turnover"].clip(lower=0)
    by_day["borrow_cost"] = short_borrow_bps / 10000.0 * by_day["short_exposure"].clip(lower=0)
    by_day["cost"] = by_day["cost"] + by_day["borrow_cost"]
    by_day["daily_return_net"] = by_day["gross_return"] - by_day["cost"]
    out = add_calendar_zeros(by_day, calendar)
    out["strategy"] = strategy
    return out


def point_stats(daily: pd.DataFrame) -> dict[str, Any]:
    returns = daily["daily_return_net"].astype(float).to_numpy() if len(daily) else np.asarray([], dtype=float)
    return {
        "num_trading_days": int(len(daily)),
        "nonzero_position_days": int((daily["positions"] > 0).sum()) if len(daily) and "positions" in daily else 0,
        "coverage": float((daily["positions"] > 0).mean()) if len(daily) and "positions" in daily else 0.0,
        "mean_daily_return": float(np.mean(returns)) if len(returns) else 0.0,
        "sharpe_annualized": compute_sharpe(returns),
        "sortino_annualized": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "total_turnover": float(daily["turnover"].sum()) if len(daily) and "turnover" in daily else 0.0,
    }


def ci_stats(
    daily: pd.DataFrame,
    *,
    benchmark_daily: pd.DataFrame | None,
    default_daily: pd.DataFrame | None,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> dict[str, Any]:
    returns = daily.sort_values("date")["daily_return_net"].astype(float).to_numpy()
    stats = point_stats(daily)
    sharpe_est, sharpe_low, sharpe_high, sharpe_p = bootstrap_series_stat(
        returns, compute_sharpe, n_bootstrap=n_bootstrap, block_size=block_size, seed=seed
    )
    mean_est, mean_low, mean_high, mean_p = bootstrap_series_stat(
        returns,
        lambda arr: float(np.mean(arr)) if len(arr) else float("nan"),
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        seed=seed + 1,
    )
    stats.update(
        {
            "sharpe_annualized": sharpe_est,
            "sharpe_ci95_low": sharpe_low,
            "sharpe_ci95_high": sharpe_high,
            "sharpe_p_gt_zero": sharpe_p,
            "mean_daily_return": mean_est,
            "mean_daily_return_ci95_low": mean_low,
            "mean_daily_return_ci95_high": mean_high,
            "mean_daily_return_p_gt_zero": mean_p,
            "ci_support_vs_zero": bool(sharpe_low > 0 and mean_low > 0),
        }
    )
    if benchmark_daily is not None:
        bench = benchmark_daily.sort_values("date")["daily_return_net"].astype(float).to_numpy()
        delta_sharpe, delta_sharpe_low, delta_sharpe_high, delta_sharpe_p = paired_block_bootstrap_delta(
            returns, bench, compute_sharpe, n_bootstrap=n_bootstrap, block_size=block_size, seed=seed + 10
        )
        delta_mean, delta_mean_low, delta_mean_high, delta_mean_p = paired_block_bootstrap_delta(
            returns,
            bench,
            lambda arr: float(np.mean(arr)) if len(arr) else float("nan"),
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            seed=seed + 11,
        )
        stats.update(
            {
                "delta_sharpe_vs_technical": delta_sharpe,
                "delta_sharpe_vs_technical_ci95_low": delta_sharpe_low,
                "delta_sharpe_vs_technical_ci95_high": delta_sharpe_high,
                "delta_sharpe_vs_technical_p_gt_zero": delta_sharpe_p,
                "delta_mean_return_vs_technical": delta_mean,
                "delta_mean_return_vs_technical_ci95_low": delta_mean_low,
                "delta_mean_return_vs_technical_ci95_high": delta_mean_high,
                "delta_mean_return_vs_technical_p_gt_zero": delta_mean_p,
                "ci_support_vs_technical": bool(delta_sharpe_low > 0 and delta_mean_low > 0),
            }
        )
    if default_daily is not None:
        default = default_daily.sort_values("date")["daily_return_net"].astype(float).to_numpy()
        delta_sharpe, delta_sharpe_low, delta_sharpe_high, delta_sharpe_p = paired_block_bootstrap_delta(
            returns, default, compute_sharpe, n_bootstrap=n_bootstrap, block_size=block_size, seed=seed + 20
        )
        delta_mean, delta_mean_low, delta_mean_high, delta_mean_p = paired_block_bootstrap_delta(
            returns,
            default,
            lambda arr: float(np.mean(arr)) if len(arr) else float("nan"),
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            seed=seed + 21,
        )
        stats.update(
            {
                "delta_sharpe_vs_default_dpo": delta_sharpe,
                "delta_sharpe_vs_default_dpo_ci95_low": delta_sharpe_low,
                "delta_sharpe_vs_default_dpo_ci95_high": delta_sharpe_high,
                "delta_sharpe_vs_default_dpo_p_gt_zero": delta_sharpe_p,
                "delta_mean_return_vs_default_dpo": delta_mean,
                "delta_mean_return_vs_default_dpo_ci95_low": delta_mean_low,
                "delta_mean_return_vs_default_dpo_ci95_high": delta_mean_high,
                "delta_mean_return_vs_default_dpo_p_gt_zero": delta_mean_p,
            }
        )
    return stats


def select_policy(
    val_contexts: pd.DataFrame,
    val_predictions: pd.DataFrame,
    *,
    thresholds: list[float],
    position_caps: list[int],
    min_val_nonzero_days: int,
    return_col: str,
    cost_bps: float,
    slippage_bps: float,
    short_borrow_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cap in position_caps:
        for threshold in thresholds:
            daily = build_policy_daily(
                val_contexts,
                val_predictions,
                strategy=SELECTED_STRATEGY,
                threshold=float(threshold),
                position_cap=int(cap),
                return_col=return_col,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
                short_borrow_bps=short_borrow_bps,
            )
            stats = point_stats(daily)
            rows.append(
                {
                    "threshold": float(threshold),
                    "position_cap": int(cap),
                    **{f"val_{key}": value for key, value in stats.items()},
                    "eligible_for_selection": bool(stats["nonzero_position_days"] >= min_val_nonzero_days and stats["total_turnover"] > 0),
                }
            )
    grid = pd.DataFrame(rows)
    eligible = grid[grid["eligible_for_selection"].astype(bool)].copy()
    if eligible.empty:
        raise ValueError("no validation policy satisfied the minimum activity gate")
    eligible = eligible.sort_values(
        ["val_sharpe_annualized", "val_mean_daily_return", "val_nonzero_position_days", "position_cap", "threshold"],
        ascending=[False, False, False, True, True],
    )
    selected = eligible.iloc[0].to_dict()
    return grid, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-contexts", default="data/processed/current_v6_validation_contexts.parquet")
    parser.add_argument("--val-predictions", default="outputs/predictions/current_v6_dpo_val_predictions_repaired.parquet")
    parser.add_argument("--test-contexts", default="data/processed/current_v6_prediction_contexts.parquet")
    parser.add_argument("--test-predictions", default="outputs/predictions/current_v6_dpo_predictions_repaired.parquet")
    parser.add_argument("--thresholds", default="0.00:0.90:0.01")
    parser.add_argument("--position-caps", default="1,2,3,5,10,20")
    parser.add_argument("--min-val-nonzero-days", type=int, default=5)
    parser.add_argument("--return-col", default="target_return")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--short-borrow-bps", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-output", default="outputs/tables/15_6_v6_validation_selected_trading_grid.csv")
    parser.add_argument("--summary-output", default="outputs/tables/15_6_v6_validation_selected_trading_summary.csv")
    parser.add_argument("--daily-output", default="outputs/tables/15_6_v6_validation_selected_trading_daily_returns.csv")
    parser.add_argument("--metrics", default="outputs/metrics/15_6_v6_validation_selected_trading_policy.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    required_inputs = [args.val_contexts, args.val_predictions, args.test_contexts, args.test_predictions]
    for path in required_inputs:
        if not Path(path).exists():
            failures.append(f"missing input: {path}")

    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    grid = pd.DataFrame()
    summary = pd.DataFrame()
    daily_long = pd.DataFrame()
    if not failures:
        try:
            thresholds = [float(x) for x in parse_grid(args.thresholds, cast=float)]
            position_caps = [int(x) for x in parse_grid(args.position_caps, cast=int)]
            val_contexts = pd.read_parquet(args.val_contexts)
            val_predictions = pd.read_parquet(args.val_predictions)
            test_contexts = pd.read_parquet(args.test_contexts)
            test_predictions = pd.read_parquet(args.test_predictions)
            grid, selected = select_policy(
                val_contexts,
                val_predictions,
                thresholds=thresholds,
                position_caps=position_caps,
                min_val_nonzero_days=args.min_val_nonzero_days,
                return_col=args.return_col,
                cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
                short_borrow_bps=args.short_borrow_bps,
            )
            selected_threshold = float(selected["threshold"])
            selected_cap = int(selected["position_cap"])
            selected_val_daily = build_policy_daily(
                val_contexts,
                val_predictions,
                strategy=SELECTED_STRATEGY,
                threshold=selected_threshold,
                position_cap=selected_cap,
                return_col=args.return_col,
                cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
                short_borrow_bps=args.short_borrow_bps,
            )
            selected_test_daily = build_policy_daily(
                test_contexts,
                test_predictions,
                strategy=SELECTED_STRATEGY,
                threshold=selected_threshold,
                position_cap=selected_cap,
                return_col=args.return_col,
                cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
                short_borrow_bps=args.short_borrow_bps,
            )
            default_test_daily = build_policy_daily(
                test_contexts,
                test_predictions,
                strategy=DEFAULT_STRATEGY,
                threshold=0.20,
                position_cap=20,
                return_col=args.return_col,
                cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
                short_borrow_bps=args.short_borrow_bps,
            )
            technical_test_daily = build_technical_daily(
                test_contexts,
                strategy=TECHNICAL_STRATEGY,
                position_cap=20,
                return_col=args.return_col,
                cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
                short_borrow_bps=args.short_borrow_bps,
            )
            selected_test_stats = ci_stats(
                selected_test_daily,
                benchmark_daily=technical_test_daily,
                default_daily=default_test_daily,
                n_bootstrap=args.n_bootstrap,
                block_size=args.block_size,
                seed=args.seed,
            )
            default_test_stats = ci_stats(
                default_test_daily,
                benchmark_daily=technical_test_daily,
                default_daily=None,
                n_bootstrap=args.n_bootstrap,
                block_size=args.block_size,
                seed=args.seed + 100,
            )
            technical_test_stats = ci_stats(
                technical_test_daily,
                benchmark_daily=None,
                default_daily=None,
                n_bootstrap=args.n_bootstrap,
                block_size=args.block_size,
                seed=args.seed + 200,
            )
            selected_val_stats = point_stats(selected_val_daily)
            summary_rows = [
                {"split": "validation", "strategy": SELECTED_STRATEGY, "threshold": selected_threshold, "position_cap": selected_cap, **selected_val_stats},
                {"split": "test", "strategy": SELECTED_STRATEGY, "threshold": selected_threshold, "position_cap": selected_cap, **selected_test_stats},
                {"split": "test", "strategy": DEFAULT_STRATEGY, "threshold": 0.20, "position_cap": 20, **default_test_stats},
                {"split": "test", "strategy": TECHNICAL_STRATEGY, "threshold": np.nan, "position_cap": 20, **technical_test_stats},
            ]
            summary = pd.DataFrame(summary_rows)
            daily_parts = []
            for split_name, frame in [
                ("validation", selected_val_daily),
                ("test", selected_test_daily),
                ("test", default_test_daily),
                ("test", technical_test_daily),
            ]:
                part = frame.copy()
                part["split"] = split_name
                daily_parts.append(part)
            daily_long = pd.concat(daily_parts, ignore_index=True)
            alpha_sharpe_ci_support = bool(selected_test_stats["sharpe_ci95_low"] > 0)
            alpha_mean_ci_support = bool(selected_test_stats["mean_daily_return_ci95_low"] > 0)
            alpha_vs_technical_ci_support = bool(
                selected_test_stats["delta_sharpe_vs_technical_ci95_low"] > 0
                and selected_test_stats["delta_mean_return_vs_technical_ci95_low"] > 0
            )
            alpha_paper_level_supported = bool(
                selected_test_stats["num_trading_days"] >= 120
                and alpha_sharpe_ci_support
                and alpha_mean_ci_support
                and alpha_vs_technical_ci_support
            )
            metrics = {
                "pipeline_pass": True,
                "claim_allowed": alpha_paper_level_supported,
                "alpha_paper_level_supported": alpha_paper_level_supported,
                "selection_protocol": "validation_grid_search_then_locked_test_evaluation",
                "selection_uses_test_returns": False,
                "grid_policy_count": int(len(grid)),
                "eligible_policy_count": int(grid["eligible_for_selection"].astype(bool).sum()) if len(grid) else 0,
                "threshold_grid": args.thresholds,
                "position_cap_grid": args.position_caps,
                "selected_threshold": selected_threshold,
                "selected_position_cap": selected_cap,
                "min_val_nonzero_days": int(args.min_val_nonzero_days),
                "val_num_trading_days": int(selected_val_stats["num_trading_days"]),
                "val_nonzero_position_days": int(selected_val_stats["nonzero_position_days"]),
                "val_sharpe_annualized": float(selected_val_stats["sharpe_annualized"]),
                "val_mean_daily_return": float(selected_val_stats["mean_daily_return"]),
                "test_num_trading_days": int(selected_test_stats["num_trading_days"]),
                "test_nonzero_position_days": int(selected_test_stats["nonzero_position_days"]),
                "test_coverage": float(selected_test_stats["coverage"]),
                "test_sharpe_annualized": float(selected_test_stats["sharpe_annualized"]),
                "test_sharpe_ci95_low": float(selected_test_stats["sharpe_ci95_low"]),
                "test_sharpe_ci95_high": float(selected_test_stats["sharpe_ci95_high"]),
                "test_mean_daily_return": float(selected_test_stats["mean_daily_return"]),
                "test_mean_daily_return_ci95_low": float(selected_test_stats["mean_daily_return_ci95_low"]),
                "test_mean_daily_return_ci95_high": float(selected_test_stats["mean_daily_return_ci95_high"]),
                "delta_sharpe_vs_technical": float(selected_test_stats["delta_sharpe_vs_technical"]),
                "delta_sharpe_vs_technical_ci95_low": float(selected_test_stats["delta_sharpe_vs_technical_ci95_low"]),
                "delta_sharpe_vs_technical_ci95_high": float(selected_test_stats["delta_sharpe_vs_technical_ci95_high"]),
                "delta_mean_return_vs_technical": float(selected_test_stats["delta_mean_return_vs_technical"]),
                "delta_mean_return_vs_technical_ci95_low": float(selected_test_stats["delta_mean_return_vs_technical_ci95_low"]),
                "delta_mean_return_vs_technical_ci95_high": float(selected_test_stats["delta_mean_return_vs_technical_ci95_high"]),
                "delta_sharpe_vs_default_dpo": float(selected_test_stats["delta_sharpe_vs_default_dpo"]),
                "delta_mean_return_vs_default_dpo": float(selected_test_stats["delta_mean_return_vs_default_dpo"]),
                "default_dpo_sharpe_annualized": float(default_test_stats["sharpe_annualized"]),
                "technical_rule_sharpe_annualized": float(technical_test_stats["sharpe_annualized"]),
                "alpha_sharpe_ci_support": alpha_sharpe_ci_support,
                "alpha_mean_return_ci_support": alpha_mean_ci_support,
                "alpha_vs_technical_ci_support": alpha_vs_technical_ci_support,
                "bootstrap_repetitions": int(args.n_bootstrap),
                "block_size": int(args.block_size),
                "seed": int(args.seed),
                "cost_bps": float(args.cost_bps),
                "slippage_bps": float(args.slippage_bps),
                "short_borrow_bps": float(args.short_borrow_bps),
                "diagnostic_if_blocked_reason": (
                    "Validation-selected policy cannot open a paper-level alpha claim unless absolute "
                    "Sharpe, absolute mean return, and paired deltas versus Technical_Rule have positive CIs."
                ),
            }
        except Exception as exc:
            failures.append(f"validation-selected trading policy failed: {type(exc).__name__}: {str(exc)[:500]}")

    Path(args.grid_output).parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_output, index=False)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    Path(args.daily_output).parent.mkdir(parents=True, exist_ok=True)
    daily_long.to_csv(args.daily_output, index=False)
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [*required_inputs, args.grid_output, args.summary_output, args.daily_output, args.metrics],
        STEP,
        extra={
            "references": [
                "White 2000 Reality Check data-snooping caution",
                "Bailey and Lopez de Prado Deflated Sharpe Ratio / backtest overfitting caution",
                "Politis and Romano stationary bootstrap / dependent resampling motivation",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        required_inputs,
        [args.grid_output, args.summary_output, args.daily_output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
