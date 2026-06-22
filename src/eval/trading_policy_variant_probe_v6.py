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
from src.repro.run_v6_statistical_tests import bootstrap_series_stat, paired_block_bootstrap_delta
from src.reward.build_flow_decision_dataset_v6 import technical_rule_action
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "15_5_TRADING_POLICY_VARIANT_PROBE_V6"


def sortino(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    return float(np.sqrt(252) * np.mean(returns) / downside_std) if downside_std > 0 else 0.0


def max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / np.where(running_max == 0, 1.0, running_max)
    return float(np.max(drawdown)) if len(drawdown) else 0.0


def calendar_from_contexts(contexts: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"date": sorted(pd.to_datetime(contexts["event_date"]).dt.date.dropna().unique())})


def add_calendar_zeros(day_df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    out = calendar.merge(day_df, on="date", how="left")
    for col in ["gross_return", "positions", "turnover", "short_exposure", "cost", "borrow_cost", "daily_return_net"]:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0)
    return out.sort_values("date").reset_index(drop=True)


def side_confidence(frame: pd.DataFrame, position: pd.Series) -> pd.Series:
    required = {"p_mild_up", "p_strong_up", "p_mild_down", "p_strong_down"}
    if not required <= set(frame.columns):
        return pd.Series(1.0, index=frame.index)
    up_prob = frame["p_mild_up"].astype(float) + frame["p_strong_up"].astype(float)
    down_prob = frame["p_mild_down"].astype(float) + frame["p_strong_down"].astype(float)
    return pd.Series(
        np.where(position.astype(float) > 0, up_prob, np.where(position.astype(float) < 0, down_prob, 0.0)),
        index=frame.index,
    )


def build_daily(
    contexts: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_col: str,
    strategy: str,
    position_cap: int,
    cost_bps: float,
    slippage_bps: float,
    short_borrow_bps: float,
    threshold: float | None = None,
) -> pd.DataFrame:
    ctx = contexts.drop_duplicates("sample_id").copy()
    calendar = calendar_from_contexts(ctx)
    lab = labels.copy().drop_duplicates("sample_id")
    keep = ["sample_id", label_col]
    for col in ["p_mild_up", "p_strong_up", "p_mild_down", "p_strong_down"]:
        if col in lab.columns:
            keep.append(col)
    df = ctx.merge(lab[keep].rename(columns={label_col: "pred_label"}), on="sample_id", how="inner")
    df["position"] = df["pred_label"].apply(action_to_position).astype(float)
    if threshold is not None:
        conf = side_confidence(df, df["position"])
        df.loc[(df["position"] > 0) & (conf < threshold), "position"] = 0.0
        df.loc[(df["position"] < 0) & (conf < threshold), "position"] = 0.0
    df = df[df["position"] != 0].copy()
    if df.empty:
        out = add_calendar_zeros(pd.DataFrame(columns=["date"]), calendar)
        out["strategy"] = strategy
        return out
    df["date"] = pd.to_datetime(df["event_date"]).dt.date
    agg = df.groupby(["date", "ticker"]).agg(position=("position", "mean"), ret=("target_return", "first")).reset_index()
    agg["position"] = agg["position"].clip(-1, 1)

    def cap_positions(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) > position_cap:
            return group.sort_values("ticker").head(position_cap)
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


def strategy_inputs(
    contexts: pd.DataFrame,
    dpo: pd.DataFrame,
    hybrid: pd.DataFrame,
    stacked: pd.DataFrame,
    supervised: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, str, float | None]]:
    tech = contexts[["sample_id", "technical_event_tokens_json"]].copy()
    tech["label"] = tech["technical_event_tokens_json"].apply(technical_rule_action)
    out: dict[str, tuple[pd.DataFrame, str, float | None]] = {
        "Technical_Rule": (tech[["sample_id", "label"]], "label", None),
    }
    if not dpo.empty:
        dpo_valid = contexts[["sample_id"]].merge(dpo, on="sample_id", how="left")
        dpo_valid["label"] = np.where(
            dpo_valid.get("schema_ok", pd.Series(False, index=dpo_valid.index)).fillna(False).astype(bool),
            dpo_valid.get("action", dpo_valid.get("pred_label", "neutral")),
            "neutral",
        )
        prob_cols = [col for col in ["p_mild_up", "p_strong_up", "p_mild_down", "p_strong_down"] if col in dpo_valid.columns]
        out["Qwen_DPO_V6_Official_Action"] = (dpo_valid[["sample_id", "label", *prob_cols]], "label", 0.20)
        dpo_label = contexts[["sample_id"]].merge(dpo, on="sample_id", how="left")
        dpo_label["label"] = np.where(
            dpo_label.get("schema_ok", pd.Series(False, index=dpo_label.index)).fillna(False).astype(bool),
            dpo_label.get("pred_label", "neutral"),
            "neutral",
        )
        out["Qwen_DPO_V6_Label"] = (dpo_label[["sample_id", "label"]], "label", None)
    if not hybrid.empty and {"sample_id", "hybrid_pred"} <= set(hybrid.columns):
        out["Validation_Calibrated_Hybrid"] = (
            hybrid[["sample_id", "hybrid_pred"]].rename(columns={"hybrid_pred": "label"}),
            "label",
            None,
        )
    if not stacked.empty and {"sample_id", "stacked_pred"} <= set(stacked.columns):
        out["Validation_Stacked_Logistic"] = (
            stacked[["sample_id", "stacked_pred"]].rename(columns={"stacked_pred": "label"}),
            "label",
            None,
        )
    if not supervised.empty and {"sample_id", "supervised_logreg_pred"} <= set(supervised.columns):
        out["Supervised_LogReg_TFIDF"] = (
            supervised[["sample_id", "supervised_logreg_pred"]].rename(columns={"supervised_logreg_pred": "label"}),
            "label",
            None,
        )
    return out


def summary_rows(
    daily_long: pd.DataFrame,
    *,
    benchmark: str,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    benchmark_returns = daily_long[daily_long["strategy"].eq(benchmark)].sort_values("date")["daily_return_net"].astype(float).to_numpy()
    for idx, (strategy, frame) in enumerate(daily_long.groupby("strategy")):
        frame = frame.sort_values("date")
        returns = frame["daily_return_net"].astype(float).to_numpy()
        sharpe_est, sharpe_low, sharpe_high, sharpe_p = bootstrap_series_stat(
            returns, compute_sharpe, n_bootstrap=n_bootstrap, block_size=block_size, seed=seed + idx
        )
        mean_est, mean_low, mean_high, mean_p = bootstrap_series_stat(
            returns, lambda arr: float(np.mean(arr)), n_bootstrap=n_bootstrap, block_size=block_size, seed=seed + 100 + idx
        )
        delta_sharpe, delta_sharpe_low, delta_sharpe_high, delta_sharpe_p = paired_block_bootstrap_delta(
            returns,
            benchmark_returns,
            compute_sharpe,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            seed=seed + 200 + idx,
        )
        delta_mean, delta_mean_low, delta_mean_high, delta_mean_p = paired_block_bootstrap_delta(
            returns,
            benchmark_returns,
            lambda arr: float(np.mean(arr)),
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            seed=seed + 300 + idx,
        )
        rows.append(
            {
                "strategy": strategy,
                "num_trading_days": int(len(frame)),
                "nonzero_position_days": int((frame["positions"] > 0).sum()),
                "coverage": float((frame["positions"] > 0).mean()) if len(frame) else 0.0,
                "mean_daily_return": mean_est,
                "mean_daily_return_ci95_low": mean_low,
                "mean_daily_return_ci95_high": mean_high,
                "mean_daily_return_p_gt_zero": mean_p,
                "sharpe_annualized": sharpe_est,
                "sharpe_ci95_low": sharpe_low,
                "sharpe_ci95_high": sharpe_high,
                "sharpe_p_gt_zero": sharpe_p,
                "sortino_annualized": sortino(returns),
                "max_drawdown": max_drawdown(returns),
                "total_turnover": float(frame["turnover"].sum()) if len(frame) else 0.0,
                "delta_sharpe_vs_technical": delta_sharpe,
                "delta_sharpe_vs_technical_ci95_low": delta_sharpe_low,
                "delta_sharpe_vs_technical_ci95_high": delta_sharpe_high,
                "delta_sharpe_vs_technical_p_gt_zero": delta_sharpe_p,
                "delta_mean_return_vs_technical": delta_mean,
                "delta_mean_return_vs_technical_ci95_low": delta_mean_low,
                "delta_mean_return_vs_technical_ci95_high": delta_mean_high,
                "delta_mean_return_vs_technical_p_gt_zero": delta_mean_p,
                "ci_support_vs_zero": bool(sharpe_low > 0 and mean_low > 0),
                "ci_support_vs_technical": bool(delta_sharpe_low > 0 and delta_mean_low > 0),
            }
        )
    return rows


def read_parquet_or_empty(path: str | None) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--dpo-predictions", required=True)
    parser.add_argument("--hybrid-predictions", default="outputs/predictions/current_v6_validation_calibrated_hybrid_predictions.parquet")
    parser.add_argument("--stacked-predictions", default="outputs/predictions/current_v6_validation_stacked_forecast_predictions.parquet")
    parser.add_argument("--supervised-predictions", default="outputs/predictions/current_v6_supervised_signal_ceiling_predictions.parquet")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--short-borrow-bps", type=float, default=1.0)
    parser.add_argument("--position-cap", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", default="outputs/tables/15_5_v6_trading_policy_variant_summary.csv")
    parser.add_argument("--daily-output", default="outputs/tables/15_5_v6_trading_policy_variant_daily_returns.csv")
    parser.add_argument("--metrics", default="outputs/metrics/15_5_v6_trading_policy_variant_probe.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    required_inputs = [args.contexts, args.dpo_predictions]
    for path in required_inputs:
        if not Path(path).exists():
            failures.append(f"missing input: {path}")
    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    summary = pd.DataFrame()
    daily_long = pd.DataFrame()
    if not failures:
        try:
            contexts = pd.read_parquet(args.contexts)
            if "split" in contexts.columns:
                contexts = contexts[contexts["split"].eq("test")].copy()
            dpo = pd.read_parquet(args.dpo_predictions)
            hybrid = read_parquet_or_empty(args.hybrid_predictions)
            stacked = read_parquet_or_empty(args.stacked_predictions)
            supervised = read_parquet_or_empty(args.supervised_predictions)
            strategies = strategy_inputs(contexts, dpo, hybrid, stacked, supervised)
            daily_frames = []
            for strategy, (labels, label_col, threshold) in strategies.items():
                day = build_daily(
                    contexts,
                    labels,
                    label_col=label_col,
                    strategy=strategy,
                    position_cap=args.position_cap,
                    cost_bps=args.cost_bps,
                    slippage_bps=args.slippage_bps,
                    short_borrow_bps=args.short_borrow_bps,
                    threshold=threshold,
                )
                daily_frames.append(day)
            daily_long = pd.concat(daily_frames, ignore_index=True)
            summary = pd.DataFrame(
                summary_rows(
                    daily_long,
                    benchmark="Technical_Rule",
                    n_bootstrap=args.n_bootstrap,
                    block_size=args.block_size,
                    seed=args.seed,
                )
            ).sort_values("sharpe_annualized", ascending=False)
            best = summary.iloc[0].to_dict() if len(summary) else {}
            dpo_row = summary[summary["strategy"].eq("Qwen_DPO_V6_Official_Action")]
            metrics.update(
                {
                    "pipeline_pass": True,
                    "claim_allowed": False,
                    "strategy_count": int(len(summary)),
                    "num_trading_days": int(daily_long["date"].nunique()) if len(daily_long) else 0,
                    "best_strategy_by_sharpe": best.get("strategy"),
                    "best_strategy_sharpe": float(best.get("sharpe_annualized", 0.0)) if best else 0.0,
                    "best_strategy_mean_daily_return": float(best.get("mean_daily_return", 0.0)) if best else 0.0,
                    "best_strategy_ci_support_vs_zero": bool(best.get("ci_support_vs_zero", False)) if best else False,
                    "best_strategy_ci_support_vs_technical": bool(best.get("ci_support_vs_technical", False)) if best else False,
                    "dpo_official_sharpe": float(dpo_row.iloc[0]["sharpe_annualized"]) if len(dpo_row) else 0.0,
                    "dpo_official_mean_daily_return": float(dpo_row.iloc[0]["mean_daily_return"]) if len(dpo_row) else 0.0,
                    "multiple_testing_warning": True,
                    "diagnostic_only_reason": (
                        "Variants are compared after observing the same held-out test period; "
                        "positive variants require preregistered validation and fresh out-of-sample confirmation."
                    ),
                    "bootstrap_repetitions": int(args.n_bootstrap),
                    "block_size": int(args.block_size),
                    "seed": int(args.seed),
                }
            )
        except Exception as exc:
            failures.append(f"trading policy variant probe failed: {type(exc).__name__}: {str(exc)[:500]}")

    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    Path(args.daily_output).parent.mkdir(parents=True, exist_ok=True)
    daily_long.to_csv(args.daily_output, index=False)
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    manifest_inputs = [
        args.contexts,
        args.dpo_predictions,
        args.hybrid_predictions,
        args.stacked_predictions,
        args.supervised_predictions,
        args.summary_output,
        args.daily_output,
        args.metrics,
    ]
    write_manifest(
        args.manifest,
        manifest_inputs,
        STEP,
        extra={
            "references": [
                "Lo 2002 Sharpe ratio inference",
                "White 2000 Reality Check for data snooping",
                "Bailey and Lopez de Prado Deflated Sharpe Ratio / backtest overfitting",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        required_inputs,
        [args.summary_output, args.daily_output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
