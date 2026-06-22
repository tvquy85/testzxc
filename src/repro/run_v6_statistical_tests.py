from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

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
from src.eval.backtest_daily_portfolio_v3 import action_to_position, compute_sharpe
from src.reward.build_flow_decision_dataset_v6 import technical_rule_action
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "18_5_STATISTICAL_TESTS_AND_CI_V6"


def percentile_interval(values: list[float] | np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(arr, alpha / 2.0)),
        float(np.quantile(arr, 1.0 - alpha / 2.0)),
    )


def moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=int)
    block_size = max(1, min(int(block_size), n))
    starts = rng.integers(0, n, size=int(math.ceil(n / block_size)))
    chunks = [(start + np.arange(block_size)) % n for start in starts]
    return np.concatenate(chunks)[:n]


def bootstrap_series_stat(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    estimate = float(stat_fn(arr)) if len(arr) else float("nan")
    rng = np.random.default_rng(seed)
    boot = [
        float(stat_fn(arr[moving_block_indices(len(arr), block_size, rng)]))
        for _ in range(max(1, n_bootstrap))
    ]
    low, high = percentile_interval(boot)
    p_greater_zero = float((1 + np.sum(np.asarray(boot) <= 0.0)) / (len(boot) + 1))
    return estimate, low, high, p_greater_zero


def paired_block_bootstrap_delta(
    candidate: np.ndarray,
    benchmark: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> tuple[float, float, float, float]:
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    estimate = float(stat_fn(a) - stat_fn(b)) if n else float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(max(1, n_bootstrap)):
        idx = moving_block_indices(n, block_size, rng)
        boot.append(float(stat_fn(a[idx]) - stat_fn(b[idx])))
    low, high = percentile_interval(boot)
    p_greater_zero = float((1 + np.sum(np.asarray(boot) <= 0.0)) / (len(boot) + 1))
    return estimate, low, high, p_greater_zero


def paired_sign_flip_pvalue_greater(diff: np.ndarray, *, n_permutations: int, seed: int) -> float:
    arr = np.asarray(diff, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    observed = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(max(1, n_permutations)):
        signs = rng.choice([-1.0, 1.0], size=len(arr))
        if float(np.mean(signs * arr)) >= observed:
            count += 1
    return float((count + 1) / (max(1, n_permutations) + 1))


def mcnemar_one_sided_pvalue(candidate_correct: np.ndarray, benchmark_correct: np.ndarray) -> tuple[float, int, int]:
    cand = np.asarray(candidate_correct, dtype=bool)
    bench = np.asarray(benchmark_correct, dtype=bool)
    n = min(len(cand), len(bench))
    cand = cand[:n]
    bench = bench[:n]
    cand_only = int(np.sum(cand & ~bench))
    bench_only = int(np.sum(~cand & bench))
    discordant = cand_only + bench_only
    if discordant == 0:
        return 1.0, cand_only, bench_only
    tail = sum(math.comb(discordant, k) for k in range(cand_only, discordant + 1))
    return float(tail / (2**discordant)), cand_only, bench_only


def metric_value(y_true: list[str], y_pred: list[str], metric: str) -> float:
    if not y_true:
        return float("nan")
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    raise ValueError(f"unknown metric: {metric}")


def paired_row_bootstrap_delta(
    y_true: list[str],
    candidate: list[str],
    benchmark: list[str],
    metric: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, float]:
    n = min(len(y_true), len(candidate), len(benchmark))
    y = np.asarray(y_true[:n], dtype=object)
    a = np.asarray(candidate[:n], dtype=object)
    b = np.asarray(benchmark[:n], dtype=object)
    estimate = float(metric_value(y.tolist(), a.tolist(), metric) - metric_value(y.tolist(), b.tolist(), metric))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(max(1, n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boot.append(float(metric_value(y[idx].tolist(), a[idx].tolist(), metric) - metric_value(y[idx].tolist(), b[idx].tolist(), metric)))
    low, high = percentile_interval(boot)
    p_greater_zero = float((1 + np.sum(np.asarray(boot) <= 0.0)) / (len(boot) + 1))
    return estimate, low, high, p_greater_zero


def add_calendar_zeros(day_df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    out = calendar.merge(day_df, on="date", how="left")
    for col in ["gross_return", "positions", "turnover", "short_exposure", "cost", "borrow_cost", "daily_return_net"]:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0)
    return out.sort_values("date").reset_index(drop=True)


def build_technical_rule_daily(
    contexts: pd.DataFrame,
    *,
    return_col: str,
    max_positions_per_day: int,
    cost_bps: float,
    slippage_bps: float,
    short_borrow_bps: float,
) -> pd.DataFrame:
    source = contexts.copy()
    if "split" in source.columns:
        source = source[source["split"].eq("test")].copy()
    source["date"] = pd.to_datetime(source["event_date"]).dt.date
    calendar = pd.DataFrame({"date": sorted(source["date"].dropna().unique())})
    if "technical_event_tokens_json" not in source.columns or source.empty:
        return add_calendar_zeros(pd.DataFrame(columns=["date"]), calendar)

    rule = source.copy()
    rule["technical_rule_action"] = rule["technical_event_tokens_json"].apply(technical_rule_action)
    rule["position"] = rule["technical_rule_action"].apply(action_to_position).astype(float)
    rule = rule[rule["position"] != 0].copy()
    if rule.empty:
        return add_calendar_zeros(pd.DataFrame(columns=["date"]), calendar)

    agg = rule.groupby(["date", "ticker"]).agg(
        position=("position", "mean"),
        ret=(return_col, "first"),
    ).reset_index()
    agg["position"] = agg["position"].clip(-1, 1)

    def cap_positions(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) > max_positions_per_day:
            return group.sort_values(["ticker"], ascending=[True]).head(max_positions_per_day)
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
    return add_calendar_zeros(by_day, calendar)


def prediction_frame(contexts: pd.DataFrame, predictions: pd.DataFrame, label_col: str) -> pd.DataFrame:
    pred = predictions.copy()
    if "schema_ok" in pred.columns:
        pred = pred[pred["schema_ok"].astype(bool)].copy()
    cols = ["sample_id", "pred_label"]
    if "action" in pred.columns:
        cols.append("action")
    ctx = contexts.copy()
    if "split" in ctx.columns:
        ctx = ctx[ctx["split"].eq("test")].copy()
    return ctx.merge(pred[cols], on="sample_id", how="inner")


def forecast_prediction_lists(
    contexts: pd.DataFrame,
    dpo: pd.DataFrame,
    rwsft: pd.DataFrame,
    *,
    label_col: str,
) -> dict[str, list[str]]:
    dpo_frame = prediction_frame(contexts, dpo, label_col)
    rwsft_frame = prediction_frame(contexts, rwsft, label_col)
    shared = sorted(set(dpo_frame["sample_id"].astype(str)) & set(rwsft_frame["sample_id"].astype(str)))
    dpo_frame = dpo_frame[dpo_frame["sample_id"].astype(str).isin(shared)].drop_duplicates("sample_id").copy()
    rwsft_frame = rwsft_frame[rwsft_frame["sample_id"].astype(str).isin(shared)].drop_duplicates("sample_id").copy()
    base = dpo_frame.merge(
        rwsft_frame[["sample_id", "pred_label"]].rename(columns={"pred_label": "rwsft_pred_label"}),
        on="sample_id",
        how="inner",
    )
    y_true = base[label_col].map(normalize_label).tolist()
    dpo_pred = base["pred_label"].map(normalize_label).tolist()
    rwsft_pred = base["rwsft_pred_label"].map(normalize_label).tolist()
    technical_pred = [label_from_score(score) for score in base.apply(technical_score, axis=1)]
    return {
        "y_true": y_true,
        "dpo": dpo_pred,
        "rwsft": rwsft_pred,
        "technical_rule": technical_pred,
        "sample_ids": base["sample_id"].astype(str).tolist(),
    }


def test_row(
    *,
    test_id: str,
    family: str,
    comparison: str,
    metric: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
    p_value: float,
    method: str,
    n: int,
    block_size: int | None,
    notes: str,
) -> dict[str, Any]:
    support = bool(np.isfinite(estimate) and estimate > 0 and np.isfinite(ci_low) and ci_low > 0)
    return {
        "test_id": test_id,
        "family": family,
        "comparison": comparison,
        "metric": metric,
        "estimate": float(estimate),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value_one_sided_positive": float(p_value) if np.isfinite(p_value) else np.nan,
        "method": method,
        "n": int(n),
        "block_size": int(block_size) if block_size is not None else "",
        "claim_support": support,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-returns", default="outputs/tables/15_v6_daily_returns.csv")
    parser.add_argument("--contexts", default="data/processed/current_v6_prediction_contexts.parquet")
    parser.add_argument("--dpo-predictions", default="outputs/predictions/current_v6_dpo_predictions.parquet")
    parser.add_argument("--rwsft-predictions", default="outputs/predictions/current_v6_rwsft_predictions.parquet")
    parser.add_argument("--backtest-metrics", default="outputs/metrics/15_v6_backtest_track_baseline.json")
    parser.add_argument("--output", default="outputs/tables/19_v6_statistical_tests.csv")
    parser.add_argument("--daily-comparison-output", default="outputs/tables/19_v6_backtest_daily_comparison.csv")
    parser.add_argument("--metrics", default="outputs/metrics/19_v6_statistical_tests.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--label-col", default="target_label_5")
    parser.add_argument("--return-col", default="target_return")
    parser.add_argument("--max-positions-per-day", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    failures: list[str] = []
    required_inputs = [args.daily_returns, args.contexts, args.dpo_predictions, args.rwsft_predictions, args.backtest_metrics]
    for path in required_inputs:
        if not Path(path).exists():
            failures.append(f"missing input: {path}")

    rows: list[dict[str, Any]] = []
    daily_comparison = pd.DataFrame()
    metrics: dict[str, Any] = {"pipeline_pass": False}
    if not failures:
        daily = pd.read_csv(args.daily_returns)
        contexts = pd.read_parquet(args.contexts)
        dpo = pd.read_parquet(args.dpo_predictions)
        rwsft = pd.read_parquet(args.rwsft_predictions)
        backtest_metrics = json.loads(Path(args.backtest_metrics).read_text(encoding="utf-8"))
        for col in ["date", "daily_return_net"]:
            if col not in daily.columns:
                failures.append(f"daily returns missing column: {col}")
        if args.label_col not in contexts.columns:
            failures.append(f"contexts missing label column: {args.label_col}")

    if not failures:
        cost_bps = float(backtest_metrics.get("cost_bps", 5.0))
        slippage_bps = float(backtest_metrics.get("slippage_bps", 2.0))
        short_borrow_bps = float(backtest_metrics.get("short_borrow_bps", 1.0))
        tech_daily = build_technical_rule_daily(
            contexts,
            return_col=args.return_col,
            max_positions_per_day=args.max_positions_per_day,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            short_borrow_bps=short_borrow_bps,
        )
        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        tech_daily["date"] = pd.to_datetime(tech_daily["date"]).dt.date
        daily_comparison = daily[["date", "daily_return_net"]].rename(columns={"daily_return_net": "dpo_daily_return_net"}).merge(
            tech_daily[["date", "daily_return_net"]].rename(columns={"daily_return_net": "technical_rule_daily_return_net"}),
            on="date",
            how="inner",
        )
        dpo_ret = daily_comparison["dpo_daily_return_net"].astype(float).to_numpy()
        tech_ret = daily_comparison["technical_rule_daily_return_net"].astype(float).to_numpy()

        for test_id, series, metric_name, stat_fn, note, seed_offset in [
            ("dpo_sharpe_ci", dpo_ret, "sharpe_annualized", compute_sharpe, "Moving-block bootstrap CI for DPO daily net returns.", 0),
            ("dpo_mean_return_ci", dpo_ret, "mean_daily_return", lambda x: float(np.mean(x)) if len(x) else float("nan"), "Moving-block bootstrap CI for DPO mean daily net return.", 1),
        ]:
            est, low, high, pval = bootstrap_series_stat(
                series,
                stat_fn,
                n_bootstrap=args.n_bootstrap,
                block_size=args.block_size,
                seed=args.seed + seed_offset,
            )
            rows.append(
                test_row(
                    test_id=test_id,
                    family="backtest_ci",
                    comparison="Qwen_DPO_V6",
                    metric=metric_name,
                    estimate=est,
                    ci_low=low,
                    ci_high=high,
                    p_value=pval,
                    method="moving_block_bootstrap",
                    n=len(series),
                    block_size=args.block_size,
                    notes=note,
                )
            )

        for test_id, metric_name, stat_fn, seed_offset in [
            ("dpo_vs_technical_delta_sharpe_ci", "delta_sharpe_annualized", compute_sharpe, 10),
            ("dpo_vs_technical_delta_mean_return_ci", "delta_mean_daily_return", lambda x: float(np.mean(x)) if len(x) else float("nan"), 11),
        ]:
            est, low, high, pval = paired_block_bootstrap_delta(
                dpo_ret,
                tech_ret,
                stat_fn,
                n_bootstrap=args.n_bootstrap,
                block_size=args.block_size,
                seed=args.seed + seed_offset,
            )
            if metric_name == "delta_mean_daily_return":
                pval = paired_sign_flip_pvalue_greater(
                    dpo_ret - tech_ret,
                    n_permutations=args.n_permutations,
                    seed=args.seed + seed_offset + 100,
                )
            rows.append(
                test_row(
                    test_id=test_id,
                    family="backtest_comparison",
                    comparison="Qwen_DPO_V6 - Technical_Rule",
                    metric=metric_name,
                    estimate=est,
                    ci_low=low,
                    ci_high=high,
                    p_value=pval,
                    method="paired_moving_block_bootstrap",
                    n=len(dpo_ret),
                    block_size=args.block_size,
                    notes="Paired time-series bootstrap on the shared V6 daily-return calendar.",
                )
            )

        preds = forecast_prediction_lists(contexts, dpo, rwsft, label_col=args.label_col)
        y_true = preds["y_true"]
        forecast_pairs = [
            ("dpo_vs_technical", "Qwen_DPO_V6 - Technical_Rule", preds["dpo"], preds["technical_rule"]),
            ("dpo_vs_rwsft", "Qwen_DPO_V6 - Qwen_RWSFT_V6", preds["dpo"], preds["rwsft"]),
        ]
        for prefix, comparison, cand, bench in forecast_pairs:
            for metric in ["accuracy", "macro_f1", "mcc"]:
                est, low, high, pval = paired_row_bootstrap_delta(
                    y_true,
                    cand,
                    bench,
                    metric,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + len(rows),
                )
                rows.append(
                    test_row(
                        test_id=f"{prefix}_{metric}_delta_ci",
                        family="forecast_comparison",
                        comparison=comparison,
                        metric=f"delta_{metric}",
                        estimate=est,
                        ci_low=low,
                        ci_high=high,
                        p_value=pval,
                        method="paired_row_bootstrap",
                        n=len(y_true),
                        block_size=None,
                        notes="Paired bootstrap on shared schema-valid prediction rows.",
                    )
                )
            cand_correct = np.asarray(cand, dtype=object) == np.asarray(y_true, dtype=object)
            bench_correct = np.asarray(bench, dtype=object) == np.asarray(y_true, dtype=object)
            pval, cand_only, bench_only = mcnemar_one_sided_pvalue(cand_correct, bench_correct)
            estimate = float(np.mean(cand_correct) - np.mean(bench_correct)) if len(y_true) else float("nan")
            rows.append(
                {
                    **test_row(
                        test_id=f"{prefix}_mcnemar_accuracy",
                        family="forecast_comparison",
                        comparison=comparison,
                        metric="accuracy_discordance",
                        estimate=estimate,
                        ci_low=float("nan"),
                        ci_high=float("nan"),
                        p_value=pval,
                        method="mcnemar_exact_one_sided",
                        n=len(y_true),
                        block_size=None,
                        notes="Exact one-sided McNemar binomial test on discordant correctness pairs.",
                    ),
                    "claim_support": bool(estimate > 0 and pval < 0.05),
                    "candidate_only_correct": cand_only,
                    "benchmark_only_correct": bench_only,
                }
            )

        table = pd.DataFrame(rows)
        required_tests = {
            "dpo_sharpe_ci",
            "dpo_mean_return_ci",
            "dpo_vs_technical_delta_sharpe_ci",
            "dpo_vs_technical_delta_mean_return_ci",
            "dpo_vs_technical_macro_f1_delta_ci",
            "dpo_vs_technical_mcc_delta_ci",
            "dpo_vs_rwsft_macro_f1_delta_ci",
            "dpo_vs_rwsft_mcc_delta_ci",
        }
        present_tests = set(table["test_id"].astype(str))
        missing_tests = sorted(required_tests - present_tests)
        if missing_tests:
            failures.append(f"missing required statistical tests: {missing_tests}")

        lookup = table.set_index("test_id").to_dict(orient="index")
        metrics = {
            "test_count": int(len(table)),
            "required_tests_present": not missing_tests,
            "confidence_interval_available": bool(not missing_tests and table["ci95_low"].notna().sum() >= len(required_tests)),
            "daily_return_rows": int(len(daily_comparison)),
            "forecast_rows": int(len(y_true)),
            "bootstrap_method": "moving_block_bootstrap_for_daily_returns; paired_row_bootstrap_for_forecast_metrics",
            "bootstrap_repetitions": int(args.n_bootstrap),
            "permutation_repetitions": int(args.n_permutations),
            "block_size": int(args.block_size),
            "seed": int(args.seed),
            "dpo_sharpe": float(lookup["dpo_sharpe_ci"]["estimate"]),
            "dpo_sharpe_ci95_low": float(lookup["dpo_sharpe_ci"]["ci95_low"]),
            "dpo_sharpe_ci95_high": float(lookup["dpo_sharpe_ci"]["ci95_high"]),
            "dpo_mean_daily_return": float(lookup["dpo_mean_return_ci"]["estimate"]),
            "dpo_mean_daily_return_ci95_low": float(lookup["dpo_mean_return_ci"]["ci95_low"]),
            "dpo_mean_daily_return_ci95_high": float(lookup["dpo_mean_return_ci"]["ci95_high"]),
            "delta_sharpe_vs_technical_rule": float(lookup["dpo_vs_technical_delta_sharpe_ci"]["estimate"]),
            "delta_sharpe_vs_technical_rule_ci95_low": float(lookup["dpo_vs_technical_delta_sharpe_ci"]["ci95_low"]),
            "delta_sharpe_vs_technical_rule_ci95_high": float(lookup["dpo_vs_technical_delta_sharpe_ci"]["ci95_high"]),
            "delta_mean_return_vs_technical_rule": float(lookup["dpo_vs_technical_delta_mean_return_ci"]["estimate"]),
            "delta_mean_return_vs_technical_rule_ci95_low": float(lookup["dpo_vs_technical_delta_mean_return_ci"]["ci95_low"]),
            "delta_mean_return_vs_technical_rule_ci95_high": float(lookup["dpo_vs_technical_delta_mean_return_ci"]["ci95_high"]),
            "alpha_sharpe_ci_support": bool(lookup["dpo_sharpe_ci"]["ci95_low"] > 0),
            "alpha_mean_return_ci_support": bool(lookup["dpo_mean_return_ci"]["ci95_low"] > 0),
            "alpha_vs_technical_ci_support": bool(lookup["dpo_vs_technical_delta_mean_return_ci"]["ci95_low"] > 0),
            "forecast_dpo_beats_technical_ci_support": bool(
                lookup["dpo_vs_technical_macro_f1_delta_ci"]["ci95_low"] > 0
                and lookup["dpo_vs_technical_mcc_delta_ci"]["ci95_low"] > 0
            ),
            "forecast_dpo_beats_rwsft_ci_support": bool(
                lookup["dpo_vs_rwsft_macro_f1_delta_ci"]["ci95_low"] > 0
                and lookup["dpo_vs_rwsft_mcc_delta_ci"]["ci95_low"] > 0
            ),
            "pipeline_pass": not failures,
            "claim_allowed": False,
        }
        metrics["alpha_paper_level_supported"] = bool(
            metrics["alpha_sharpe_ci_support"]
            and metrics["alpha_mean_return_ci_support"]
            and metrics["alpha_vs_technical_ci_support"]
        )
    else:
        table = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    Path(args.daily_comparison_output).parent.mkdir(parents=True, exist_ok=True)
    daily_comparison.to_csv(args.daily_comparison_output, index=False)
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [*required_inputs, args.output, args.daily_comparison_output, args.metrics],
        STEP,
        extra={
            "references": [
                "Lo 2002 Sharpe-ratio inference and serial correlation caution",
                "White 2000 Reality Check data-snooping caution",
                "Hansen 2005 Superior Predictive Ability test motivation",
                "Diebold-Mariano 1995 paired forecast-accuracy testing motivation",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        required_inputs,
        [args.output, args.daily_comparison_output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
