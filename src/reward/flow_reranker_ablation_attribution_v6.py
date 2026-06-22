from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reward.evaluate_flow_vs_proxy_v4 import preference_pair_accuracy, spearman, top_decile_utility
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "11_7_FLOW_RERANKER_ABLATION_ATTRIBUTION_V6"

FEATURE_SETS: dict[str, list[str]] = {
    "full": [
        "flow_reward_score",
        "proxy_average_reward",
        "single_best_judge_reward",
        "news_grounding_score",
        "technical_grounding_score",
        "unsupported_news_claim_rate",
    ],
    "no_flow": [
        "proxy_average_reward",
        "single_best_judge_reward",
        "news_grounding_score",
        "technical_grounding_score",
        "unsupported_news_claim_rate",
    ],
    "only_flow": ["flow_reward_score"],
    "no_proxy": [
        "flow_reward_score",
        "single_best_judge_reward",
        "news_grounding_score",
        "technical_grounding_score",
        "unsupported_news_claim_rate",
    ],
    "no_single_judge": [
        "flow_reward_score",
        "proxy_average_reward",
        "news_grounding_score",
        "technical_grounding_score",
        "unsupported_news_claim_rate",
    ],
    "scores_only": ["flow_reward_score", "proxy_average_reward", "single_best_judge_reward"],
    "grounding_only": [
        "news_grounding_score",
        "technical_grounding_score",
        "unsupported_news_claim_rate",
    ],
}

REQUIRED_ATTRIBUTION_SETS = ["full", "no_flow", "only_flow"]


def feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df[feature_cols].astype(float).copy()
    if "unsupported_news_claim_rate" in out.columns:
        out["neg_unsupported_news_claim_rate"] = -out["unsupported_news_claim_rate"]
    return out


def method_metrics(df: pd.DataFrame, score_col: str) -> dict[str, float | None]:
    return {
        "rank": spearman(df[score_col], df["raw_realized_utility"]),
        "pair": preference_pair_accuracy(df, score_col, "raw_realized_utility"),
        "top_decile": top_decile_utility(df, score_col, "raw_realized_utility"),
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def win_flags(candidate: dict[str, float | None], proxy: dict[str, float | None]) -> dict[str, bool]:
    cand_rank = safe_float(candidate.get("rank"))
    proxy_rank = safe_float(proxy.get("rank"))
    cand_pair = candidate.get("pair")
    proxy_pair = proxy.get("pair")
    cand_top = candidate.get("top_decile")
    proxy_top = proxy.get("top_decile")
    return {
        "rank": bool(cand_rank >= proxy_rank + 0.03),
        "pair": bool(cand_pair is not None and proxy_pair is not None and cand_pair >= proxy_pair + 0.02),
        "top_decile": bool(cand_top is not None and proxy_top is not None and cand_top >= proxy_top),
    }


def core_win_count(flags: dict[str, bool]) -> int:
    return int(sum(bool(value) for value in flags.values()))


def candidate_summary_row(
    *,
    feature_set: str,
    feature_cols: list[str],
    method: str,
    regularization: float,
    train_metrics: dict[str, float | None],
    val_metrics: dict[str, float | None],
    train_proxy: dict[str, float | None],
    val_proxy: dict[str, float | None],
    selected: bool = False,
    training_error: str = "",
) -> dict[str, Any]:
    train_wins = win_flags(train_metrics, train_proxy)
    val_wins = win_flags(val_metrics, val_proxy)
    return {
        "feature_set": feature_set,
        "feature_cols": ",".join(feature_cols),
        "method": method,
        "regularization": regularization,
        "selected_by_train": bool(selected),
        "training_error": training_error,
        "train_rank": train_metrics.get("rank"),
        "train_pair": train_metrics.get("pair"),
        "train_top_decile": train_metrics.get("top_decile"),
        "train_win_count_vs_proxy": core_win_count(train_wins),
        "train_rank_win": train_wins["rank"],
        "train_pair_win": train_wins["pair"],
        "train_top_decile_win": train_wins["top_decile"],
        "val_rank": val_metrics.get("rank"),
        "val_pair": val_metrics.get("pair"),
        "val_top_decile": val_metrics.get("top_decile"),
        "val_win_count_vs_proxy": core_win_count(val_wins),
        "val_rank_win": val_wins["rank"],
        "val_pair_win": val_wins["pair"],
        "val_top_decile_win": val_wins["top_decile"],
    }


def build_pairwise_training(train: pd.DataFrame, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    train = train.reset_index(drop=True)
    feats = feats.reset_index(drop=True)
    for _, group_idx in train.groupby("sample_id").groups.items():
        idxs = list(group_idx)
        if len(idxs) < 2:
            continue
        for left_pos, left_idx in enumerate(idxs):
            for right_idx in idxs[left_pos + 1 :]:
                utility_delta = float(train.loc[left_idx, "raw_realized_utility"] - train.loc[right_idx, "raw_realized_utility"])
                if abs(utility_delta) < 1e-9:
                    continue
                diff = feats.loc[left_idx].to_numpy(dtype=float) - feats.loc[right_idx].to_numpy(dtype=float)
                label = 1 if utility_delta > 0 else 0
                rows.append(diff)
                labels.append(label)
                rows.append(-diff)
                labels.append(1 - label)
    if not rows:
        return np.zeros((0, feats.shape[1]), dtype=float), np.zeros((0,), dtype=int)
    return np.vstack(rows), np.asarray(labels, dtype=int)


def select_candidate(summary: pd.DataFrame) -> pd.Series:
    candidates = summary[summary["training_error"].astype(str).eq("")].copy()
    if candidates.empty:
        return pd.Series(dtype=object)
    for col in ["train_pair", "train_rank", "train_top_decile"]:
        candidates[col] = candidates[col].map(safe_float)
    return candidates.sort_values(
        ["train_win_count_vs_proxy", "train_pair", "train_rank", "train_top_decile", "method", "regularization"],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]


def evaluate_feature_set(
    *,
    name: str,
    feature_cols: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    train_proxy: dict[str, float | None],
    val_proxy: dict[str, float | None],
    ridge_alpha_grid: list[float],
    pairwise_c_grid: list[float],
) -> tuple[pd.DataFrame, pd.Series]:
    train_features = feature_frame(train, feature_cols)
    val_features = feature_frame(val, feature_cols)
    rows: list[dict[str, Any]] = []

    for alpha in ridge_alpha_grid:
        try:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train_features, train["raw_realized_utility"].astype(float))
            train_scored = train.copy()
            val_scored = val.copy()
            train_scored["_score"] = model.predict(train_features)
            val_scored["_score"] = model.predict(val_features)
            rows.append(
                candidate_summary_row(
                    feature_set=name,
                    feature_cols=feature_cols,
                    method="ridge_utility",
                    regularization=alpha,
                    train_metrics=method_metrics(train_scored, "_score"),
                    val_metrics=method_metrics(val_scored, "_score"),
                    train_proxy=train_proxy,
                    val_proxy=val_proxy,
                )
            )
        except Exception as exc:
            rows.append(
                candidate_summary_row(
                    feature_set=name,
                    feature_cols=feature_cols,
                    method="ridge_utility",
                    regularization=alpha,
                    train_metrics={},
                    val_metrics={},
                    train_proxy=train_proxy,
                    val_proxy=val_proxy,
                    training_error=str(exc),
                )
            )

    pair_x, pair_y = build_pairwise_training(train, train_features)
    for c_value in pairwise_c_grid:
        try:
            if len(pair_y) == 0:
                raise ValueError("no non-tied train utility pairs")
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, solver="lbfgs"),
            )
            model.fit(pair_x, pair_y)
            train_scored = train.copy()
            val_scored = val.copy()
            train_scored["_score"] = model.decision_function(train_features.to_numpy(dtype=float))
            val_scored["_score"] = model.decision_function(val_features.to_numpy(dtype=float))
            rows.append(
                candidate_summary_row(
                    feature_set=name,
                    feature_cols=feature_cols,
                    method="pairwise_logistic",
                    regularization=c_value,
                    train_metrics=method_metrics(train_scored, "_score"),
                    val_metrics=method_metrics(val_scored, "_score"),
                    train_proxy=train_proxy,
                    val_proxy=val_proxy,
                )
            )
        except Exception as exc:
            rows.append(
                candidate_summary_row(
                    feature_set=name,
                    feature_cols=feature_cols,
                    method="pairwise_logistic",
                    regularization=c_value,
                    train_metrics={},
                    val_metrics={},
                    train_proxy=train_proxy,
                    val_proxy=val_proxy,
                    training_error=str(exc),
                )
            )

    summary = pd.DataFrame(rows)
    selected = select_candidate(summary)
    if not selected.empty:
        summary.loc[selected.name, "selected_by_train"] = True
        selected = summary.loc[selected.name].copy()
    return summary, selected


def selected_lookup(selected: pd.DataFrame, feature_set: str) -> dict[str, Any]:
    rows = selected[selected["feature_set"].astype(str).eq(feature_set)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def val_delta(left: dict[str, Any], right: dict[str, Any], metric: str) -> float:
    return safe_float(left.get(f"val_{metric}")) - safe_float(right.get(f"val_{metric}"))


def attribution_metrics(selected: pd.DataFrame, proxy_val: dict[str, float | None]) -> dict[str, Any]:
    full = selected_lookup(selected, "full")
    no_flow = selected_lookup(selected, "no_flow")
    only_flow = selected_lookup(selected, "only_flow")
    if not full or not no_flow or not only_flow:
        return {
            "flow_attribution_supported": False,
            "no_flow_matches_or_exceeds_full": False,
            "missing_required_attribution_set": True,
        }

    full_vs_no_flow = {
        "rank": val_delta(full, no_flow, "rank"),
        "pair": val_delta(full, no_flow, "pair"),
        "top_decile": val_delta(full, no_flow, "top_decile"),
    }
    only_flow_vs_full = {
        "rank": val_delta(only_flow, full, "rank"),
        "pair": val_delta(only_flow, full, "pair"),
        "top_decile": val_delta(only_flow, full, "top_decile"),
    }
    flow_specific_edge_count = int(
        (full_vs_no_flow["rank"] >= 0.03)
        + (full_vs_no_flow["pair"] >= 0.02)
        + (full_vs_no_flow["top_decile"] > 0.0)
    )
    no_flow_matches_or_exceeds_full = bool(
        int(no_flow.get("val_win_count_vs_proxy", 0)) >= int(full.get("val_win_count_vs_proxy", 0))
        and val_delta(no_flow, full, "rank") >= -1e-9
        and val_delta(no_flow, full, "pair") >= -1e-9
    )
    only_flow_underperforms_full = bool(
        int(only_flow.get("val_win_count_vs_proxy", 0)) < int(full.get("val_win_count_vs_proxy", 0))
        or only_flow_vs_full["pair"] < -0.02
    )
    flow_attribution_supported = bool(
        flow_specific_edge_count >= 2
        and not no_flow_matches_or_exceeds_full
        and int(full.get("val_win_count_vs_proxy", 0)) >= 2
    )
    return {
        "flow_attribution_supported": flow_attribution_supported,
        "no_flow_matches_or_exceeds_full": no_flow_matches_or_exceeds_full,
        "only_flow_underperforms_full": only_flow_underperforms_full,
        "missing_required_attribution_set": False,
        "flow_specific_edge_metric_count": flow_specific_edge_count,
        "full_vs_no_flow_val_delta": full_vs_no_flow,
        "only_flow_vs_full_val_delta": only_flow_vs_full,
        "proxy_eval_rank": safe_float(proxy_val.get("rank")),
        "proxy_eval_pair": safe_float(proxy_val.get("pair")),
        "proxy_eval_top_decile": safe_float(proxy_val.get("top_decile")),
        "full_selected": full,
        "no_flow_selected": no_flow,
        "only_flow_selected": only_flow,
    }


def parse_float_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/tables/11_v6_flow_predictions.csv")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--ridge-alpha-grid", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--pairwise-c-grid", default="0.01,0.03,0.1,0.3,1,3,10,100")
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-eval-rows", type=int, default=300)
    parser.add_argument("--output", default="outputs/tables/11_7_v6_flow_reranker_ablation_attribution.csv")
    parser.add_argument("--grid-output", default="outputs/tables/11_7_v6_flow_reranker_ablation_grid.csv")
    parser.add_argument("--metrics", default="outputs/metrics/11_7_v6_flow_reranker_ablation_attribution.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.predictions).exists():
        failures.append(f"predictions missing: {args.predictions}")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(args.predictions)

    required_cols = {
        "sample_id",
        "split",
        "raw_realized_utility",
        *{col for cols in FEATURE_SETS.values() for col in cols},
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        failures.append(f"predictions missing columns: {missing_cols}")

    train = df[df["split"].astype(str).eq(args.train_split)].copy() if not df.empty else pd.DataFrame()
    val = df[df["split"].astype(str).eq(args.eval_split)].copy() if not df.empty else pd.DataFrame()
    if len(train) < args.min_train_rows:
        failures.append(f"train rows {len(train)} < {args.min_train_rows}")
    if len(val) < args.min_eval_rows:
        failures.append(f"eval rows {len(val)} < {args.min_eval_rows}")

    metrics: dict[str, Any] = {
        "pipeline_pass": False,
        "claim_allowed": False,
        "diagnostic_only": True,
    }
    selected_table = pd.DataFrame()
    grid_table = pd.DataFrame()
    if not failures:
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        train_proxy = method_metrics(train, "proxy_average_reward")
        val_proxy = method_metrics(val, "proxy_average_reward")
        ridge_alpha_grid = parse_float_grid(args.ridge_alpha_grid)
        pairwise_c_grid = parse_float_grid(args.pairwise_c_grid)

        grid_parts: list[pd.DataFrame] = []
        selected_rows: list[pd.Series] = []
        for name, feature_cols in FEATURE_SETS.items():
            grid, selected = evaluate_feature_set(
                name=name,
                feature_cols=feature_cols,
                train=train,
                val=val,
                train_proxy=train_proxy,
                val_proxy=val_proxy,
                ridge_alpha_grid=ridge_alpha_grid,
                pairwise_c_grid=pairwise_c_grid,
            )
            grid_parts.append(grid)
            if selected.empty:
                failures.append(f"no selected candidate for feature set: {name}")
            else:
                selected_rows.append(selected)

        grid_records = [record for part in grid_parts for record in part.to_dict(orient="records")]
        grid_table = pd.DataFrame(grid_records)
        selected_table = pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else pd.DataFrame()
        missing_sets = sorted(set(REQUIRED_ATTRIBUTION_SETS) - set(selected_table.get("feature_set", [])))
        if missing_sets:
            failures.append(f"required attribution feature sets missing selected rows: {missing_sets}")

        attr = attribution_metrics(selected_table, val_proxy)
        metrics.update(
            {
                "pipeline_pass": not failures,
                "claim_allowed": False,
                "diagnostic_only": True,
                "train_rows": int(len(train)),
                "eval_rows": int(len(val)),
                "feature_set_count": int(len(FEATURE_SETS)),
                "selected_feature_sets": selected_table.get("feature_set", pd.Series(dtype=str)).astype(str).tolist(),
                "flow_attribution_supported": bool(attr.get("flow_attribution_supported", False)),
                "no_flow_matches_or_exceeds_full": bool(attr.get("no_flow_matches_or_exceeds_full", False)),
                "only_flow_underperforms_full": bool(attr.get("only_flow_underperforms_full", False)),
                "flow_specific_edge_metric_count": int(attr.get("flow_specific_edge_metric_count", 0)),
                "proxy_eval_rank": safe_float(val_proxy.get("rank")),
                "proxy_eval_pair": safe_float(val_proxy.get("pair")),
                "proxy_eval_top_decile": safe_float(val_proxy.get("top_decile")),
                "full_vs_no_flow_val_delta": attr.get("full_vs_no_flow_val_delta", {}),
                "only_flow_vs_full_val_delta": attr.get("only_flow_vs_full_val_delta", {}),
                "full_selected": attr.get("full_selected", {}),
                "no_flow_selected": attr.get("no_flow_selected", {}),
                "only_flow_selected": attr.get("only_flow_selected", {}),
                "claim_boundary": (
                    "diagnostic only; Step 11.7 tests attribution of the Step 11.6 reranker signal. "
                    "A Flow claim remains blocked unless an official Flow checkpoint beats proxy and "
                    "feature ablations show the improvement depends on Flow-specific signal."
                ),
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    selected_table.to_csv(args.output, index=False)
    Path(args.grid_output).parent.mkdir(parents=True, exist_ok=True)
    grid_table.to_csv(args.grid_output, index=False)
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [args.predictions, args.output, args.grid_output, args.metrics],
        STEP,
        extra={
            "references": [
                "Bradley-Terry paired-comparison ranking motivation",
                "Burges RankNet/LambdaRank/LambdaMART learning-to-rank motivation",
                "ablation attribution: compare full, no-flow, and only-flow feature sets under train-only selection",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.predictions],
        [args.output, args.grid_output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
