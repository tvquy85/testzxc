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

STEP = "11_6_FLOW_UTILITY_RERANKER_PROBE_V6"

FEATURE_COLS = [
    "flow_reward_score",
    "proxy_average_reward",
    "single_best_judge_reward",
    "news_grounding_score",
    "technical_grounding_score",
    "unsupported_news_claim_rate",
]


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURE_COLS].astype(float).copy()
    out["neg_unsupported_news_claim_rate"] = -out["unsupported_news_claim_rate"]
    return out


def method_metrics(df: pd.DataFrame, score_col: str) -> dict[str, float | None]:
    return {
        "rank": spearman(df[score_col], df["raw_realized_utility"]),
        "pair": preference_pair_accuracy(df, score_col, "raw_realized_utility"),
        "top_decile": top_decile_utility(df, score_col, "raw_realized_utility"),
    }


def win_flags(candidate: dict[str, float | None], proxy: dict[str, float | None]) -> dict[str, bool]:
    cand_rank = candidate.get("rank") or 0.0
    proxy_rank = proxy.get("rank") or 0.0
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def candidate_summary_row(
    *,
    method: str,
    alpha_or_c: float,
    train_metrics: dict[str, float | None],
    val_metrics: dict[str, float | None],
    train_proxy: dict[str, float | None],
    val_proxy: dict[str, float | None],
) -> dict[str, Any]:
    train_wins = win_flags(train_metrics, train_proxy)
    val_wins = win_flags(val_metrics, val_proxy)
    return {
        "method": method,
        "regularization": alpha_or_c,
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


def build_pairwise_training(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    feats = feature_frame(train)
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
        return np.zeros((0, feature_frame(train).shape[1]), dtype=float), np.zeros((0,), dtype=int)
    return np.vstack(rows), np.asarray(labels, dtype=int)


def add_baseline_rows(
    rows: list[dict[str, Any]],
    train: pd.DataFrame,
    val: pd.DataFrame,
    train_proxy: dict[str, float | None],
    val_proxy: dict[str, float | None],
) -> None:
    for method, col in [
        ("flow_reward_v6", "flow_reward_score"),
        ("proxy_average_reward", "proxy_average_reward"),
        ("single_best_judge_reward", "single_best_judge_reward"),
    ]:
        rows.append(
            candidate_summary_row(
                method=method,
                alpha_or_c=0.0,
                train_metrics=method_metrics(train, col),
                val_metrics=method_metrics(val, col),
                train_proxy=train_proxy,
                val_proxy=val_proxy,
            )
        )


def select_candidate(summary: pd.DataFrame) -> pd.Series:
    candidates = summary[summary["method"].str.startswith(("ridge_utility", "pairwise_logistic"))].copy()
    if candidates.empty:
        return pd.Series(dtype=object)
    return candidates.sort_values(
        ["train_win_count_vs_proxy", "train_pair", "train_rank", "train_top_decile", "method", "regularization"],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/tables/11_v6_flow_predictions.csv")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--ridge-alpha-grid", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--pairwise-c-grid", default="0.01,0.03,0.1,0.3,1,3,10,100")
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-eval-rows", type=int, default=300)
    parser.add_argument("--summary-output", default="outputs/tables/11_6_v6_flow_utility_reranker_summary.csv")
    parser.add_argument("--predictions-output", default="outputs/tables/11_6_v6_flow_utility_reranker_predictions.csv")
    parser.add_argument("--metrics", default="outputs/metrics/11_6_v6_flow_utility_reranker_probe.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.predictions).exists():
        failures.append(f"predictions missing: {args.predictions}")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(args.predictions)
    missing = sorted({"sample_id", "split", "raw_realized_utility", *FEATURE_COLS} - set(df.columns))
    if missing:
        failures.append(f"predictions missing columns: {missing}")
    train = df[df["split"].astype(str).eq(args.train_split)].copy() if not df.empty else pd.DataFrame()
    val = df[df["split"].astype(str).eq(args.eval_split)].copy() if not df.empty else pd.DataFrame()
    if len(train) < args.min_train_rows:
        failures.append(f"train rows {len(train)} < {args.min_train_rows}")
    if len(val) < args.min_eval_rows:
        failures.append(f"eval rows {len(val)} < {args.min_eval_rows}")

    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    summary = pd.DataFrame()
    prediction_table = df.copy() if not df.empty else pd.DataFrame()
    if not failures:
        train_features = feature_frame(train)
        val_features = feature_frame(val)
        train_proxy = method_metrics(train, "proxy_average_reward")
        val_proxy = method_metrics(val, "proxy_average_reward")
        rows: list[dict[str, Any]] = []
        add_baseline_rows(rows, train, val, train_proxy, val_proxy)

        for alpha in [float(item) for item in args.ridge_alpha_grid.split(",") if item.strip()]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train_features, train["raw_realized_utility"].astype(float))
            col = f"ridge_utility_alpha_{alpha:g}"
            train[col] = model.predict(train_features)
            val[col] = model.predict(val_features)
            rows.append(
                candidate_summary_row(
                    method="ridge_utility",
                    alpha_or_c=alpha,
                    train_metrics=method_metrics(train, col),
                    val_metrics=method_metrics(val, col),
                    train_proxy=train_proxy,
                    val_proxy=val_proxy,
                )
            )

        pair_x, pair_y = build_pairwise_training(train)
        if len(pair_y) == 0:
            failures.append("no non-tied train utility pairs for pairwise reranker")
        else:
            for c_value in [float(item) for item in args.pairwise_c_grid.split(",") if item.strip()]:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, solver="lbfgs"),
                )
                model.fit(pair_x, pair_y)
                col = f"pairwise_logistic_c_{c_value:g}"
                train[col] = model.decision_function(train_features.to_numpy(dtype=float))
                val[col] = model.decision_function(val_features.to_numpy(dtype=float))
                rows.append(
                    candidate_summary_row(
                        method="pairwise_logistic",
                        alpha_or_c=c_value,
                        train_metrics=method_metrics(train, col),
                        val_metrics=method_metrics(val, col),
                        train_proxy=train_proxy,
                        val_proxy=val_proxy,
                    )
                )

        summary = pd.DataFrame(rows)
        selected = select_candidate(summary)
        if selected.empty:
            failures.append("no utility reranker candidate selected")
        else:
            selected_method = str(selected["method"])
            selected_reg = float(selected["regularization"])
            if selected_method == "ridge_utility":
                selected_col = f"ridge_utility_alpha_{selected_reg:g}"
            else:
                selected_col = f"pairwise_logistic_c_{selected_reg:g}"
            prediction_table = df.copy()
            prediction_table["utility_reranker_score"] = np.nan
            prediction_table.loc[train.index, "utility_reranker_score"] = train[selected_col].to_numpy()
            prediction_table.loc[val.index, "utility_reranker_score"] = val[selected_col].to_numpy()
            val_wins = {
                "rank": bool(selected["val_rank_win"]),
                "pair": bool(selected["val_pair_win"]),
                "top_decile": bool(selected["val_top_decile_win"]),
            }
            train_wins = {
                "rank": bool(selected["train_rank_win"]),
                "pair": bool(selected["train_pair_win"]),
                "top_decile": bool(selected["train_top_decile_win"]),
            }
            metrics.update(
                {
                    "pipeline_pass": True,
                    "claim_allowed": False,
                    "diagnostic_only": True,
                    "train_rows": int(len(train)),
                    "eval_rows": int(len(val)),
                    "pairwise_train_examples": int(len(pair_y)) if "pair_y" in locals() else 0,
                    "selected_method": selected_method,
                    "selected_regularization": selected_reg,
                    "train_metric_wins_vs_proxy": train_wins,
                    "train_core_win_count_vs_proxy": core_win_count(train_wins),
                    "eval_metric_wins_vs_proxy": val_wins,
                    "eval_core_win_count_vs_proxy": core_win_count(val_wins),
                    "eval_core_utility_win_vs_proxy": bool(core_win_count(val_wins) >= 2),
                    "eval_rank": safe_float(selected["val_rank"]),
                    "eval_pair": safe_float(selected["val_pair"]),
                    "eval_top_decile": safe_float(selected["val_top_decile"]),
                    "proxy_eval_rank": safe_float(val_proxy["rank"]),
                    "proxy_eval_pair": safe_float(val_proxy["pair"]),
                    "proxy_eval_top_decile": safe_float(val_proxy["top_decile"]),
                    "claim_boundary": (
                        "diagnostic only; utility-aware reranker is trained on existing train split and evaluated on val, "
                        "but it is not the official Flow distribution-matching checkpoint and lacks required ablation checkpoints"
                    ),
                }
            )

    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(args.predictions_output, index=False)
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [args.predictions, args.summary_output, args.predictions_output, args.metrics],
        STEP,
        extra={
            "references": [
                "Bradley-Terry paired-comparison ranking motivation",
                "Burges RankNet/LambdaRank/LambdaMART learning-to-rank motivation",
                "scikit-learn Ridge regularized utility regression",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.predictions],
        [args.summary_output, args.predictions_output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
