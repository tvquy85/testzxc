from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reward.evaluate_flow_vs_proxy_v4 import preference_pair_accuracy, spearman, top_decile_utility
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "11_5_FLOW_UTILITY_SURFACE_DIAGNOSTIC_V6"

METHODS = {
    "flow_reward_v6": "flow_reward_score",
    "proxy_average_reward": "proxy_average_reward",
    "single_best_judge_reward": "single_best_judge_reward",
    "technical_rule_reward": "technical_rule_reward",
}


def pair_anatomy(df: pd.DataFrame) -> dict[str, Any]:
    total_pairs = 0
    utility_varying_pairs = 0
    sample_count = 0
    samples_with_utility_variation = 0
    for _, group in df.groupby("sample_id"):
        if len(group) < 2:
            continue
        sample_count += 1
        values = group["raw_realized_utility"].astype(float).to_numpy()
        sample_has_variation = False
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total_pairs += 1
                if abs(float(values[i] - values[j])) >= 1e-9:
                    utility_varying_pairs += 1
                    sample_has_variation = True
        samples_with_utility_variation += int(sample_has_variation)
    return {
        "candidate_pair_count": int(total_pairs),
        "utility_varying_pair_count": int(utility_varying_pairs),
        "utility_varying_pair_rate": float(utility_varying_pairs / max(1, total_pairs)),
        "multi_candidate_sample_count": int(sample_count),
        "samples_with_utility_variation": int(samples_with_utility_variation),
        "samples_with_utility_variation_rate": float(samples_with_utility_variation / max(1, sample_count)),
    }


def method_summary(df: pd.DataFrame, method: str, score_col: str) -> dict[str, Any]:
    cutoff = df[score_col].quantile(0.90)
    top = df[df[score_col] >= cutoff].copy()
    return {
        "method": method,
        "score_col": score_col,
        "rank_correlation_with_raw_realized_utility": spearman(df[score_col], df["raw_realized_utility"]),
        "preference_pair_accuracy_by_raw_utility": preference_pair_accuracy(df, score_col, "raw_realized_utility"),
        "top_decile_raw_realized_utility": top_decile_utility(df, score_col, "raw_realized_utility"),
        "top_decile_rows": int(len(top)),
        "top_decile_score_cutoff": float(cutoff),
        "top_decile_mean_raw_realized_utility": float(top["raw_realized_utility"].mean()) if len(top) else 0.0,
        "top_decile_mean_technical_rule_delta": float(top["technical_rule_delta"].mean()) if len(top) else 0.0,
        "top_decile_mean_news_grounding_score": float(top["news_grounding_score"].mean()) if len(top) else 0.0,
        "top_decile_mean_technical_grounding_score": float(top["technical_grounding_score"].mean()) if len(top) else 0.0,
        "top_decile_mean_target_probability": float(top["single_best_judge_reward"].mean()) if len(top) else 0.0,
        "top_decile_unsupported_claim_rate": float(top["unsupported_news_claim_rate"].mean()) if len(top) else 0.0,
    }


def top_indices(df: pd.DataFrame, score_col: str) -> set[int]:
    cutoff = df[score_col].quantile(0.90)
    return set(int(idx) for idx in df.index[df[score_col] >= cutoff])


def overlap_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    top_sets = {method: top_indices(df, col) for method, col in METHODS.items()}
    rows: list[dict[str, Any]] = []
    names = list(top_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            inter = top_sets[left] & top_sets[right]
            union = top_sets[left] | top_sets[right]
            rows.append(
                {
                    "left_method": left,
                    "right_method": right,
                    "intersection": int(len(inter)),
                    "union": int(len(union)),
                    "jaccard_overlap": float(len(inter) / max(1, len(union))),
                }
            )
    return rows


def quantile_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, score_col in METHODS.items():
        ranked = df[[score_col, "raw_realized_utility", "technical_rule_delta", "news_grounding_score"]].copy()
        ranked["_bucket"] = pd.qcut(ranked[score_col].rank(method="first"), 10, labels=False, duplicates="drop")
        for bucket, group in ranked.groupby("_bucket"):
            rows.append(
                {
                    "method": method,
                    "score_decile": int(bucket),
                    "rows": int(len(group)),
                    "score_min": float(group[score_col].min()),
                    "score_max": float(group[score_col].max()),
                    "mean_raw_realized_utility": float(group["raw_realized_utility"].mean()),
                    "mean_technical_rule_delta": float(group["technical_rule_delta"].mean()),
                    "mean_news_grounding_score": float(group["news_grounding_score"].mean()),
                }
            )
    return rows


def disagreement_examples(df: pd.DataFrame, limit: int = 100) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for sample_id, group in df.groupby("sample_id"):
        if len(group) < 2:
            continue
        records = group.reset_index()[[
            "index",
            "candidate_id",
            "raw_realized_utility",
            "flow_reward_score",
            "proxy_average_reward",
            "technical_rule_reward",
        ]].to_dict(orient="records")
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                left, right = records[i], records[j]
                utility_delta = float(left["raw_realized_utility"] - right["raw_realized_utility"])
                if abs(utility_delta) < 1e-9:
                    continue
                flow_delta = float(left["flow_reward_score"] - right["flow_reward_score"])
                proxy_delta = float(left["proxy_average_reward"] - right["proxy_average_reward"])
                technical_delta = float(left["technical_rule_reward"] - right["technical_rule_reward"])
                flow_correct = abs(flow_delta) >= 1e-9 and ((utility_delta > 0) == (flow_delta > 0))
                proxy_correct = abs(proxy_delta) >= 1e-9 and ((utility_delta > 0) == (proxy_delta > 0))
                technical_correct = abs(technical_delta) >= 1e-9 and ((utility_delta > 0) == (technical_delta > 0))
                if not flow_correct and (proxy_correct or technical_correct):
                    examples.append(
                        {
                            "sample_id": str(sample_id),
                            "left_candidate_id": left["candidate_id"],
                            "right_candidate_id": right["candidate_id"],
                            "utility_delta": utility_delta,
                            "flow_delta": flow_delta,
                            "proxy_delta": proxy_delta,
                            "technical_delta": technical_delta,
                            "proxy_correct": bool(proxy_correct),
                            "technical_correct": bool(technical_correct),
                        }
                    )
                    if len(examples) >= limit:
                        return examples
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/tables/11_v6_flow_predictions.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--metrics", default="outputs/metrics/11_5_v6_flow_utility_diagnostic.json")
    parser.add_argument("--summary-table", default="outputs/tables/11_5_v6_flow_method_summary.csv")
    parser.add_argument("--overlap-table", default="outputs/tables/11_5_v6_flow_top_decile_overlap.csv")
    parser.add_argument("--quantile-table", default="outputs/tables/11_5_v6_flow_score_deciles.csv")
    parser.add_argument("--examples", default="review_samples/currentdata_v6/11_5_flow_pair_disagreement_examples.jsonl")
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
        "candidate_id",
        "split",
        "raw_realized_utility",
        "technical_rule_delta",
        "news_grounding_score",
        "technical_grounding_score",
        "unsupported_news_claim_rate",
        *METHODS.values(),
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        failures.append(f"predictions missing columns: {missing}")
    eval_df = df[df["split"].astype(str).eq(args.split)].copy() if not df.empty and "split" in df.columns else pd.DataFrame()
    if not failures and len(eval_df) < 300:
        failures.append(f"eval rows {len(eval_df)} < 300")

    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    summary = pd.DataFrame()
    overlaps = pd.DataFrame()
    quantiles = pd.DataFrame()
    examples: list[dict[str, Any]] = []
    if not failures:
        pair_info = pair_anatomy(eval_df)
        summary_rows = [method_summary(eval_df, method, col) for method, col in METHODS.items()]
        summary = pd.DataFrame(summary_rows)
        overlaps = pd.DataFrame(overlap_rows(eval_df))
        quantiles = pd.DataFrame(quantile_rows(eval_df))
        examples = disagreement_examples(eval_df)
        lookup = {row["method"]: row for row in summary_rows}
        flow = lookup["flow_reward_v6"]
        proxy = lookup["proxy_average_reward"]
        technical = lookup["technical_rule_reward"]
        flow_vs_proxy_top_gap = float(flow["top_decile_raw_realized_utility"] - proxy["top_decile_raw_realized_utility"])
        flow_vs_technical_top_gap = float(flow["top_decile_raw_realized_utility"] - technical["top_decile_raw_realized_utility"])
        overlap_lookup = {
            (row["left_method"], row["right_method"]): row["jaccard_overlap"]
            for row in overlaps.to_dict(orient="records")
        }
        metrics = {
            "pipeline_pass": True,
            "claim_allowed": False,
            "eval_split": args.split,
            "eval_rows": int(len(eval_df)),
            **pair_info,
            "flow_rank_win_vs_proxy": bool(
                flow["rank_correlation_with_raw_realized_utility"]
                > proxy["rank_correlation_with_raw_realized_utility"]
            ),
            "flow_pair_accuracy_gap_vs_proxy": float(
                flow["preference_pair_accuracy_by_raw_utility"]
                - proxy["preference_pair_accuracy_by_raw_utility"]
            ),
            "flow_top_decile_gap_vs_proxy": flow_vs_proxy_top_gap,
            "flow_top_decile_gap_vs_technical": flow_vs_technical_top_gap,
            "flow_top_decile_overlap_with_proxy": float(overlap_lookup.get(("flow_reward_v6", "proxy_average_reward"), 0.0)),
            "flow_top_decile_overlap_with_technical": float(overlap_lookup.get(("flow_reward_v6", "technical_rule_reward"), 0.0)),
            "diagnostic_root_cause": (
                "Flow ranks utility better than proxy globally, but pairwise supervision is sparse and "
                "top-decile selection is misaligned with technical/raw utility."
            ),
            "recommended_next_fix": (
                "Train/evaluate a listwise or pairwise utility-aware objective with explicit top-decile/technical-utility terms; "
                "do not claim Flow improvement from the current MSE distribution-matching model."
            ),
        }

    for path, table in [
        (args.summary_table, summary),
        (args.overlap_table, overlaps),
        (args.quantile_table, quantiles),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    Path(args.examples).parent.mkdir(parents=True, exist_ok=True)
    with open(args.examples, "w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [args.predictions, args.metrics, args.summary_table, args.overlap_table, args.quantile_table, args.examples],
        STEP,
        extra={
            "references": [
                "Burges 2010 RankNet/LambdaRank/LambdaMART top-of-list ranking motivation",
                "Bradley-Terry pairwise preference modeling motivation",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.predictions],
        [args.metrics, args.summary_table, args.overlap_table, args.quantile_table, args.examples, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
