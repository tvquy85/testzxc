from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment.build_alignment_current_v4 import (
    action_value,
    finite_series,
    load_json,
    normalize_score,
    normalize_utility,
    parse_action,
    write_jsonl,
)
from src.utils.artifacts import sha256_file, write_json, write_manifest, write_status

STEP = "13_ALIGNMENT_DATASET_MEDIUM_RWSFT_DPO"


def merge_reward_sources(args: argparse.Namespace, failures: list[str]) -> pd.DataFrame:
    rationales = pd.read_parquet(args.rationales)
    inferability = pd.read_parquet(args.inferability)
    grounding = pd.read_parquet(args.grounding)
    tracks = pd.read_parquet(args.tracks) if args.tracks and os.path.exists(args.tracks) else pd.DataFrame()
    flow_scores = pd.read_csv(args.flow_table) if args.flow_table and os.path.exists(args.flow_table) else pd.DataFrame()
    flow_metrics = load_json(args.flow_metrics)
    flow_allowed = bool(flow_metrics.get("flow_reward_improvement") or flow_metrics.get("flow_claim_allowed"))

    df = rationales.copy()
    df["candidate_id"] = df["candidate_id"].astype(int)
    inferability["candidate_id"] = inferability["candidate_id"].astype(int)
    grounding["candidate_id"] = grounding["candidate_id"].astype(int)
    inf_cols = [
        col
        for col in [
            "sample_id",
            "candidate_id",
            "true_label_probability_debiased",
            "true_label_probability",
            "argmax_consistency_multi",
            "argmax_consistency",
            "l1_probability_delta",
        ]
        if col in inferability.columns
    ]
    df = df.merge(inferability[inf_cols], on=["sample_id", "candidate_id"], how="left")
    g_cols = [
        col
        for col in [
            "sample_id",
            "candidate_id",
            "status",
            "total_claims",
            "supported_claims",
            "news_grounding_score",
            "technical_grounding_score",
        ]
        if col in grounding.columns
    ]
    df = df.merge(grounding[g_cols], on=["sample_id", "candidate_id"], how="left")
    df.rename(columns={"status": "grounding_status"}, inplace=True)
    if not tracks.empty:
        t_cols = [
            col
            for col in [
                "sample_id",
                "target_return",
                "abnormal_return_h1",
                "mean_evidence_quality_score",
                "split",
                "track",
                "news_reasoning_track",
            ]
            if col in tracks.columns
        ]
        df = df.merge(tracks[t_cols].drop_duplicates("sample_id"), on="sample_id", how="left", suffixes=("", "_track"))
        if "split_track" in df.columns:
            df["split"] = df["split"].combine_first(df["split_track"]) if "split" in df.columns else df["split_track"]
        if "track_track" in df.columns:
            df["track"] = df["track"].combine_first(df["track_track"]) if "track" in df.columns else df["track_track"]

    true_prob = finite_series(df, "true_label_probability_debiased", np.nan).fillna(finite_series(df, "true_label_probability", 0.0))
    news_ground = finite_series(df, "news_grounding_score", 0.5).fillna(0.5)
    tech_ground = finite_series(df, "technical_grounding_score", 0.5).fillna(0.5)
    evidence_weight = finite_series(df, "mean_evidence_quality_score", 0.5).fillna(0.5).clip(0.0, 1.0)
    stability = finite_series(df, "argmax_consistency_multi", np.nan).fillna(finite_series(df, "argmax_consistency", 0.5)).clip(0.0, 1.0)
    schema = df["schema_ok"].astype(float) if "schema_ok" in df.columns else pd.Series(0.0, index=df.index)
    returns = finite_series(df, "target_return", np.nan).fillna(finite_series(df, "abnormal_return_h1", 0.0))
    action = df.get("parsed_json", "").apply(parse_action)
    utility = action.apply(action_value) * returns - action.apply(action_value).abs() * 0.001
    utility_norm = normalize_utility(utility)
    df["proxy_reward"] = (
        0.30 * true_prob.clip(0.0, 1.0)
        + 0.20 * news_ground.clip(0.0, 1.0)
        + 0.20 * tech_ground.clip(0.0, 1.0)
        + 0.10 * evidence_weight
        + 0.10 * stability
        + 0.05 * utility_norm
        + 0.05 * schema
    )
    df["final_reward"] = df["proxy_reward"]
    df["reward_source"] = "proxy_average_independent"
    if flow_allowed and not flow_scores.empty and {"sample_id", "candidate_id", "flow_reward"} <= set(flow_scores.columns):
        fs = flow_scores[["sample_id", "candidate_id", "flow_reward"]].copy()
        fs["candidate_id"] = fs["candidate_id"].astype(int)
        df = df.merge(fs, on=["sample_id", "candidate_id"], how="left")
        df["flow_reward_raw"] = pd.to_numeric(df["flow_reward"], errors="coerce")
        df["flow_reward_normalized"] = normalize_score(df["flow_reward_raw"])
        df["final_reward"] = df["flow_reward_normalized"].where(df["flow_reward_raw"].notna(), df["proxy_reward"])
        df["reward_source"] = "flow_v5"
    elif flow_allowed:
        failures.append("flow was allowed but per-candidate flow scores are unavailable; using proxy fallback")
        df["reward_source"] = "flow_allowed_scores_missing_proxy_used"
    return df


def valid_candidates(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.copy()
    if "schema_ok" in valid.columns:
        valid = valid[valid["schema_ok"].astype(bool)].copy()
    if "grounding_status" in valid.columns:
        valid = valid[~valid["grounding_status"].isin(["contradiction", "unsupported"])].copy()
    if "split" in valid.columns:
        valid = valid[valid["split"].astype(str).eq("train")].copy()
    return valid.sort_values(["sample_id", "final_reward", "candidate_id"], ascending=[True, False, True]).copy()


def build_rwsft(valid: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in valid.iterrows():
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "split": row.get("split"),
                "prompt": row.get("prompt"),
                "output": row.get("raw_output", row.get("raw_text")),
                "messages": [
                    {"role": "user", "content": row.get("prompt")},
                    {"role": "assistant", "content": row.get("raw_output", row.get("raw_text"))},
                ],
                "reward": float(row["final_reward"]),
                "reward_source": row.get("reward_source"),
                "candidate_id": int(row.get("candidate_id", 0)),
            }
        )
    return rows


def build_dpo(df: pd.DataFrame, min_reward_gap: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        group = group.sort_values("final_reward", ascending=False).copy()
        if len(group) < 2:
            continue
        chosen = group.iloc[0]
        for _, rejected in group.iloc[1:].iterrows():
            gap = float(chosen["final_reward"] - rejected["final_reward"])
            if gap < min_reward_gap:
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": chosen.get("split"),
                    "prompt": chosen.get("prompt"),
                    "chosen": chosen.get("raw_output", chosen.get("raw_text")),
                    "rejected": rejected.get("raw_output", rejected.get("raw_text")),
                    "chosen_reward": float(chosen["final_reward"]),
                    "rejected_reward": float(rejected["final_reward"]),
                    "reward_gap": gap,
                    "chosen_candidate_id": int(chosen.get("candidate_id", 0)),
                    "rejected_candidate_id": int(rejected.get("candidate_id", 0)),
                    "reward_source": chosen.get("reward_source"),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--inferability", required=True)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--flow-table", required=True)
    parser.add_argument("--flow-metrics", required=True)
    parser.add_argument("--tracks", default="data/processed/medium_clean_v4_contexts_gated.parquet")
    parser.add_argument("--rwsft-output", default="data/alignment/medium_clean_v4_rwsft.jsonl")
    parser.add_argument("--dpo-output", default="data/alignment/medium_clean_v4_dpo_pairs.jsonl")
    parser.add_argument("--scored-output", default="data/alignment/medium_clean_v4_scored_candidates.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/13_alignment_dataset_medium.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--min-rwsft", type=int, default=1000)
    parser.add_argument("--min-dpo", type=int, default=300)
    parser.add_argument("--min-reward-gap", type=float, default=0.005)
    args = parser.parse_args()

    failures: list[str] = []
    for path in [args.rationales, args.inferability, args.grounding, args.flow_metrics, args.flow_table, args.tracks]:
        if path and not os.path.exists(path):
            failures.append(f"missing input: {path}")
    if failures:
        write_status(args.status, STEP, "FAIL", [args.rationales, args.inferability, args.grounding, args.flow_metrics, args.flow_table, args.tracks], [args.status], {}, failures, False)
        return 1

    df = merge_reward_sources(args, failures)
    valid = valid_candidates(df)
    rwsft = build_rwsft(valid)
    dpo = build_dpo(valid, args.min_reward_gap)
    write_jsonl(args.rwsft_output, rwsft)
    write_jsonl(args.dpo_output, dpo)
    Path(args.scored_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.scored_output, index=False)

    train_only = True
    if rwsft or dpo:
        splits = {str(row.get("split")) for row in [*rwsft, *dpo]}
        train_only = splits <= {"train"}
    chosen_mean = float(np.mean([row["chosen_reward"] for row in dpo])) if dpo else 0.0
    rejected_mean = float(np.mean([row["rejected_reward"] for row in dpo])) if dpo else 0.0
    mean_gap = float(np.mean([row["reward_gap"] for row in dpo])) if dpo else 0.0
    metrics = {
        "pipeline_pass": True,
        "claim_allowed": False,
        "rwsft_examples": len(rwsft),
        "dpo_pairs": len(dpo),
        "unique_samples": int(df["sample_id"].nunique()) if len(df) else 0,
        "candidate_rows": int(len(df)),
        "valid_candidate_rows": int(len(valid)),
        "chosen_reward_mean": chosen_mean,
        "rejected_reward_mean": rejected_mean,
        "mean_chosen_reward": chosen_mean,
        "mean_rejected_reward": rejected_mean,
        "mean_reward_gap": mean_gap,
        "min_reward_gap": args.min_reward_gap,
        "reward_source": str(valid["reward_source"].mode().iloc[0]) if len(valid) and "reward_source" in valid else "unknown",
        "flow_reward_improvement": bool(load_json(args.flow_metrics).get("flow_reward_improvement")),
        "flow_scores_used": bool(str(valid["reward_source"].mode().iloc[0]).startswith("flow")) if len(valid) and "reward_source" in valid else False,
        "train_only": bool(train_only),
        "rwsft_sha256": sha256_file(args.rwsft_output) if os.path.exists(args.rwsft_output) else None,
        "dpo_sha256": sha256_file(args.dpo_output) if os.path.exists(args.dpo_output) else None,
    }
    if len(rwsft) < args.min_rwsft:
        failures.append(f"RWSFT count {len(rwsft)} < {args.min_rwsft}")
    if len(dpo) < args.min_dpo:
        failures.append(f"DPO count {len(dpo)} < {args.min_dpo}")
    if dpo and chosen_mean <= rejected_mean + 0.03:
        failures.append("chosen mean reward is not greater than rejected mean reward by 0.03")
    if not train_only:
        failures.append("alignment data contains non-train rows")

    outputs = [args.rwsft_output, args.dpo_output, args.scored_output, args.metrics]
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, outputs, STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.rationales, args.inferability, args.grounding, args.flow_metrics, args.flow_table, args.tracks],
        [*outputs, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
