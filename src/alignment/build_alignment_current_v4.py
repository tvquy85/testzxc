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
from src.utils.artifacts import sha256_file, write_json, write_manifest, write_status

STEP = "16_ALIGNMENT_REBUILD_CLEAN_V4"


def load_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_action(value: Any) -> str:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return str(parsed.get("action", "hold")).lower() if isinstance(parsed, dict) else "hold"
    except Exception:
        return "hold"


def action_value(action: str) -> float:
    return {"long": 1.0, "short": -1.0, "hold": 0.0}.get(str(action).lower(), 0.0)


def finite_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def normalize_utility(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if vals.nunique() <= 1:
        return pd.Series(0.5, index=vals.index, dtype=float)
    lo = float(vals.quantile(0.05))
    hi = float(vals.quantile(0.95))
    if abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=vals.index, dtype=float)
    return ((vals.clip(lo, hi) - lo) / (hi - lo)).clip(0.0, 1.0)


def normalize_score(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if vals.nunique() <= 1:
        return pd.Series(0.5, index=vals.index, dtype=float)
    lo = float(vals.quantile(0.05))
    hi = float(vals.quantile(0.95))
    if abs(hi - lo) < 1e-12:
        lo = float(vals.min())
        hi = float(vals.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=vals.index, dtype=float)
    return ((vals.clip(lo, hi) - lo) / (hi - lo)).clip(0.0, 1.0)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_alignment(df: pd.DataFrame, min_reward_gap: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    rwsft: list[dict[str, Any]] = []
    dpo: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        group = group.sort_values("final_reward", ascending=False).copy()
        valid = group[(group["schema_ok"].astype(bool)) & (~group["grounding_status"].isin(["contradiction", "unsupported"]))]
        if valid.empty:
            continue
        best = valid.iloc[0]
        rwsft.append(
            {
                "sample_id": sample_id,
                "split": best.get("split"),
                "prompt": best.get("prompt"),
                "output": best.get("raw_output", best.get("raw_text")),
                "reward": float(best["final_reward"]),
                "reward_source": best.get("reward_source"),
                "candidate_id": int(best.get("candidate_id", 0)),
            }
        )
        selected_rows.append(best.to_dict())
        if len(group) >= 2:
            rejected_pool = group[group["candidate_id"].astype(int) != int(best.get("candidate_id", 0))]
            if not rejected_pool.empty:
                rejected = rejected_pool.sort_values("final_reward", ascending=True).iloc[0]
                gap = float(best["final_reward"] - rejected["final_reward"])
                if gap >= min_reward_gap:
                    dpo.append(
                        {
                            "sample_id": sample_id,
                            "split": best.get("split"),
                            "prompt": best.get("prompt"),
                            "chosen": best.get("raw_output", best.get("raw_text")),
                            "rejected": rejected.get("raw_output", rejected.get("raw_text")),
                            "chosen_reward": float(best["final_reward"]),
                            "rejected_reward": float(rejected["final_reward"]),
                            "reward_gap": gap,
                            "chosen_candidate_id": int(best.get("candidate_id", 0)),
                            "rejected_candidate_id": int(rejected.get("candidate_id", 0)),
                            "reward_source": best.get("reward_source"),
                        }
                    )
    return rwsft, dpo, pd.DataFrame(selected_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--inferability", required=True)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--flow-metrics", required=True)
    parser.add_argument("--flow-scores", default=None)
    parser.add_argument("--tracks", default=None)
    parser.add_argument("--rwsft-output", required=True)
    parser.add_argument("--dpo-output", required=True)
    parser.add_argument("--scored-output", default=None)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default="outputs/manifests/16_ALIGNMENT_REBUILD_CLEAN_V4.manifest.json")
    parser.add_argument("--min-rwsft", type=int, default=1000)
    parser.add_argument("--min-dpo", type=int, default=300)
    parser.add_argument("--min-reward-gap", type=float, default=0.08)
    args = parser.parse_args()

    failures: list[str] = []
    for path in [args.rationales, args.inferability, args.grounding, args.flow_metrics]:
        if not os.path.exists(path):
            failures.append(f"missing input: {path}")
    if failures:
        write_status(args.status, STEP, "FAIL", [args.rationales, args.inferability, args.grounding, args.flow_metrics], [args.status], {}, failures, False)
        return 1

    rationales = pd.read_parquet(args.rationales)
    inferability = pd.read_parquet(args.inferability)
    grounding = pd.read_parquet(args.grounding)
    tracks = pd.read_parquet(args.tracks) if args.tracks and os.path.exists(args.tracks) else pd.DataFrame()
    flow_scores = pd.read_csv(args.flow_scores) if args.flow_scores and os.path.exists(args.flow_scores) else pd.DataFrame()
    flow_metrics = load_json(args.flow_metrics)
    flow_allowed = bool(flow_metrics.get("flow_reward_improvement") or flow_metrics.get("flow_claim_allowed"))

    df = rationales.copy()
    inf_cols = [
        col
        for col in [
            "sample_id",
            "candidate_id",
            "true_label_probability_debiased",
            "true_label_probability",
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
        t_cols = [col for col in ["sample_id", "target_return", "abnormal_return_h1", "mean_evidence_quality_score", "split", "track"] if col in tracks.columns]
        df = df.merge(tracks[t_cols].drop_duplicates("sample_id"), on="sample_id", how="left", suffixes=("", "_track"))
        if "split_track" in df.columns:
            df["split"] = df["split"].combine_first(df["split_track"]) if "split" in df.columns else df["split_track"]
        if "track_track" in df.columns:
            df["track"] = df["track"].combine_first(df["track_track"]) if "track" in df.columns else df["track_track"]

    true_prob = finite_series(df, "true_label_probability_debiased", np.nan)
    true_prob = true_prob.fillna(finite_series(df, "true_label_probability", 0.0))
    news_ground = finite_series(df, "news_grounding_score", 0.5)
    tech_ground = finite_series(df, "technical_grounding_score", 0.5)
    evidence_weight = finite_series(df, "mean_evidence_quality_score", 0.5).clip(0.0, 1.0)
    schema = df["schema_ok"].astype(float) if "schema_ok" in df.columns else pd.Series(0.0, index=df.index)
    returns = finite_series(df, "target_return", np.nan).fillna(finite_series(df, "abnormal_return_h1", 0.0))
    action = df.get("parsed_json", "").apply(parse_action)
    utility = action.apply(action_value) * returns - action.apply(action_value).abs() * 0.001
    utility_norm = normalize_utility(utility)

    df["proxy_reward"] = (
        0.35 * true_prob.clip(0.0, 1.0)
        + 0.20 * news_ground.clip(0.0, 1.0)
        + 0.20 * tech_ground.clip(0.0, 1.0)
        + 0.15 * evidence_weight
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
        df["reward_source"] = "flow_v4"
    elif flow_allowed:
        df["reward_source"] = "flow_v4_allowed_but_per_candidate_flow_score_unavailable_proxy_used"

    rwsft, dpo, selected = build_alignment(df, args.min_reward_gap)
    write_jsonl(args.rwsft_output, rwsft)
    write_jsonl(args.dpo_output, dpo)
    if args.scored_output:
        Path(args.scored_output).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.scored_output, index=False)

    train_only = True
    if rwsft or dpo:
        splits = {str(row.get("split")) for row in [*rwsft, *dpo]}
        train_only = splits <= {"train"}
    metrics = {
        "rwsft_examples": len(rwsft),
        "dpo_pairs": len(dpo),
        "unique_samples": int(df["sample_id"].nunique()) if len(df) else 0,
        "candidate_rows": int(len(df)),
        "mean_reward_selected": float(selected["final_reward"].mean()) if len(selected) else 0.0,
        "mean_chosen_reward": float(np.mean([row["chosen_reward"] for row in dpo])) if dpo else 0.0,
        "mean_rejected_reward": float(np.mean([row["rejected_reward"] for row in dpo])) if dpo else 0.0,
        "min_reward_gap": args.min_reward_gap,
        "reward_source": "flow_v4" if flow_allowed else "proxy_average_independent",
        "flow_reward_improvement": flow_allowed,
        "flow_scores_used": bool(flow_allowed and not flow_scores.empty and "flow_reward" in flow_scores.columns),
        "flow_reward_normalized_for_selection": bool(flow_allowed and not flow_scores.empty and "flow_reward" in flow_scores.columns),
        "train_only": bool(train_only),
        "rwsft_sha256": sha256_file(args.rwsft_output) if os.path.exists(args.rwsft_output) else None,
        "dpo_sha256": sha256_file(args.dpo_output) if os.path.exists(args.dpo_output) else None,
        "small_scale_thresholds": {"min_rwsft": args.min_rwsft, "min_dpo": args.min_dpo},
    }
    if len(rwsft) < args.min_rwsft:
        failures.append(f"RWSFT count {len(rwsft)} < {args.min_rwsft}")
    if len(dpo) < args.min_dpo:
        failures.append(f"DPO count {len(dpo)} < {args.min_dpo}")
    if dpo and metrics["mean_chosen_reward"] <= metrics["mean_rejected_reward"]:
        failures.append("chosen reward is not greater than rejected reward")
    if not train_only:
        failures.append("alignment data contains non-train rows")

    outputs = [args.rwsft_output, args.dpo_output, args.metrics]
    if args.scored_output:
        outputs.append(args.scored_output)
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, outputs, STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.rationales, args.inferability, args.grounding, args.flow_metrics]
        + ([args.flow_scores] if args.flow_scores else [])
        + ([args.tracks] if args.tracks else []),
        [*outputs, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
