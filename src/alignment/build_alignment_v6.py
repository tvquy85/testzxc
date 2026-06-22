from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import sha256_file, write_json, write_manifest, write_status

STEP = "12_ALIGNMENT_PAIRS_MIN_GAP_AND_DIVERSITY"
LABELS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
PROB_COLS = [f"p_{label}" for label in LABELS]
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
SEMANTIC_STOP_TOKENS = {
    "action",
    "and",
    "confidence",
    "direction",
    "evidence_id",
    "factor",
    "forecast_distribution",
    "for",
    "from",
    "hold",
    "into",
    "key_risks",
    "long",
    "mild_down",
    "mild_up",
    "moderate",
    "neutral",
    "news",
    "news_rationale",
    "opposing",
    "over",
    "rationale",
    "risk_checks",
    "short",
    "signal",
    "signal_id",
    "strength",
    "strong",
    "strong_down",
    "strong_up",
    "supporting",
    "technical",
    "technical_rationale",
    "that",
    "the",
    "this",
    "under",
    "value",
    "weak",
    "with",
}


def load_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def semantic_tokens(text: Any) -> set[str]:
    tokens = set()
    for token in TOKEN_RE.findall(safe_text(text)):
        norm = token.lower()
        if len(norm) < 3 or norm in SEMANTIC_STOP_TOKENS:
            continue
        tokens.add(norm)
    return tokens


def semantic_distance(left: Any, right: Any) -> float:
    left_tokens = semantic_tokens(left)
    right_tokens = semantic_tokens(right)
    if not left_tokens and not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return 1.0 - (len(left_tokens & right_tokens) / len(union))


def accept_pair(
    chosen_reward: float,
    rejected_reward: float,
    distance: float,
    min_reward_gap: float = 0.05,
    min_semantic_distance: float = 0.15,
) -> bool:
    return (float(chosen_reward) - float(rejected_reward)) >= min_reward_gap and float(distance) >= min_semantic_distance


def normalize_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PROB_COLS + [
        "true_label_probability_ensemble",
        "argmax_consistency_ensemble",
        "label_order_kl_mean",
        "judge_disagreement_entropy",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def merge_inputs(rationales: pd.DataFrame, judge: pd.DataFrame) -> pd.DataFrame:
    left = rationales.copy()
    right = judge.copy()
    left["candidate_id"] = pd.to_numeric(left["candidate_id"], errors="coerce").astype("Int64")
    right["candidate_id"] = pd.to_numeric(right["candidate_id"], errors="coerce").astype("Int64")
    merged = left.merge(right, on=["sample_id", "candidate_id"], how="inner", suffixes=("", "_judge"))
    return normalize_probability_columns(merged)


def add_proxy_reward(df: pd.DataFrame, flow_metrics: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    flow_allowed = bool(flow_metrics.get("flow_claim_allowed")) and bool(flow_metrics.get("flow_reward_improvement"))
    if flow_allowed:
        out["reward_source"] = "flow_v6_allowed_but_per_candidate_scores_unavailable_proxy_used"
    else:
        out["reward_source"] = "proxy_true_label_probability_ensemble"
    out["final_reward"] = pd.to_numeric(out["true_label_probability_ensemble"], errors="coerce")
    return out


def valid_candidates(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.copy()
    valid = valid[valid["parse_ok"].astype(bool) & valid["schema_ok"].astype(bool)].copy()
    if "judge_schema_ok" in valid.columns:
        valid = valid[valid["judge_schema_ok"].astype(bool)].copy()
    for col in PROB_COLS:
        valid = valid[np.isfinite(pd.to_numeric(valid[col], errors="coerce"))].copy()
    prob_sum = valid[PROB_COLS].sum(axis=1)
    valid = valid[prob_sum.between(0.999, 1.001)].copy()
    valid = valid[np.isfinite(pd.to_numeric(valid["final_reward"], errors="coerce"))].copy()
    if "split" in valid.columns:
        valid = valid[valid["split"].astype(str).eq("train")].copy()
    valid["candidate_id"] = valid["candidate_id"].astype(int)
    valid["final_reward"] = valid["final_reward"].astype(float)
    return valid.sort_values(
        ["final_reward", "argmax_consistency_ensemble", "sample_id", "candidate_id"],
        ascending=[False, False, True, True],
    ).copy()


def row_output(row: pd.Series) -> str:
    output = safe_text(row.get("raw_output"))
    if output:
        return output
    return safe_text(row.get("raw_text"))


def build_rwsft_records(valid: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in valid.iterrows():
        prompt = safe_text(row.get("prompt"))
        output = row_output(row)
        rows.append(
            {
                "sample_id": safe_text(row.get("sample_id")),
                "split": safe_text(row.get("split", "train")),
                "prompt": prompt,
                "output": output,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                ],
                "candidate_id": int(row.get("candidate_id", 0)),
                "reward": float(row["final_reward"]),
                "reward_source": safe_text(row.get("reward_source")),
                "target_label_5": safe_text(row.get("target_label_5")),
                "argmax_consistency_ensemble": float(row.get("argmax_consistency_ensemble", 0.0)),
            }
        )
    return rows


def build_dpo_records(
    valid: pd.DataFrame,
    min_reward_gap: float,
    min_semantic_distance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = valid.sort_values(
        ["sample_id", "final_reward", "argmax_consistency_ensemble", "candidate_id"],
        ascending=[True, False, False, True],
    ).copy()
    for sample_id, group in ordered.groupby("sample_id", sort=False):
        if len(group) < 2:
            continue
        chosen = group.iloc[0]
        chosen_output = row_output(chosen)
        for _, rejected in group.iloc[1:].iterrows():
            rejected_output = row_output(rejected)
            distance = semantic_distance(chosen_output, rejected_output)
            if not accept_pair(chosen["final_reward"], rejected["final_reward"], distance, min_reward_gap, min_semantic_distance):
                continue
            rows.append(
                {
                    "sample_id": safe_text(sample_id),
                    "split": safe_text(chosen.get("split", "train")),
                    "prompt": safe_text(chosen.get("prompt")),
                    "chosen": chosen_output,
                    "rejected": rejected_output,
                    "chosen_candidate_id": int(chosen.get("candidate_id", 0)),
                    "rejected_candidate_id": int(rejected.get("candidate_id", 0)),
                    "chosen_reward": float(chosen["final_reward"]),
                    "rejected_reward": float(rejected["final_reward"]),
                    "reward_gap": float(chosen["final_reward"] - rejected["final_reward"]),
                    "semantic_distance": float(distance),
                    "reward_source": safe_text(chosen.get("reward_source")),
                    "target_label_5": safe_text(chosen.get("target_label_5")),
                    "chosen_argmax_consistency_ensemble": float(chosen.get("argmax_consistency_ensemble", 0.0)),
                    "rejected_argmax_consistency_ensemble": float(rejected.get("argmax_consistency_ensemble", 0.0)),
                }
            )
    return rows


def split_is_train_only(rows: list[dict[str, Any]]) -> bool:
    return {safe_text(row.get("split")) for row in rows} <= {"train"} if rows else True


def metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--flow-metrics", required=True)
    parser.add_argument("--rwsft-output", required=True)
    parser.add_argument("--dpo-output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--min-rwsft", type=int, default=1000)
    parser.add_argument("--min-dpo", type=int, default=300)
    parser.add_argument("--min-reward-gap", type=float, default=0.05)
    parser.add_argument("--min-semantic-distance", type=float, default=0.15)
    parser.add_argument("--sample-limit", type=int, default=50)
    args = parser.parse_args()

    failures: list[str] = []
    inputs = [args.rationales, args.judge, args.flow_metrics]
    for path in inputs:
        if not os.path.exists(path):
            failures.append(f"missing input: {path}")
    if failures:
        write_status(args.status, STEP, "FAIL", inputs, [args.status], {}, failures, False)
        return 1

    rationales = pd.read_parquet(args.rationales)
    judge = pd.read_parquet(args.judge)
    flow_metrics = load_json(args.flow_metrics)
    df = add_proxy_reward(merge_inputs(rationales, judge), flow_metrics)
    valid = valid_candidates(df)
    rwsft = build_rwsft_records(valid)
    dpo = build_dpo_records(valid, args.min_reward_gap, args.min_semantic_distance)

    write_jsonl(args.rwsft_output, rwsft)
    write_jsonl(args.dpo_output, dpo)
    write_jsonl(args.samples, dpo[: args.sample_limit])

    train_only = split_is_train_only([*rwsft, *dpo])
    flow_claim_allowed = bool(flow_metrics.get("flow_claim_allowed"))
    flow_reward_improvement = bool(flow_metrics.get("flow_reward_improvement"))
    reward_source = "proxy_true_label_probability_ensemble"
    metrics = {
        "pipeline_pass": True,
        "claim_allowed": False,
        "rationale_rows": int(len(rationales)),
        "judge_rows": int(len(judge)),
        "merged_rows": int(len(df)),
        "valid_candidate_rows": int(len(valid)),
        "valid_unique_samples": int(valid["sample_id"].nunique()) if len(valid) else 0,
        "rwsft_examples": len(rwsft),
        "dpo_pairs": len(dpo),
        "dpo_unique_samples": len({row["sample_id"] for row in dpo}),
        "reward_source": reward_source,
        "flow_claim_allowed": flow_claim_allowed,
        "flow_reward_improvement": flow_reward_improvement,
        "flow_scores_used": False,
        "flow_fallback_reason": "flow_v6_failed_or_claim_blocked",
        "chosen_reward_mean": metric_mean(dpo, "chosen_reward"),
        "rejected_reward_mean": metric_mean(dpo, "rejected_reward"),
        "mean_chosen_reward": metric_mean(dpo, "chosen_reward"),
        "mean_rejected_reward": metric_mean(dpo, "rejected_reward"),
        "mean_reward_gap": metric_mean(dpo, "reward_gap"),
        "min_observed_reward_gap": float(min((row["reward_gap"] for row in dpo), default=0.0)),
        "mean_semantic_distance": metric_mean(dpo, "semantic_distance"),
        "min_observed_semantic_distance": float(min((row["semantic_distance"] for row in dpo), default=0.0)),
        "min_reward_gap": args.min_reward_gap,
        "min_semantic_distance": args.min_semantic_distance,
        "semantic_distance": "jaccard_over_content_tokens_excluding_json_schema_tokens",
        "train_only": bool(train_only),
        "rwsft_sha256": sha256_file(args.rwsft_output) if os.path.exists(args.rwsft_output) else None,
        "dpo_sha256": sha256_file(args.dpo_output) if os.path.exists(args.dpo_output) else None,
        "samples_sha256": sha256_file(args.samples) if os.path.exists(args.samples) else None,
    }
    if len(rwsft) < args.min_rwsft:
        failures.append(f"RWSFT count {len(rwsft)} < {args.min_rwsft}")
    if len(dpo) < args.min_dpo:
        failures.append(f"DPO count {len(dpo)} < {args.min_dpo}")
    if dpo and metrics["mean_reward_gap"] < 0.035:
        failures.append(f"mean reward gap {metrics['mean_reward_gap']:.6f} < 0.035 diagnostic gate")
    if dpo and metrics["min_observed_reward_gap"] + 1e-12 < args.min_reward_gap:
        failures.append("DPO pair below minimum reward gap")
    if dpo and metrics["min_observed_semantic_distance"] + 1e-12 < args.min_semantic_distance:
        failures.append("DPO pair below minimum semantic distance")
    if dpo and metrics["mean_chosen_reward"] <= metrics["mean_rejected_reward"]:
        failures.append("mean chosen reward is not greater than mean rejected reward")
    if not train_only:
        failures.append("alignment data contains non-train rows")
    if flow_claim_allowed and flow_reward_improvement:
        failures.append("Flow V6 is allowed but Step 12 implementation has no per-candidate flow score input")

    outputs = [args.rwsft_output, args.dpo_output, args.metrics, args.samples]
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, outputs, STEP)
    outputs_with_status = [*outputs, args.manifest, args.status]
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, inputs, outputs_with_status, metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
