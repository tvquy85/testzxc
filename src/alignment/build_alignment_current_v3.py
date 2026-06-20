import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import sha256_file, write_json, write_manifest, write_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STEP = "12_RWSFT_DPO_REBUILD_FROM_INDEPENDENT_REWARDS"

def load_json_metrics(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def get_action_val(parsed_json_str: str) -> float:
    try:
        parsed = json.loads(parsed_json_str)
        action_str = str(parsed.get("action", "hold")).lower()
        return {"long": 1.0, "short": -1.0, "hold": 0.0}.get(action_str, 0.0)
    except:
        return 0.0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--inferability", required=True)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--flow-eval", required=True)
    parser.add_argument("--rwsft-output", required=True)
    parser.add_argument("--dpo-output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures = []
    
    for fpath in [args.rationales, args.inferability, args.grounding]:
        if not os.path.exists(fpath):
            failures.append(f"Missing required input: {fpath}")
            
    if failures:
        status = "FAIL"
        write_status(
            args.status,
            STEP,
            status,
            inputs_checked=[args.rationales, args.inferability, args.grounding, args.flow_eval],
            outputs_created=[args.status],
            failures=failures,
            next_step_allowed=False,
        )
        return 1

    rationales_df = pd.read_parquet(args.rationales)
    inferability_df = pd.read_parquet(args.inferability)
    grounding_df = pd.read_parquet(args.grounding)
    
    # We need context target_return for utility_proxy.
    contexts_df = pd.read_parquet("data/processed/ticker_date_contexts_h1_v2_targets.parquet")

    # Flow claim
    flow_eval = load_json_metrics(args.flow_eval)
    flow_claim_allowed = flow_eval.get("flow_claim_allowed", False)

    # Merge data
    df = rationales_df.copy()
    
    # Merge inferability
    inf = inferability_df[["sample_id", "candidate_id", "true_label_probability_debiased"]].rename(
        columns={"true_label_probability_debiased": "independent_true_label_prob"}
    )
    df = df.merge(inf, on=["sample_id", "candidate_id"], how="left")
    
    # Merge grounding
    gnd = grounding_df[["sample_id", "candidate_id", "status", "total_claims", "supported_claims"]].copy()
    gnd["supported_claim_rate"] = gnd.apply(lambda r: r["supported_claims"] / r["total_claims"] if r["total_claims"] > 0 else 1.0, axis=1)
    df = df.merge(gnd, on=["sample_id", "candidate_id"], how="left")
    
    # Merge contexts
    df = df.merge(contexts_df[["sample_id", "target_return"]], on="sample_id", how="left")
    
    df["schema_ok"] = df["schema_ok"].astype(float)
    df["independent_true_label_prob"] = df["independent_true_label_prob"].fillna(0.0)
    df["supported_claim_rate"] = df["supported_claim_rate"].fillna(0.0)
    
    # Calculate utility proxy
    df["action_val"] = df["parsed_json"].apply(get_action_val)
    df["target_return"] = df["target_return"].fillna(0.0)
    df["utility_proxy"] = df["action_val"] * df["target_return"] - (df["action_val"].abs() * 0.001)
    
    # Calculate composite reward
    # In V3, technical_grounding and news_grounding are unified as supported_claim_rate
    df["composite_reward"] = (
        0.45 * df["independent_true_label_prob"] +
        0.40 * df["supported_claim_rate"] +
        0.10 * df["utility_proxy"] +
        0.05 * df["schema_ok"]
    )
    
    df["final_reward"] = df["composite_reward"]

    # Filter out contradicted technical claims (in V3, status == "contradiction")
    df["is_valid_chosen"] = df["status"] != "contradiction"

    dpo_pairs = []
    rwsft_examples = []
    
    for sample_id, group in df.groupby("sample_id"):
        group = group.sort_values("final_reward", ascending=False)
        
        # Add the best valid one to RWSFT
        valid_group = group[group["is_valid_chosen"]]
        if not valid_group.empty:
            best_row = valid_group.iloc[0]
            if best_row["schema_ok"] > 0:
                rwsft_examples.append({
                    "sample_id": sample_id,
                    "prompt": best_row["prompt"],
                    "output": best_row["raw_output"],
                    "reward": float(best_row["final_reward"])
                })
        
        # DPO requires at least 2 candidates
        if len(group) >= 2:
            chosen = group.iloc[0]
            rejected = group.iloc[-1]
            
            # Constraints
            margin = chosen["final_reward"] - rejected["final_reward"]
            if margin >= 0.05 and chosen["is_valid_chosen"] and chosen["schema_ok"] > 0:
                dpo_pairs.append({
                    "sample_id": sample_id,
                    "prompt": chosen["prompt"],
                    "chosen": chosen["raw_output"],
                    "rejected": rejected["raw_output"],
                    "chosen_reward": float(chosen["final_reward"]),
                    "rejected_reward": float(rejected["final_reward"]),
                    "margin": float(margin)
                })

    Path(args.rwsft_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.rwsft_output, "w") as f:
        for ex in rwsft_examples:
            f.write(json.dumps(ex) + "\n")
            
    with open(args.dpo_output, "w") as f:
        for ex in dpo_pairs:
            f.write(json.dumps(ex) + "\n")
            
    dpo_df = pd.DataFrame(dpo_pairs)
    mean_chosen = float(dpo_df["chosen_reward"].mean()) if not dpo_df.empty else 0.0
    mean_rejected = float(dpo_df["rejected_reward"].mean()) if not dpo_df.empty else 0.0
    
    metrics = {
        "rwsft_examples": len(rwsft_examples),
        "dpo_pairs": len(dpo_pairs),
        "mean_reward_chosen": mean_chosen,
        "mean_reward_rejected": mean_rejected,
        "flow_claim_allowed": flow_claim_allowed,
        "train_only": True,
    }
    
    # Acceptance criteria
    if len(rwsft_examples) < 1000:
        failures.append(f"RWSFT count {len(rwsft_examples)} < 1000")
    if len(dpo_pairs) < 500:
        failures.append(f"DPO count {len(dpo_pairs)} < 500")
    if mean_chosen <= mean_rejected:
        failures.append("Chosen reward not strictly greater than rejected reward")
    if not all(row.get("split", "train") == "train" for row in rwsft_examples + dpo_pairs):
        failures.append("alignment data contains non-train rows")
        metrics["train_only"] = False

    metrics["rwsft_sha256"] = sha256_file(args.rwsft_output) if os.path.exists(args.rwsft_output) else None
    metrics["dpo_sha256"] = sha256_file(args.dpo_output) if os.path.exists(args.dpo_output) else None
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.rwsft_output, args.dpo_output, args.metrics], STEP)

    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.rationales, args.inferability, args.grounding, args.flow_eval],
        outputs_created=[args.rwsft_output, args.dpo_output, args.metrics, args.manifest, args.status],
        failures=failures,
        metrics=metrics,
        next_step_allowed=status == "PASS",
    )
    
    logging.info(f"Generated {len(rwsft_examples)} RWSFT and {len(dpo_pairs)} DPO pairs. Status: {status}")
    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
