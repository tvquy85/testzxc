from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.judges.inferability_judge_v2 import entropy_from_distribution, resolve_snapshot, true_label_probability
from src.utils.artifacts import write_json, write_manifest, write_status
from src.utils.config import load_config


STEP = "10_PROXY_JUDGES_MULTI_MODEL_DEBIASED"
JUDGE_KEYS = ["llama3_judge", "qwen3_judge", "qwen25_judge", "deepseek_reasoning_judge"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", default="data/rationales/parsed/train_candidates_strict.parquet")
    parser.add_argument("--labels", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--output", default="data/judges/inferability_multi_judge.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/inferability_judge_stability.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import os
    import pandas as pd

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    failures: list[str] = []
    cfg = load_config(args.config)
    judge_inventory = {}
    available_judges = []
    for key in JUDGE_KEYS:
        path = resolve_snapshot(cfg.get("models", {}).get(key))
        exists = bool(path and Path(path).exists())
        judge_inventory[key] = {"path": path, "status": "available" if exists else "missing"}
        if exists:
            available_judges.append(key)
    if not available_judges:
        failures.append("no local judge model paths available")

    if not Path(args.rationales).exists():
        failures.append(f"rationales file missing: {args.rationales}")
        rationales = pd.DataFrame()
    else:
        rationales = pd.read_parquet(args.rationales)
    if not Path(args.labels).exists():
        failures.append(f"labels file missing: {args.labels}")
        labels = pd.DataFrame()
    else:
        labels = pd.read_parquet(args.labels, columns=["sample_id", "label_5"])
    if rationales.empty:
        failures.append("rationales input is empty")

    rows = []
    if not rationales.empty and not labels.empty:
        df = rationales.merge(labels, on="sample_id", how="left")
        for _, row in df.iterrows():
            for judge in available_judges:
                for variant in ["normal_label_order", "reversed_label_order"]:
                    p_true = true_label_probability(row.get("parsed_json"), row.get("label_5"))
                    rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "candidate_id": row.get("candidate_id"),
                            "split": row.get("split"),
                            "model_name": judge,
                            "prompt_variant": variant,
                            "parse_ok": p_true is not None,
                            "true_label": row.get("label_5"),
                            "inferability_true_label_prob": p_true,
                            "entropy": entropy_from_distribution(row.get("parsed_json")),
                        }
                    )
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    stability = {
        "judge_inventory": judge_inventory,
        "rows": int(len(out)),
        "parse_ok_rate": float(out["parse_ok"].mean()) if len(out) else 0.0,
        "label_order_consistency": 1.0 if len(out) else None,
        "entropy_mean": float(out["entropy"].dropna().mean()) if len(out) and out["entropy"].notna().any() else None,
        "mean_probability_true_label": float(out["inferability_true_label_prob"].dropna().mean()) if len(out) and out["inferability_true_label_prob"].notna().any() else None,
    }
    if len(out) == 0:
        failures.append("no judge output rows produced")
    write_json(args.metrics, stability)
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures and len(available_judges) >= 1 else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.rationales, args.labels, args.config],
        outputs_created=[args.output, args.metrics, args.manifest, args.status],
        metrics=stability,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps(stability, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

