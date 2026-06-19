from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment.build_dpo_pairs_v2 import build_dpo_pairs
from src.alignment.build_rwsft_dataset_v2 import attach_alignment_scores, build_rwsft_records
from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "14_RWSFT_DPO_DATASET_REBUILD"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", default="data/rationales/parsed/train_candidates_stage1_strict.parquet")
    parser.add_argument("--inferability", default="data/judges/inferability_multi_judge_stage1.parquet")
    parser.add_argument("--grounding", default="data/judges/claim_grounding_scores_stage1.parquet")
    parser.add_argument("--flow-checkpoint", default="checkpoints/flow_reward_v2_stage1")
    parser.add_argument("--rwsft-output", default="data/alignment/rwsft_train_v2.jsonl")
    parser.add_argument("--dpo-output", default="data/alignment/dpo_pairs_train_v2.jsonl")
    parser.add_argument("--summary", default="outputs/metrics/alignment_dataset_v2_summary.json")
    parser.add_argument("--min-gap", type=float, default=0.02)
    parser.add_argument("--max-pairs-per-sample", type=int, default=3)
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if Path(args.rationales).exists() else pd.DataFrame()
    inferability = pd.read_parquet(args.inferability) if Path(args.inferability).exists() else pd.DataFrame()
    grounding = pd.read_parquet(args.grounding) if Path(args.grounding).exists() else pd.DataFrame()
    if rationales.empty:
        failures.append("rationales input is empty")
    if "split" in rationales.columns and set(rationales["split"].dropna()) - {"train"}:
        failures.append("non-train rows found in rationales")
    scored = attach_alignment_scores(rationales, inferability, grounding)
    rwsft_records = build_rwsft_records(scored)
    dpo_pairs = build_dpo_pairs(scored, min_gap=args.min_gap, max_pairs_per_sample=args.max_pairs_per_sample)
    for path, rows in [(args.rwsft_output, rwsft_records), (args.dpo_output, dpo_pairs)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if len(rwsft_records) < 5000:
        failures.append("RWSFT examples < 5,000 MVP gate")
    if len(dpo_pairs) < 2000:
        failures.append("DPO pairs < 2,000 MVP gate")
    if any(row.get("split") != "train" for row in rwsft_records + dpo_pairs):
        failures.append("alignment datasets contain non-train split")
    summary = {
        "rwsft_examples": len(rwsft_records),
        "rwsft_mvp_gate": len(rwsft_records) >= 5000,
        "dpo_pairs": len(dpo_pairs),
        "dpo_unique_samples": len({pair["sample_id"] for pair in dpo_pairs}),
        "dpo_mvp_gate": len(dpo_pairs) >= 2000,
        "reward_gap_min": args.min_gap,
        "ranking_source": "alignment_proxy_from_step10_11_flow_targets",
        "scored_rows": int(scored["alignment_proxy_score"].notna().sum()) if len(scored) else 0,
    }
    write_json(args.summary, summary)
    write_manifest(args.manifest, [args.rwsft_output, args.dpo_output, args.summary], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.rationales, args.inferability, args.grounding, args.flow_checkpoint],
        [args.rwsft_output, args.dpo_output, args.summary, args.manifest, args.status],
        summary,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "summary": summary, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
