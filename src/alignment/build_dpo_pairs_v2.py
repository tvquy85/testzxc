from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment.build_rwsft_dataset_v2 import attach_alignment_scores
from src.utils.artifacts import write_manifest, write_status


STEP = "14_RWSFT_DPO_DATASET_REBUILD"


def build_dpo_pairs(scored, min_gap: float = 0.02, max_pairs_per_sample: int = 3) -> list[dict]:
    pairs: list[dict] = []
    if scored.empty:
        return pairs
    valid = scored[(scored["split"] == "train") & scored["alignment_proxy_score"].notna()].copy()
    for sample_id, group in valid.groupby("sample_id"):
        group = group.sort_values("alignment_proxy_score")
        sample_pairs = []
        for lo_idx, hi_idx in itertools.combinations(range(len(group)), 2):
            lo = group.iloc[lo_idx]
            hi = group.iloc[hi_idx]
            gap = float(hi["alignment_proxy_score"] - lo["alignment_proxy_score"])
            if gap >= min_gap:
                prompt = hi.get("prompt") if isinstance(hi.get("prompt"), str) and hi.get("prompt").strip() else lo.get("prompt")
                sample_pairs.append(
                    {
                        "sample_id": sample_id,
                        "chosen_candidate_id": int(hi.get("candidate_id", 0)),
                        "rejected_candidate_id": int(lo.get("candidate_id", 0)),
                        "prompt": prompt,
                        "messages": [{"role": "user", "content": str(prompt)}] if isinstance(prompt, str) and prompt.strip() else [],
                        "chosen": hi.get("raw_text"),
                        "rejected": lo.get("raw_text"),
                        "chosen_score": float(hi["alignment_proxy_score"]),
                        "rejected_score": float(lo["alignment_proxy_score"]),
                        "reward_gap": gap,
                        "split": hi.get("split"),
                    }
                )
        sample_pairs.sort(key=lambda item: item["reward_gap"], reverse=True)
        pairs.extend(sample_pairs[:max_pairs_per_sample])
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", default="data/rationales/parsed/train_candidates_stage1_strict.parquet")
    parser.add_argument("--inferability", default="data/judges/inferability_multi_judge_stage1.parquet")
    parser.add_argument("--grounding", default="data/judges/claim_grounding_scores_stage1.parquet")
    parser.add_argument("--flow-checkpoint", default="checkpoints/flow_reward_v2_stage1")
    parser.add_argument("--output", default="data/alignment/dpo_pairs_train_v2.jsonl")
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
    pairs = build_dpo_pairs(scored, min_gap=args.min_gap, max_pairs_per_sample=args.max_pairs_per_sample)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    if any(pair["split"] != "train" for pair in pairs):
        failures.append("DPO pairs contain non-train split")
    if len(pairs) < 2000:
        failures.append("DPO pairs < 2,000 MVP gate")
    metrics = {
        "dpo_pairs": len(pairs),
        "dpo_unique_samples": len({pair["sample_id"] for pair in pairs}),
        "dpo_mvp_gate": len(pairs) >= 2000,
        "reward_gap_min": args.min_gap,
        "ranking_source": "alignment_proxy_from_step10_11_flow_targets",
    }
    write_manifest(args.manifest, [args.output], STEP, extra={"metrics": metrics})
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.rationales, args.inferability, args.grounding, args.flow_checkpoint],
        [args.output, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps(metrics, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
