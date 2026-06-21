# 16 — Alignment Dataset Rebuild V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Build RWSFT/DPO pairs from clean V4 rationales and independent rewards.

## Inputs
- `data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet`
- `outputs/metrics/flow_vs_proxy_clean_v4.json`
- `data/judges/independent_inferability_evidence_v4.parquet`
- `data/judges/claim_grounding_evidence_v4.parquet`

## Outputs
- `data/alignment/rwsft_current_clean_v4.jsonl`
- `data/alignment/dpo_current_clean_v4.jsonl`
- `outputs/metrics/alignment_dataset_current_clean_v4.json`
- `outputs/status/16_ALIGNMENT_REBUILD_CLEAN_V4.status.json`

## Reward source rule
Use flow only if `flow_reward_improvement=true`; otherwise use proxy-average / independent reward.

## Pair rule
- chosen = highest reward candidate;
- rejected = lowest reward candidate;
- reward gap >= 0.08;
- no val/test rows;
- chosen must pass schema and not fail grounding.

## Command
```bash
python -m src.alignment.build_alignment_current_v4 \
  --rationales data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet \
  --inferability data/judges/independent_inferability_evidence_v4.parquet \
  --grounding data/judges/claim_grounding_evidence_v4.parquet \
  --flow-metrics outputs/metrics/flow_vs_proxy_clean_v4.json \
  --rwsft-output data/alignment/rwsft_current_clean_v4.jsonl \
  --dpo-output data/alignment/dpo_current_clean_v4.jsonl \
  --metrics outputs/metrics/alignment_dataset_current_clean_v4.json \
  --status outputs/status/16_ALIGNMENT_REBUILD_CLEAN_V4.status.json
```

## Gates
RWSFT >= 1000 rows; DPO >= 300 pairs; chosen reward > rejected reward + 0.05; no non-train rows.
