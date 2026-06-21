# 13 — Alignment Dataset Medium RWSFT DPO

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Build medium RWSFT and DPO datasets using Flow V5 if allowed or proxy fallback.

## Why this is needed
Alignment needs enough examples and preferred/rejected pairs. V4 small examples were too few.

## Files to create or modify
Create `src/alignment/build_alignment_medium_v5.py` or extend V4 builder without breaking old outputs.

## Inputs
```text
data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet
outputs/judges/medium_clean_v4_inferability_debiased.parquet
outputs/judges/medium_clean_v4_claim_grounding.parquet
outputs/tables/medium_flow_vs_proxy.csv
outputs/metrics/12_flow_vs_proxy_medium.json
```

## Outputs
```text
data/alignment/medium_clean_v4_rwsft.jsonl
data/alignment/medium_clean_v4_dpo_pairs.jsonl
outputs/metrics/13_alignment_dataset_medium.json
outputs/status/13_ALIGNMENT_DATASET_MEDIUM_RWSFT_DPO.status.json
```

## Commands
```bash
python -m src.alignment.build_alignment_medium_v5 \
  --rationales data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet \
  --inferability outputs/judges/medium_clean_v4_inferability_debiased.parquet \
  --grounding outputs/judges/medium_clean_v4_claim_grounding.parquet \
  --flow-table outputs/tables/medium_flow_vs_proxy.csv \
  --flow-metrics outputs/metrics/12_flow_vs_proxy_medium.json \
  --rwsft-output data/alignment/medium_clean_v4_rwsft.jsonl \
  --dpo-output data/alignment/medium_clean_v4_dpo_pairs.jsonl \
  --metrics outputs/metrics/13_alignment_dataset_medium.json \
  --status outputs/status/13_ALIGNMENT_DATASET_MEDIUM_RWSFT_DPO.status.json
```

## Verification
```bash
python - <<'PY'
from pathlib import Path
import json
r=Path('data/alignment/medium_clean_v4_rwsft.jsonl')
d=Path('data/alignment/medium_clean_v4_dpo_pairs.jsonl')
assert sum(1 for _ in r.open()) >= 1000
assert sum(1 for _ in d.open()) >= 300
m=json.load(open('outputs/metrics/13_alignment_dataset_medium.json'))
assert m['chosen_reward_mean'] > m['rejected_reward_mean']
print('PASS alignment data')
PY
```

## Acceptance criteria
- RWSFT >= 1000.
- DPO pairs >= 300.
- Chosen mean reward > rejected mean reward by >= 0.03.
- Reward source recorded.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "13_ALIGNMENT_DATASET_MEDIUM_RWSFT_DPO",
  "status": "PASS|FAIL",
  "pipeline_pass": true,
  "claim_allowed": false,
  "inputs": [],
  "outputs": [],
  "metrics": {},
  "failures": [],
  "warnings": []
}
```
