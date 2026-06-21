# 11 — Flow Target Normalization and Split

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Build a normalized Flow V5 dataset with semantic conditions and true train/val split.

## Why this is needed
Flow V4 used mixed-scale targets and tiny train-only data. Medium validation needs normalized targets and val split.

## Files to create or modify
Create `src/reward/build_flow_dataset_v5.py`. Save raw utility separately for evaluation.

## Inputs
```text
data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet
outputs/judges/medium_clean_v4_inferability_debiased.parquet
outputs/judges/medium_clean_v4_claim_grounding.parquet
data/processed/medium_clean_v4_contexts_gated.parquet
data/features/medium_clean_v4_semantic_embeddings_v5.npy
data/features/medium_clean_v4_semantic_embeddings_index_v5.parquet
```

## Outputs
```text
data/reward/medium_clean_v4_flow_dataset_v5.pt
outputs/metrics/11_flow_dataset_v5.json
outputs/status/11_FLOW_TARGET_NORMALIZATION_AND_SPLIT.status.json
```

## Commands
```bash
python -m src.reward.build_flow_dataset_v5 \
  --rationales data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet \
  --inferability outputs/judges/medium_clean_v4_inferability_debiased.parquet \
  --grounding outputs/judges/medium_clean_v4_claim_grounding.parquet \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --embeddings-npy data/features/medium_clean_v4_semantic_embeddings_v5.npy \
  --embeddings-index data/features/medium_clean_v4_semantic_embeddings_index_v5.parquet \
  --output data/reward/medium_clean_v4_flow_dataset_v5.pt \
  --metrics outputs/metrics/11_flow_dataset_v5.json \
  --status outputs/status/11_FLOW_TARGET_NORMALIZATION_AND_SPLIT.status.json
```

## Verification
```bash
python - <<'PY'
import torch, numpy as np, json
D=torch.load('data/reward/medium_clean_v4_flow_dataset_v5.pt', map_location='cpu', weights_only=False)
assert D['target'].shape[0] >= 1000
assert D['cond'].shape[0] == D['target'].shape[0]
assert D['cond'].shape[1] >= 128
assert {'train','val'}.issubset(set(D['split']))
assert np.isfinite(D['target']).all()
print('PASS flow dataset')
PY
```

## Acceptance criteria
- >= 1000 rows.
- True train/val split exists.
- Target mask coverage >= 0.70.
- Utility target is winsorized/min-max normalized on train only.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "11_FLOW_TARGET_NORMALIZATION_AND_SPLIT",
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
