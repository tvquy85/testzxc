# 10 — Flow Semantic Embeddings V5

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Build semantic condition embeddings for Flow V5, replacing hash-only condition vectors.

## Why this is needed
Flow V4 uses hash embeddings. The Policy reference uses context/explanation conditioning; semantic embeddings are a practical medium-scale upgrade.

## Files to create or modify
Create `src/features/build_semantic_embeddings_v5.py`. Use `sentence-transformers/all-MiniLM-L6-v2` first; fall back to FinBERT only if configured.

## Inputs
```text
data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet
data/processed/medium_clean_v4_contexts_gated.parquet
configs/default_paths.yaml
```

## Outputs
```text
data/features/medium_clean_v4_semantic_embeddings_v5.npy
data/features/medium_clean_v4_semantic_embeddings_index_v5.parquet
outputs/metrics/10_flow_semantic_embeddings_v5.json
outputs/status/10_FLOW_SEMANTIC_EMBEDDINGS_V5.status.json
```

## Commands
```bash
python -m src.features.build_semantic_embeddings_v5 \
  --rationales data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --model-key minilm \
  --config configs/default_paths.yaml \
  --output-npy data/features/medium_clean_v4_semantic_embeddings_v5.npy \
  --output-index data/features/medium_clean_v4_semantic_embeddings_index_v5.parquet \
  --metrics outputs/metrics/10_flow_semantic_embeddings_v5.json \
  --status outputs/status/10_FLOW_SEMANTIC_EMBEDDINGS_V5.status.json
```

## Verification
```bash
python - <<'PY'
import numpy as np, pandas as pd, json
X=np.load('data/features/medium_clean_v4_semantic_embeddings_v5.npy')
idx=pd.read_parquet('data/features/medium_clean_v4_semantic_embeddings_index_v5.parquet')
assert len(X)==len(idx)
assert X.shape[1] >= 128
assert np.isfinite(X).all()
print('PASS semantic embeddings', X.shape)
PY
```

## Acceptance criteria
- Embeddings exist for every candidate.
- No NaN/Inf.
- Backend recorded; should not be hash-only.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "10_FLOW_SEMANTIC_EMBEDDINGS_V5",
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
