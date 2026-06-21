# 04 — Evidence Pack Quality Gates V4 1

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Add medium-quality track flags for hard events, soft recommendation-style news, technical-dominant contexts, and context-only evidence.

## Why this is needed
Clean V4 improved `has_company_event_news`, but samples can still contain Zacks/listicle/recommendation evidence that may not behave like hard company events.

## Files to create or modify

Create `src/data/gate_evidence_pack_quality_v4_1.py`.

Suggested keyword sets:
```python
HARD_EVENT_TERMS=['earnings','eps','revenue','guidance','margin','buyback','dividend','upgrade','downgrade','price target','merger','acquisition','lawsuit','regulatory','recall']
SOFT_RECOMMENDATION_TERMS=['zacks rank','top stock','looks ripe','why we sold','best stocks','watchlist','what you need to know']
```


## Inputs
```text
data/processed/medium_clean_v4_contexts.parquet
```

## Outputs
```text
data/processed/medium_clean_v4_contexts_gated.parquet
outputs/metrics/04_evidence_pack_quality_gates_v4_1.json
outputs/status/04_EVIDENCE_PACK_QUALITY_GATES_V4_1.status.json
```

## Commands
```bash
python -m src.data.gate_evidence_pack_quality_v4_1 \
  --input data/processed/medium_clean_v4_contexts.parquet \
  --output data/processed/medium_clean_v4_contexts_gated.parquet \
  --metrics outputs/metrics/04_evidence_pack_quality_gates_v4_1.json \
  --status outputs/status/04_EVIDENCE_PACK_QUALITY_GATES_V4_1.status.json
```

## Verification
```bash
python - <<'PY'
import pandas as pd, json
x=pd.read_parquet('data/processed/medium_clean_v4_contexts_gated.parquet')
assert 'news_reasoning_track' in x.columns
m=json.load(open('outputs/metrics/04_evidence_pack_quality_gates_v4_1.json'))
assert 'track_distribution' in m
print('PASS evidence gates')
PY
```

## Acceptance criteria
- Every row has `news_reasoning_track`.
- No row is silently dropped; use weights/tracks first.
- Distribution is reported for hard-event, soft-news, technical-dominant, context-only.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "04_EVIDENCE_PACK_QUALITY_GATES_V4_1",
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
