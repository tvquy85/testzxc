# 09 — Grounding News Negative Fix

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Audit and repair weak negative-news grounding and counterfactual readiness.

## Why this is needed
Clean V4 improved overall grounding but negative-news perturbation remained weak. We need to know whether negative evidence exists, is cited, and is directionally coherent.

## Files to create or modify
Create `src/judges/news_evidence_direction_audit_v5.py` or extend `claim_level_grounding_v4.py`.

## Inputs
```text
data/processed/medium_clean_v4_contexts_gated.parquet
data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet
```

## Outputs
```text
outputs/judges/medium_clean_v4_claim_grounding.parquet
outputs/metrics/09_grounding_news_negative_fix.json
review_samples/medium_clean_v4/news_negative_evidence_failures.jsonl
outputs/status/09_GROUNDING_NEWS_NEGATIVE_FIX.status.json
```

## Commands
```bash
python -m src.judges.news_evidence_direction_audit_v5 \
  --rationales data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --output outputs/judges/medium_clean_v4_claim_grounding.parquet \
  --metrics outputs/metrics/09_grounding_news_negative_fix.json \
  --failures-output review_samples/medium_clean_v4/news_negative_evidence_failures.jsonl \
  --status outputs/status/09_GROUNDING_NEWS_NEGATIVE_FIX.status.json
```

## Verification
```bash
python - <<'PY'
import pandas as pd, json
x=pd.read_parquet('outputs/judges/medium_clean_v4_claim_grounding.parquet')
assert {'sample_id','candidate_id'}.issubset(x.columns)
m=json.load(open('outputs/metrics/09_grounding_news_negative_fix.json'))
assert 'negative_news_claim_count' in m
assert m.get('unsupported_news_claim_rate',1) <= 0.30
print('PASS grounding')
PY
```

## Acceptance criteria
- Unsupported news claim rate <= 0.30.
- Negative news claim count > 0; otherwise news-negative counterfactual claim is disallowed.
- Saves failure samples.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "09_GROUNDING_NEWS_NEGATIVE_FIX",
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
