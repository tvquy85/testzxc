# 07 — Independent Judge Medium Full

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Run independent inferability judge on all medium candidates.

## Why this is needed
Judge must not copy the generator forecast distribution. It must read context+rationale and infer label probabilities independently.

## Files to create or modify
- Modify only the files explicitly named in the command section. If a required file is missing, create it under the stated path.

## Inputs
```text
data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet
data/processed/medium_clean_v4_contexts_gated.parquet
```

## Outputs
```text
outputs/judges/medium_clean_v4_independent_inferability.parquet
outputs/metrics/07_independent_judge_medium.json
outputs/status/07_INDEPENDENT_JUDGE_MEDIUM_FULL.status.json
```

## Commands
```bash
python -m src.judges.independent_inferability_judge_v4 \
  --rationales data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --config configs/default_paths.yaml \
  --model qwen3_4b \
  --orders normal,reversed \
  --max-new-tokens 160 \
  --temperature 0.0 \
  --batch-size 4 \
  --output outputs/judges/medium_clean_v4_independent_inferability.parquet \
  --metrics outputs/metrics/07_independent_judge_medium.json \
  --status outputs/status/07_INDEPENDENT_JUDGE_MEDIUM_FULL.status.json
```

## Verification
```bash
python - <<'PY'
import pandas as pd, json
x=pd.read_parquet('outputs/judges/medium_clean_v4_independent_inferability.parquet')
assert len(x) >= 1000
m=json.load(open('outputs/metrics/07_independent_judge_medium.json'))
assert m.get('schema_ok_rate',0) >= 0.95
assert m.get('mean_true_label_probability',0) > 0.20
print('PASS independent judge')
PY
```

## Acceptance criteria
- Schema ok >= 0.95.
- Mean true-label probability > random 0.20.
- Every candidate either has a judge row or explicit failure.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "07_INDEPENDENT_JUDGE_MEDIUM_FULL",
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
