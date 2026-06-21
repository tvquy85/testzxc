# 18 — Minimum Baselines PEN SEP Policy

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Create minimum competitive baseline table using current data and existing repo references.

## Why this is needed
AAAI reviewers will ask how FIRE-Fin compares to PEN, SEP, Policy-style no-flow proxy reward, and simple technical/text baselines.

## Files to create or modify
- Modify only the files explicitly named in the command section. If a required file is missing, create it under the stated path.

## Inputs
```text
data/processed/medium_clean_v4_contexts_gated.parquet
outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet
PEN/
sep/
policy/
```

## Outputs
```text
outputs/tables/medium_baseline_comparison.csv
outputs/metrics/18_baseline_comparison_medium.json
outputs/status/18_MINIMUM_BASELINES_PEN_SEP_POLICY.status.json
```

## Commands
```bash
python -m src.baselines.run_reference_baselines_medium \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --predictions outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet \
  --output outputs/tables/medium_baseline_comparison.csv \
  --metrics outputs/metrics/18_baseline_comparison_medium.json \
  --status outputs/status/18_MINIMUM_BASELINES_PEN_SEP_POLICY.status.json
```

## Verification
```bash
python - <<'PY'
import pandas as pd, json
t=pd.read_csv('outputs/tables/medium_baseline_comparison.csv')
assert len(t) >= 4
assert {'method','macro_f1'}.issubset(t.columns)
m=json.load(open('outputs/metrics/18_baseline_comparison_medium.json'))
assert 'best_method_by_macro_f1' in m
print('PASS baselines')
PY
```

## Acceptance criteria
- At least 4 baselines: Technical, FinBERT/Text, Qwen SFT/RWSFT, Qwen DPO.
- If PEN/SEP/Policy exact reproduction is not run, mark `reference_only=true` and explain.
- Do not claim outperforming PEN/SEP/Policy without comparable run.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "18_MINIMUM_BASELINES_PEN_SEP_POLICY",
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
