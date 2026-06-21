# 17 — Counterfactual Evidence Medium

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Run evidence-level counterfactual evaluation on medium predictions.

## Why this is needed
Clean V4 improved counterfactuals, but news perturbations are still weaker than technical. Medium evaluation must report breakdown.

## Files to create or modify
- Modify only the files explicitly named in the command section. If a required file is missing, create it under the stated path.

## Inputs
```text
outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet
data/processed/medium_clean_v4_contexts_gated.parquet
outputs/models/qwen3_medium_clean_v4_dpo_adapter/
```

## Outputs
```text
outputs/metrics/17_counterfactual_evidence_medium.json
outputs/tables/medium_counterfactual_breakdown.csv
review_samples/medium_clean_v4/counterfactual_failures.jsonl
outputs/status/17_COUNTERFACTUAL_EVIDENCE_MEDIUM.status.json
```

## Commands
```bash
python -m src.eval.build_counterfactual_clean_v4 \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --predictions outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet \
  --output-tasks data/eval/medium_clean_v4_counterfactual_tasks.jsonl \
  --metrics outputs/metrics/17_counterfactual_task_build_medium.json

python -m src.eval.evaluate_counterfactual_directional_v4 \
  --tasks data/eval/medium_clean_v4_counterfactual_tasks.jsonl \
  --model-key qwen3_4b \
  --adapter outputs/models/qwen3_medium_clean_v4_dpo_adapter \
  --output outputs/metrics/17_counterfactual_evidence_medium.json \
  --breakdown-output outputs/tables/medium_counterfactual_breakdown.csv \
  --failures-output review_samples/medium_clean_v4/counterfactual_failures.jsonl \
  --status outputs/status/17_COUNTERFACTUAL_EVIDENCE_MEDIUM.status.json
```

## Verification
```bash
python - <<'PY'
import json, pandas as pd
m=json.load(open('outputs/metrics/17_counterfactual_evidence_medium.json'))
assert 'pass_rate' in m and 'no_change_rate' in m
b=pd.read_csv('outputs/tables/medium_counterfactual_breakdown.csv')
assert len(b) >= 3
print('PASS counterfactual', m['pass_rate'])
PY
```

## Acceptance criteria
- General counterfactual claim only if pass_rate >= 0.50 and no_change_rate <= 0.35.
- News faithfulness claim only if remove_positive_evidence and remove_negative_evidence pass >= 0.35.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "17_COUNTERFACTUAL_EVIDENCE_MEDIUM",
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
