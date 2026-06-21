# 15 — Predict With Adapter V4 Medium

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Generate test predictions using the new medium adapter.

## Why this is needed
Previous evaluation used older checkpoint. This step ensures evaluation matches the upgraded training.

## Files to create or modify
- Modify only the files explicitly named in the command section. If a required file is missing, create it under the stated path.

## Inputs
```text
data/processed/medium_clean_v4_contexts_gated.parquet
outputs/models/qwen3_medium_clean_v4_dpo_adapter/
prompts/rationale_generation_prompt_evidence_v4.txt
```

## Outputs
```text
outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet
outputs/metrics/15_predict_with_adapter_medium.json
outputs/status/15_PREDICT_WITH_ADAPTER_V4_MEDIUM.status.json
```

## Commands
```bash
python -m src.eval.generate_test_predictions_v2 \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --split test \
  --model-key qwen3_4b \
  --adapter outputs/models/qwen3_medium_clean_v4_dpo_adapter \
  --prompt prompts/rationale_generation_prompt_evidence_v4.txt \
  --schema-version v4 \
  --max-new-tokens 256 --temperature 0.0 --batch-size 4 \
  --output outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet \
  --metrics outputs/metrics/15_predict_with_adapter_medium.json \
  --status outputs/status/15_PREDICT_WITH_ADAPTER_V4_MEDIUM.status.json
```

## Verification
```bash
python - <<'PY'
import pandas as pd, json
df=pd.read_parquet('outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet')
assert len(df) >= 150
m=json.load(open('outputs/metrics/15_predict_with_adapter_medium.json'))
assert m.get('schema_ok_rate',0) >= 0.90
print('PASS predictions', len(df))
PY
```

## Acceptance criteria
- Uses new adapter path.
- Test rows >= 150.
- Schema ok >= 0.90.
- Actual number of trading days reported.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "15_PREDICT_WITH_ADAPTER_V4_MEDIUM",
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
