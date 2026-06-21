# 08 — Label Order Debias Multi Permutation

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Average judge probabilities over multiple label-order permutations.

## Why this is needed
LLM-as-judge can suffer position/order bias. Normal/reversed is not enough for medium evidence.

## Files to create or modify
Create `src/judges/judge_debias_multi_permutation_v5.py` if missing.

## Inputs
```text
data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet
data/processed/medium_clean_v4_contexts_gated.parquet
outputs/judges/medium_clean_v4_independent_inferability.parquet
```

## Outputs
```text
outputs/judges/medium_clean_v4_inferability_debiased.parquet
outputs/metrics/08_label_order_debias_multi_perm.json
outputs/status/08_LABEL_ORDER_DEBIAS_MULTI_PERMUTATION.status.json
```

## Commands
```bash
python -m src.judges.judge_debias_multi_permutation_v5 \
  --rationales data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --config configs/default_paths.yaml \
  --model qwen3_4b \
  --permutations normal,reversed,random1,random2,random3 \
  --output outputs/judges/medium_clean_v4_inferability_debiased.parquet \
  --metrics outputs/metrics/08_label_order_debias_multi_perm.json \
  --status outputs/status/08_LABEL_ORDER_DEBIAS_MULTI_PERMUTATION.status.json
```

## Verification
```bash
python - <<'PY'
import pandas as pd, json
x=pd.read_parquet('outputs/judges/medium_clean_v4_inferability_debiased.parquet')
assert 'true_label_probability_debiased' in x.columns
m=json.load(open('outputs/metrics/08_label_order_debias_multi_perm.json'))
assert m['valid_permutation_count_mean'] >= 2
print('PASS debias')
PY
```

## Acceptance criteria
- At least 2 valid permutations per candidate on average.
- Reports argmax consistency and KL dispersion.
- If consistency < 0.70, set `debias_claim_allowed=false`, but keep scores for diagnostics.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "08_LABEL_ORDER_DEBIAS_MULTI_PERMUTATION",
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
