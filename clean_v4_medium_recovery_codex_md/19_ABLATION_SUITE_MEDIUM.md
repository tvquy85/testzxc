# 19 — Ablation Suite Medium

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Run minimum ablations to test contributions of evidence, technical signals, flow/proxy reward, and DPO.

## Why this is needed
Current V4 small ablations are not sufficient for method claims. Medium ablations must not use NOT_RUN rows as evidence.

## Files to create or modify
- Modify only the files explicitly named in the command section. If a required file is missing, create it under the stated path.

## Inputs
```text
outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet
data/processed/medium_clean_v4_contexts_gated.parquet
outputs/tables/medium_baseline_comparison.csv
outputs/metrics/16_backtest_track_breakdown_medium.json
outputs/metrics/17_counterfactual_evidence_medium.json
```

## Outputs
```text
outputs/tables/medium_ablation_results.csv
outputs/metrics/19_ablation_suite_medium.json
outputs/status/19_ABLATION_SUITE_MEDIUM.status.json
```

## Commands
```bash
python -m src.eval.run_clean_v4_ablation_suite \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --full-predictions outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet \
  --baselines outputs/tables/medium_baseline_comparison.csv \
  --backtest outputs/metrics/16_backtest_track_breakdown_medium.json \
  --counterfactual outputs/metrics/17_counterfactual_evidence_medium.json \
  --output outputs/tables/medium_ablation_results.csv \
  --metrics outputs/metrics/19_ablation_suite_medium.json \
  --status outputs/status/19_ABLATION_SUITE_MEDIUM.status.json
```

## Verification
```bash
python - <<'PY'
import pandas as pd, json
a=pd.read_csv('outputs/tables/medium_ablation_results.csv')
assert len(a) >= 5
assert 'NOT_RUN' not in set(a.astype(str).values.ravel())
m=json.load(open('outputs/metrics/19_ablation_suite_medium.json'))
assert m.get('all_required_ablations_present') is True
print('PASS ablations')
PY
```

## Acceptance criteria
- Must include Full, NoNews, NoTechnical, NoFlow/proxy, RWSFT-only, Technical-only.
- No `NOT_RUN` row can be used as evidence.
- Method claim requires Full to improve over relevant ablation on at least one prediction/explanation metric.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "19_ABLATION_SUITE_MEDIUM",
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
