# 20 — Strict Science Gate Medium

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Create the final medium-scale science gate and claim matrix.

## Why this is needed
The final output must separate reproducible execution from scientific claims. Negative results should not be hidden.

## Files to create or modify
Create `src/repro/currentdata_clean_v4_medium_science_gate.py`.

## Inputs
```text
outputs/metrics/02_clean_v4_failure_modes.json
outputs/metrics/05_generate_rationales_medium.json
outputs/metrics/07_independent_judge_medium.json
outputs/metrics/12_flow_vs_proxy_medium.json
outputs/metrics/13_alignment_dataset_medium.json
outputs/metrics/14_alignment_train_medium.json
outputs/metrics/16_backtest_track_breakdown_medium.json
outputs/metrics/17_counterfactual_evidence_medium.json
outputs/tables/medium_baseline_comparison.csv
outputs/tables/medium_ablation_results.csv
```

## Outputs
```text
outputs/repro/currentdata_clean_v4_medium_science_gate_report.json
outputs/tables/medium_claim_matrix.csv
outputs/status/20_STRICT_SCIENCE_GATE_MEDIUM.status.json
```

## Commands
```bash
python -m src.repro.currentdata_clean_v4_medium_science_gate \
  --metrics-dir outputs/metrics \
  --tables-dir outputs/tables \
  --output outputs/repro/currentdata_clean_v4_medium_science_gate_report.json \
  --claim-table outputs/tables/medium_claim_matrix.csv \
  --status outputs/status/20_STRICT_SCIENCE_GATE_MEDIUM.status.json
```

## Verification
```bash
python - <<'PY'
import json, pandas as pd
r=json.load(open('outputs/repro/currentdata_clean_v4_medium_science_gate_report.json'))
assert 'claim_decision' in r
assert 'claims' in r
c=pd.read_csv('outputs/tables/medium_claim_matrix.csv')
assert len(c) >= 5
print('PASS medium science gate', r['claim_decision'])
PY
```

## Acceptance criteria
- `aaaai_main_ready_allowed` should remain false unless full-scale statistical validation exists.
- `trading_alpha_allowed` follows strict alpha gate.
- `flow_reward_improvement_allowed` follows Flow-vs-proxy validation.
- Missing files/status fail the pipeline.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "20_STRICT_SCIENCE_GATE_MEDIUM",
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
