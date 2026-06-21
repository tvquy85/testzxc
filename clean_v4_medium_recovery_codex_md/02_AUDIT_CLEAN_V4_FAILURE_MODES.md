# 02 — Audit Clean V4 Failure Modes

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Generate a machine-readable failure-mode report from Clean V4 outputs.

## Why this is needed
Before medium upgrades, Codex must know what is currently weak: negative Sharpe, small Flow evidence, weak news counterfactuals, template-heavy rationales, and no V4 adapter evaluation.

## Files to create or modify
Create `src/repro/audit_clean_v4_failure_modes.py`. Use regex extraction from the Markdown report plus JSON loading from metrics where available.

## Inputs
```text
BaoCaoCodexFixCleanData_20062026.md
outputs/repro/currentdata_clean_v4_science_gate_report.json
outputs/metrics/*clean*v4*.json
outputs/tables/*clean*v4*.csv
review_samples/dataclean_v4_20062026/*.jsonl
```

## Outputs
```text
outputs/metrics/02_clean_v4_failure_modes.json
outputs/status/02_AUDIT_CLEAN_V4_FAILURE_MODES.status.json
```

## Commands
```bash
python -m src.repro.audit_clean_v4_failure_modes \
  --report BaoCaoCodexFixCleanData_20062026.md \
  --metrics-dir outputs/metrics \
  --tables-dir outputs/tables \
  --samples-dir review_samples/dataclean_v4_20062026 \
  --output outputs/metrics/02_clean_v4_failure_modes.json \
  --status outputs/status/02_AUDIT_CLEAN_V4_FAILURE_MODES.status.json
```

## Verification
```bash
python - <<'PY'
import json
m=json.load(open('outputs/metrics/02_clean_v4_failure_modes.json'))
for k in ['full_clean_v4_sharpe','counterfactual_pass_rate','flow_rows','adapter_v4_trained']:
    assert k in m, k
print('PASS audit', m)
PY
```

## Acceptance criteria
- Records whether flow, alpha, counterfactual, and AAAI-ready claims are currently allowed.
- Records if V4 adapter has actually been trained and evaluated.
- Does not mark scientific success from pipeline pass alone.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "02_AUDIT_CLEAN_V4_FAILURE_MODES",
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
