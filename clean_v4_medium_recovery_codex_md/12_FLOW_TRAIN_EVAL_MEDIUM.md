# 12 — Flow Train Eval Medium

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Train Flow V5 and compare to proxy on true validation split.

## Why this is needed
Flow can only be claimed if it beats proxy under validation; small-slice positivity is not enough.

## Files to create or modify
- Modify only the files explicitly named in the command section. If a required file is missing, create it under the stated path.

## Inputs
```text
data/reward/medium_clean_v4_flow_dataset_v5.pt
```

## Outputs
```text
outputs/models/flow_reward_v5_medium/model.pt
outputs/metrics/12_flow_train_medium.json
outputs/metrics/12_flow_vs_proxy_medium.json
outputs/tables/medium_flow_vs_proxy.csv
outputs/status/12_FLOW_TRAIN_EVAL_MEDIUM.status.json
```

## Commands
```bash
python -m src.reward.train_flow_reward_v5 \
  --dataset data/reward/medium_clean_v4_flow_dataset_v5.pt \
  --output-dir outputs/models/flow_reward_v5_medium \
  --epochs 150 --batch-size 128 --lr 2e-4 --seed 42 \
  --metrics outputs/metrics/12_flow_train_medium.json \
  --status outputs/status/12_FLOW_TRAIN_MEDIUM.status.json

python -m src.reward.evaluate_flow_vs_proxy_v5 \
  --dataset data/reward/medium_clean_v4_flow_dataset_v5.pt \
  --checkpoint outputs/models/flow_reward_v5_medium \
  --split val \
  --output outputs/metrics/12_flow_vs_proxy_medium.json \
  --output-csv outputs/tables/medium_flow_vs_proxy.csv \
  --status outputs/status/12_FLOW_TRAIN_EVAL_MEDIUM.status.json
```

## Verification
```bash
python - <<'PY'
import json
m=json.load(open('outputs/metrics/12_flow_vs_proxy_medium.json'))
assert 'flow_claim_allowed' in m
assert m['eval_rows'] >= 100
print('PASS flow eval', m['flow_claim_allowed'])
PY
```

## Acceptance criteria
- Flow claim allowed only if it wins >= 2/3 metrics: rank corr, pair accuracy, top-decile utility.
- If Flow fails, downstream must fall back to proxy reward and record this.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "12_FLOW_TRAIN_EVAL_MEDIUM",
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
