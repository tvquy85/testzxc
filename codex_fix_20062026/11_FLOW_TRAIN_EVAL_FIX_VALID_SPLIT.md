# 11 — Train and Evaluate Flow V3

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Train Flow V3 and compare fairly against proxy reward on a proper holdout.

## Inputs
- `data/reward/flow_v3_dataset.pt`

## Outputs
- `checkpoints/flow_reward_v3/model.pt`
- `outputs/metrics/flow_reward_v3_train_metrics.json`
- `outputs/metrics/flow_vs_proxy_v3_eval.json`
- `outputs/status/11_FLOW_TRAIN_EVAL_FIX_VALID_SPLIT.status.json`

## Codex tasks
Create:
- `src/reward/train_flow_reward_v3.py`
- `src/reward/evaluate_flow_vs_proxy_v3.py`

## Training defaults
- epochs 20
- lr 2e-4
- batch size 128
- holdout fraction 0.2
- save best holdout model

## Claim rule
Pipeline PASS does not imply flow claim. `flow_claim_allowed=true` only if:
- rank correlation beats proxy by >= 0.03
- preference pair accuracy beats proxy by >= 0.02
- holdout loss decreases

## Run
```bash
python -m src.reward.train_flow_reward_v3 \
  --dataset data/reward/flow_v3_dataset.pt \
  --output-dir checkpoints/flow_reward_v3 \
  --metrics outputs/metrics/flow_reward_v3_train_metrics.json \
  --epochs 20 --batch-size 128

python -m src.reward.evaluate_flow_vs_proxy_v3 \
  --dataset data/reward/flow_v3_dataset.pt \
  --model checkpoints/flow_reward_v3/model.pt \
  --output outputs/metrics/flow_vs_proxy_v3_eval.json \
  --status outputs/status/11_FLOW_TRAIN_EVAL_FIX_VALID_SPLIT.status.json
```

## Verify
```bash
python - <<'PY'
import json, pathlib
m=json.load(open("outputs/metrics/flow_vs_proxy_v3_eval.json"))
assert "methods" in m
assert pathlib.Path("checkpoints/flow_reward_v3/model.pt").exists()
print(m)
PY
```

## Acceptance
- Model and eval metrics saved.
- Claim flag explicit.
