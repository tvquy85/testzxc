# 10 — Flow Dataset V3

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Build Flow V3 dataset using independent targets and better embeddings.

## Inputs
- parsed clean rationales
- debiased independent inferability
- claim grounding
- clean contexts
- optional `sentence-transformers/all-MiniLM-L6-v2`

## Outputs
- `data/reward/flow_v3_dataset.pt`
- `outputs/metrics/flow_v3_dataset_metrics.json`
- `outputs/status/10_FLOW_DATASET_V3_TARGETS_AND_EMBEDDINGS.status.json`

## Target vector
7 dimensions:
1. independent true-label probability
2. 1 - normalized judge entropy
3. news grounding score
4. technical grounding score
5. supported claim rate
6. utility proxy = signed action × target_return - cost
7. calibration proxy

## Conditioning vector
Use sentence-transformer embedding of context+rationale if available; otherwise hash embedding. Record backend.

## Codex task
Create `src/reward/flow_dataset_v3.py`.

## Run
```bash
python -m src.reward.flow_dataset_v3 \
  --rationales data/rationales/parsed/current_clean_train_qwen3_4b_v3.parquet \
  --inferability data/judges/independent_inferability_v3_debiased.parquet \
  --grounding data/judges/claim_grounding_v3.parquet \
  --contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --output data/reward/flow_v3_dataset.pt \
  --metrics outputs/metrics/flow_v3_dataset_metrics.json \
  --status outputs/status/10_FLOW_DATASET_V3_TARGETS_AND_EMBEDDINGS.status.json
```

## Verify
```bash
python - <<'PY'
import torch, json
d=torch.load("data/reward/flow_v3_dataset.pt", map_location="cpu", weights_only=False)
assert d["target"].shape[0] > 1000
assert d["target"].shape[1] == 7
assert d["cond"].shape[0] == d["target"].shape[0]
assert d["cond"].shape[1] >= 64
print(json.load(open("outputs/metrics/flow_v3_dataset_metrics.json")))
PY
```

## Acceptance
- Target dim = 7.
- Uses independent judge, not generator forecast distribution.
- Rows > 1,000.
