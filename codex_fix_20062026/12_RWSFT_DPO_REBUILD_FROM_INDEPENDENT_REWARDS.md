# 12 — Rebuild RWSFT/DPO from Independent Rewards

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Create better current-data RWSFT/DPO examples using independent rewards.

## Inputs
- clean parsed rationales
- debiased inferability
- claim grounding
- flow eval metrics

## Outputs
- `data/alignment/rwsft_current_v3.jsonl`
- `data/alignment/dpo_current_v3.jsonl`
- `outputs/metrics/alignment_dataset_current_v3.json`
- `outputs/status/12_RWSFT_DPO_REBUILD_FROM_INDEPENDENT_REWARDS.status.json`

## Default reward
```text
reward = 0.45 * independent_true_label_probability
       + 0.20 * technical_grounding_score
       + 0.20 * news_grounding_score
       + 0.10 * utility_proxy_current
       + 0.05 * schema_ok
```
If Flow V3 claim is allowed, use:
```text
reward = 0.5 * flow_v3_reward + 0.5 * composite_reward
```

## Pair rule
For sample_id with >=2 candidates:
- chosen = highest reward
- rejected = lowest reward
- margin >= 0.05
- chosen must not have contradicted technical claims

## Codex task
Create `src/alignment/build_alignment_current_v3.py`.

## Run
```bash
python -m src.alignment.build_alignment_current_v3 \
  --rationales data/rationales/parsed/current_clean_train_qwen3_4b_v3.parquet \
  --inferability data/judges/independent_inferability_v3_debiased.parquet \
  --grounding data/judges/claim_grounding_v3.parquet \
  --flow-eval outputs/metrics/flow_vs_proxy_v3_eval.json \
  --rwsft-output data/alignment/rwsft_current_v3.jsonl \
  --dpo-output data/alignment/dpo_current_v3.jsonl \
  --metrics outputs/metrics/alignment_dataset_current_v3.json \
  --status outputs/status/12_RWSFT_DPO_REBUILD_FROM_INDEPENDENT_REWARDS.status.json
```

## Verify
```bash
python - <<'PY'
import json, pathlib
m=json.load(open("outputs/metrics/alignment_dataset_current_v3.json"))
assert pathlib.Path("data/alignment/rwsft_current_v3.jsonl").exists()
assert pathlib.Path("data/alignment/dpo_current_v3.jsonl").exists()
assert m["rwsft_examples"] >= 1000
assert m["dpo_pairs"] >= 500
assert m["mean_reward_chosen"] > m["mean_reward_rejected"]
print(m)
PY
```

## Acceptance
- RWSFT >= 1,000.
- DPO pairs >= 500.
- Chosen reward mean > rejected reward mean.
