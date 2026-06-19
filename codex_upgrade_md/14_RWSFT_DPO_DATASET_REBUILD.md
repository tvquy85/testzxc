# 14 — Rebuild RWSFT and DPO datasets from flow v2

## Goal

Construct clean train-only alignment datasets with enough scale and no test contamination.

## Files to create

```text
src/alignment/build_rwsft_dataset_v2.py
src/alignment/build_dpo_pairs_v2.py
```

## RWSFT selection

From train candidates:
- score each candidate with flow reward v2;
- select top candidate per sample;
- optionally keep top 2 if score gap is small;
- weight each example by normalized flow reward.

## DPO pair construction

For each sample:
- chosen = high reward candidate;
- rejected = low reward candidate;
- require minimum reward gap:

```text
reward_gap >= 0.10
```

Add counterfactual negative pairs if available.

## Outputs

```text
data/alignment/rwsft_train_v2.jsonl
data/alignment/dpo_pairs_train_v2.jsonl
outputs/metrics/alignment_dataset_v2_summary.json
```

## Verification commands

```bash
python src/alignment/build_rwsft_dataset_v2.py \
  --rationales data/rationales/parsed/train_candidates_strict.parquet \
  --flow-checkpoint checkpoints/flow_reward_v2 \
  --output data/alignment/rwsft_train_v2.jsonl \
  --summary outputs/metrics/alignment_dataset_v2_summary.json

python src/alignment/build_dpo_pairs_v2.py \
  --rationales data/rationales/parsed/train_candidates_strict.parquet \
  --flow-checkpoint checkpoints/flow_reward_v2 \
  --output data/alignment/dpo_pairs_train_v2.jsonl
```

## Acceptance criteria

- All rows are `split=train`.
- RWSFT examples >= 5,000 for MVP; target >= 20,000.
- DPO pairs >= 2,000 for MVP; target >= 20,000.
- Pair reward gap distribution is reported.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "14_RWSFT_DPO_DATASET_REBUILD",
  "status": "PASS|FAIL",
  "inputs_checked": [],
  "outputs_created": [],
  "metrics": {},
  "failures": [],
  "next_step_allowed": true
}
```

Save it to:

```text
outputs/status/14_RWSFT_DPO_DATASET_REBUILD.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
