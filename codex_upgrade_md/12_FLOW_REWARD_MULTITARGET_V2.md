# 12 — Multi-target regime-conditioned flow reward v2

## Goal

Upgrade `FlowRewardLite` from class-probability-only target to a true multi-signal distributional reward model.

## Existing files to inspect

```text
src/reward/flow_model_lite.py
src/reward/flow_dataset.py
src/reward/train_flow_reward_lite.py
```

## Codex task

Create v2 files:

```text
src/reward/flow_dataset_v2.py
src/reward/flow_model_v2.py
src/reward/train_flow_reward_v2.py
```

## Target vector

Each candidate rationale must have target vector:

```text
z1 = [
  inferability_true_label_prob,
  multi_judge_agreement,
  news_grounding_score,
  technical_grounding_score,
  counterfactual_directional_score_if_available,
  utility_score,
  calibration_proxy
]
```

If one component is missing, store a mask. The loss must support masked targets.

## Conditioning vector

Use:

```text
text_embedding
technical_feature_vector
regime_one_hot
volatility_features
model_id_embedding
```

## Regime-conditioned noise

Support:

```text
sigma = 0.5 for low_vol
sigma = 1.0 for normal_vol
sigma = 1.5 for high_vol
```

Make these configurable.

## Outputs

```text
data/reward/flow_v2_train_dataset.pt
checkpoints/flow_reward_v2/
outputs/metrics/flow_reward_v2_train_metrics.json
```

## Verification commands

```bash
python src/reward/flow_dataset_v2.py \
  --rationales data/rationales/parsed/train_candidates_strict.parquet \
  --inferability data/judges/inferability_multi_judge.parquet \
  --grounding data/judges/claim_grounding_scores.parquet \
  --output data/reward/flow_v2_train_dataset.pt

python src/reward/train_flow_reward_v2.py \
  --dataset data/reward/flow_v2_train_dataset.pt \
  --output-dir checkpoints/flow_reward_v2 \
  --metrics outputs/metrics/flow_reward_v2_train_metrics.json
```

## Acceptance criteria

- Target dimension >= 5.
- Missing reward components are masked, not filled with zero silently.
- Training loss decreases versus first epoch.
- Saved checkpoint includes config and feature schema.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "12_FLOW_REWARD_MULTITARGET_V2",
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
outputs/status/12_FLOW_REWARD_MULTITARGET_V2.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
