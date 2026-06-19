# 13 — Evaluate flow reward versus proxy average

## Goal

Prove that flow reward adds measurable value beyond simple proxy-score averaging.

## Files to create

```text
src/reward/evaluate_flow_vs_proxy.py
```

## Evaluation tasks

Compare:

```text
A. proxy_average_reward
B. single_best_judge_reward
C. flow_reward_v1 if available
D. flow_reward_v2
```

Metrics:

```text
rank_correlation_with_realized_utility
calibration_error
top_k_rationale_quality
preference_pair_accuracy
variance_by_regime
```

Use validation split only.

## Required outputs

```text
outputs/metrics/flow_vs_proxy_eval.json
outputs/tables/flow_vs_proxy_eval.csv
outputs/figures/flow_vs_proxy_calibration.png
```

## Verification commands

```bash
python src/reward/evaluate_flow_vs_proxy.py \
  --dataset data/reward/flow_v2_train_dataset.pt \
  --checkpoint checkpoints/flow_reward_v2 \
  --split val \
  --output-json outputs/metrics/flow_vs_proxy_eval.json \
  --output-csv outputs/tables/flow_vs_proxy_eval.csv
```

## Acceptance criteria

- Evaluation is on validation split only.
- Table includes all four methods or marks missing methods explicitly.
- Do not claim improvement unless flow beats proxy average on at least two pre-specified metrics.
- Save examples where proxy and flow disagree.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "13_FLOW_REWARD_EVAL_VS_PROXY",
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
outputs/status/13_FLOW_REWARD_EVAL_VS_PROXY.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
