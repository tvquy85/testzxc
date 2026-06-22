# 11 — Flow Evaluation against Proxy and Raw Realized Utility

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Evaluate Flow V6 fairly against proxy-average on held-out validation using both decision-distribution metrics and raw realized utility.

## Research-backed novelty
Step 11 is the scientific proof point for `DFD-FlowReward-V6`. It must show that Flow is not just a smoother proxy score. Flow claim is allowed only if the learned distribution improves decision utility while preserving distributional fidelity and evidence faithfulness.

## Outputs
```text
outputs/models/flow_reward_v6_decision.pt
outputs/metrics/11_v6_flow_vs_proxy_raw_utility.json
outputs/status/11_FLOW_EVAL_RAW_UTILITY_AND_PROXY.status.json
```

## Metrics
`rank_correlation_with_true_label_probability`, `rank_correlation_with_raw_realized_utility`, `preference_pair_accuracy`, `kl_to_judge_distribution`, `top_decile_raw_utility`, `top_decile_target_probability`.

Flow improvement requires beating proxy on at least 2/3 core metrics: raw utility rank correlation, pair accuracy, top-decile raw utility.

## V6 evaluation families

### Distributional fidelity

```text
kl_to_calibrated_judge_distribution
js_to_calibrated_judge_distribution
brier_true_label_probability
ece_true_label_probability
top_decile_target_probability
```

### Decision utility

```text
rank_correlation_with_raw_realized_utility
preference_pair_accuracy_by_raw_utility
top_decile_raw_realized_utility
technical_rule_delta_top_decile
```

### Evidence faithfulness

```text
top_decile_news_grounding_score
top_decile_technical_grounding_score
unsupported_claim_rate_top_decile
faithfulness_lift_vs_proxy
```

### Robustness by track/regime

```text
metric_wins_by_track
metric_wins_by_volatility_regime
hard_event_news_win_rate
weak_context_failure_rate
```

Core utility wins remain:

```text
1. rank_correlation_with_raw_realized_utility
2. preference_pair_accuracy_by_raw_utility
3. top_decile_raw_realized_utility
```

Claim rule:

```text
flow_claim_allowed =
  wins_at_least_2_of_3_core_utility_metrics
  and distributional_fidelity_not_worse_than_proxy
  and top_decile_unsupported_claim_rate_not_worse_than_proxy
  and no_critical_track_regression
```

`pipeline_pass=true` means the evaluation ran and artifacts are valid. It does not imply `flow_claim_allowed=true`.

## Test case
```python
from src.reward.evaluate_flow_v6 import metric_wins

def test_metric_wins_requires_two_of_three():
    assert metric_wins({'rank':True,'pair':False,'top_decile':True})
    assert not metric_wins({'rank':True,'pair':False,'top_decile':False})
```

## Commands
```bash
python -m src.reward.train_flow_reward_v6 --dataset data/reward/current_v6_flow_decision_dataset.pt --output outputs/models/flow_reward_v6_decision.pt --metrics outputs/metrics/11_v6_flow_train.json --status outputs/status/11_FLOW_TRAIN_V6.status.json
python -m src.reward.evaluate_flow_v6 --dataset data/reward/current_v6_flow_decision_dataset.pt --model outputs/models/flow_reward_v6_decision.pt --metrics outputs/metrics/11_v6_flow_vs_proxy_raw_utility.json --status outputs/status/11_FLOW_EVAL_RAW_UTILITY_AND_PROXY.status.json --predictions outputs/tables/11_v6_flow_predictions.csv --split val --integration-steps 16
python -m pytest -q tests/test_flow_eval_v6.py tests
```

## Required comparison methods

The evaluation table must include at least:

```text
flow_reward_v6
proxy_average_reward
single_best_judge_reward
technical_rule_reward
flow_reward_without_reliability_weight
flow_reward_without_grounding_auxiliary
```

If a method cannot be run, record it as `NOT_RUN` and keep `flow_claim_allowed=false`; do not silently drop the method from the table.

## Acceptance

- Validation rows >=20% of reward dataset or >=300 rows, whichever is smaller for current-data V6.
- Flow beats proxy on at least 2/3 core utility metrics for `flow_claim_allowed=true`.
- Distributional fidelity cannot be materially worse than proxy average.
- Top-decile unsupported claim rate cannot be worse than proxy average.
- Track/regime table must exist and must not hide `weak_or_context_only` failures inside aggregate averages.

## Progress Update 2026-06-22
Status: `stage_3_full_scale PIPELINE_PASS`; Flow train/eval ran successfully on the full Step 10 V6 reward dataset, but `flow_claim_allowed=false`.

Artifacts verified:
```text
outputs/models/flow_reward_v6_decision.pt
outputs/metrics/11_v6_flow_train.json
outputs/metrics/11_v6_flow_vs_proxy_raw_utility.json
outputs/tables/11_v6_flow_predictions.csv
outputs/manifests/11_FLOW_TRAIN_V6.manifest.json
outputs/manifests/11_FLOW_EVAL_RAW_UTILITY_AND_PROXY.manifest.json
outputs/status/11_FLOW_TRAIN_V6.status.json
outputs/status/11_FLOW_EVAL_RAW_UTILITY_AND_PROXY.status.json
```

Training metrics from `outputs/metrics/11_v6_flow_train.json`:
```text
rows: 3000
train_rows: 2361
val_rows: 639
epochs_completed: 80
best_val_loss: 0.2227
loss_decreased: true
device: cuda
weighted_by: judge_reliability_weight
```

Evaluation metrics from `outputs/metrics/11_v6_flow_vs_proxy_raw_utility.json`:
```text
rows: 3000
eval_rows: 639
eval_split: val
flow_claim_allowed: false
flow_reward_improvement: false
core_utility_win: false
metric_wins: rank=true, pair=false, top_decile=false
distributional_fidelity_not_worse_than_proxy: false
top_decile_unsupported_claim_rate_not_worse_than_proxy: true
missing_required_methods: flow_reward_without_reliability_weight, flow_reward_without_grounding_auxiliary
```

Core utility comparison:
```text
rank_correlation_with_raw_realized_utility:
  flow_reward_v6 = 0.3383
  proxy_average_reward = 0.1967

preference_pair_accuracy_by_raw_utility:
  flow_reward_v6 = 0.5352
  proxy_average_reward = 0.6164

top_decile_raw_realized_utility:
  flow_reward_v6 = -0.000055
  proxy_average_reward = 0.002353
```

Other comparison anchors:
```text
single_best_judge_reward rank correlation = 0.4382
technical_rule_reward pair accuracy = 1.0000
flow_reward_without_reliability_weight = NOT_RUN
flow_reward_without_grounding_auxiliary = NOT_RUN
```

Interpretation: Step 11 is a valid negative result. Flow training/evaluation artifacts are reproducible and status JSON passes, but Flow does not beat proxy on the required 2/3 utility metrics and does not pass the distributional-fidelity claim gate. Downstream steps may continue only with `flow_claim_allowed=false`; no paper claim should state that Flow V6 beats proxy on current data.

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.reward.train_flow_reward_v6 --dataset data/reward/current_v6_flow_decision_dataset.pt --output outputs/models/flow_reward_v6_decision.pt --metrics outputs/metrics/11_v6_flow_train.json --status outputs/status/11_FLOW_TRAIN_V6.status.json --epochs 80 --batch-size 128 --lr 0.0002 --seed 42
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.reward.evaluate_flow_v6 --dataset data/reward/current_v6_flow_decision_dataset.pt --model outputs/models/flow_reward_v6_decision.pt --metrics outputs/metrics/11_v6_flow_vs_proxy_raw_utility.json --status outputs/status/11_FLOW_EVAL_RAW_UTILITY_AND_PROXY.status.json --predictions outputs/tables/11_v6_flow_predictions.csv --split val --integration-steps 16
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/11_FLOW_TRAIN_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/11_FLOW_EVAL_RAW_UTILITY_AND_PROXY.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests
```
