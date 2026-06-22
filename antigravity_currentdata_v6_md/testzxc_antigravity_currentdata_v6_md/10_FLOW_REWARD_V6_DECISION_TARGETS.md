# 10 — Flow Reward V6 with Decision-Distribution Targets

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Build Flow primary target from independent judge decision distributions, closer to Policy-style proxy logits/distributions. Keep utility/grounding as auxiliary columns.

## Research-backed novelty
Use `DFD-FlowReward-V6`: **Decision-Faithful Distributional Flow Reward**.

The novelty is not merely replacing a scalar proxy by Flow. The V6 reward dataset must expose noisy/debiased judge distributions, grounding, utility, and track/regime metadata so the Flow model can be evaluated as a denoising distributional reward, not as a black-box score. See `FLOW_REWARD_V6_NOVELTY_PROPOSAL.md`.

Core selling point:

```text
Debiased LLM judge distribution -> reliability-weighted rectified-flow reward -> validation by after-cost decision utility and evidence faithfulness.
```

## Outputs
```text
data/reward/current_v6_flow_decision_dataset.pt
outputs/metrics/10_v6_flow_dataset_decision_targets.json
outputs/status/10_FLOW_REWARD_V6_DECISION_TARGETS.status.json
```

## Target design
Primary target dimension = 5: `[p_strong_down, p_mild_down, p_neutral, p_mild_up, p_strong_up]`. Save auxiliary `news_grounding_score`, `technical_grounding_score`, `raw_realized_utility`, `abnormal_return_h1`, `evidence_quality_weight`.

## V6 decision-faithful target contract

Primary target remains exactly five dimensions:

```text
target_distribution = [
  p_strong_down,
  p_mild_down,
  p_neutral,
  p_mild_up,
  p_strong_up
]
```

The target distribution must come from the calibrated/debiased independent judge ensemble from Steps 08-09. It must not be overwritten by realized return, action, or utility labels.

Required auxiliary fields:

```text
true_label_probability_ensemble
judge_reliability_weight
label_order_kl_mean
judge_disagreement_entropy
news_grounding_score
technical_grounding_score
unsupported_news_claim_rate
evidence_quality_weight
abnormal_return_h1
raw_realized_utility
technical_rule_delta
hard_event_track
volatility_regime
source_split
```

Recommended reliability weight:

```text
judge_reliability_weight =
  judge_schema_ok
  * argmax_consistency_ensemble
  * (1 - normalized_label_order_kl)
  * (1 - normalized_judge_disagreement_entropy)
  * evidence_quality_weight
```

`raw_realized_utility` is used as auxiliary/evaluation evidence, not as the primary Flow target:

```text
position(short,hold,long) = [-1, 0, 1]
raw_realized_utility = position(action_from_distribution) * abnormal_return_h1 - transaction_cost_proxy
technical_rule_delta = raw_realized_utility(candidate) - raw_realized_utility(technical_rule_baseline)
```

All target and utility fields must be computed only for train/val rows under the locked split. Test rows must not appear in reward/alignment data.

## Test case
```python
import torch
from pathlib import Path

def test_flow_decision_dataset_shape():
    p=Path('data/reward/current_v6_flow_decision_dataset.pt')
    assert p.exists()
    data=torch.load(p,map_location='cpu')
    assert data['target'].shape[1]==5
    assert data['cond'].shape[0]==data['target'].shape[0]
    assert set(data['split']).issuperset({'train','val'})
```

## Commands
```bash
python -m src.reward.build_flow_decision_dataset_v6 --rationales data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet --judge data/judges/current_v6_independent_judge_ensemble.parquet --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --output data/reward/current_v6_flow_decision_dataset.pt --metrics outputs/metrics/10_v6_flow_dataset_decision_targets.json --status outputs/status/10_FLOW_REWARD_V6_DECISION_TARGETS.status.json
python -m pytest -q tests/test_flow_decision_dataset_v6.py tests
```

## Acceptance
Rows >=2700, target dim=5, train/val split exists, target rows sum to one, condition dim >=128.

Additional V6 acceptance:

- `judge_reliability_weight` finite and in `[0,1]`.
- `target_distribution` rows sum to one within tolerance `1e-4`.
- `raw_realized_utility` and `technical_rule_delta` are present but masked if unavailable.
- No `source_split == "test"` rows.
- Track/regime columns are present, even if value is `unknown`.
- Metrics must report `target_source="calibrated_debiased_judge_distribution"` and `utility_is_auxiliary=true`.

## Progress Update 2026-06-22
Status: `stage_3_full_scale PASS`; Step 10 was rerun after Step 08/09 full-scale judge coverage reached 3000 candidate rows. Status file reports `PASS` and `next_step_allowed=true`; `claim_allowed=false` remains unchanged because Flow has not yet been trained/evaluated.

Artifacts verified:
```text
data/reward/current_v6_flow_decision_dataset.pt
outputs/metrics/10_v6_flow_dataset_decision_targets.json
outputs/manifests/10_FLOW_REWARD_V6_DECISION_TARGETS.manifest.json
outputs/status/10_FLOW_REWARD_V6_DECISION_TARGETS.status.json
```

Metrics from `outputs/metrics/10_v6_flow_dataset_decision_targets.json`:
```text
rows: 3000
target_dim: 5
cond_dim: 128
split_distribution: train=2361, val=639
source_split_distribution: train=3000
target_source: calibrated_debiased_judge_distribution
utility_is_auxiliary: true
target_sum_ok_rate: 1.000
judge_reliability_weight_min: 0.000
judge_reliability_weight_max: 0.900
raw_realized_utility_finite_rate: 1.000
technical_rule_delta_finite_rate: 1.000
hard_event_track_present: true
volatility_regime_present: true
```

Gate result:
```text
rows gate: PASS (3000 >= 2700)
target dim gate: PASS (5 == 5)
train/val split gate: PASS
target sum gate: PASS
condition dim gate: PASS (128 >= 128)
reliability weight gate: PASS
no test source rows gate: PASS
utility auxiliary fields gate: PASS
track/regime fields gate: PASS
status JSON contract: PASS
```

Interpretation: Step 10 now permits Step 11 Flow training/evaluation. This is a dataset-contract pass only; Flow claim, alignment claim, and forecast claim remain blocked until downstream utility/fidelity gates pass.

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.reward.build_flow_decision_dataset_v6 --rationales data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet --judge data/judges/current_v6_independent_judge_ensemble.parquet --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --output data/reward/current_v6_flow_decision_dataset.pt --metrics outputs/metrics/10_v6_flow_dataset_decision_targets.json --status outputs/status/10_FLOW_REWARD_V6_DECISION_TARGETS.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_flow_decision_dataset_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/10_FLOW_REWARD_V6_DECISION_TARGETS.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests
```
