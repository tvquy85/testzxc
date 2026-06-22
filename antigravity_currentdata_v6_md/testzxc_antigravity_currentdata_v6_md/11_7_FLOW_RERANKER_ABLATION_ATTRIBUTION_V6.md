# 11.7 - Flow Reranker Ablation Attribution V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Attribute the Step 11.6 utility-reranker diagnostic win. Step 11.6 showed a train-selected pairwise reranker can beat proxy on validation rank and pair accuracy, but that does not prove the improvement comes from Flow. This step compares train-selected rerankers across full, no-flow, only-flow, and related feature sets.

## Outputs
```text
outputs/tables/11_7_v6_flow_reranker_ablation_attribution.csv
outputs/tables/11_7_v6_flow_reranker_ablation_grid.csv
outputs/metrics/11_7_v6_flow_reranker_ablation_attribution.json
outputs/status/11_7_FLOW_RERANKER_ABLATION_ATTRIBUTION_V6.status.json
```

## Method
For each feature set, train the same candidate family as Step 11.6 using only the train split:

```text
models: ridge utility regression, pairwise logistic utility ranking
selection: train win count vs proxy, then train pair/rank/top-decile tie-breakers
evaluation: held-out Flow validation split
required attribution sets: full, no_flow, only_flow
```

Feature sets:
```text
full: flow + proxy + single judge + grounding/error features
no_flow: proxy + single judge + grounding/error features
only_flow: flow_reward_score only
no_proxy: flow + single judge + grounding/error features
no_single_judge: flow + proxy + grounding/error features
scores_only: flow + proxy + single judge
grounding_only: grounding/error features
```

Attribution is supported only if the full feature set has a clear validation edge over the no-flow ablation. If the no-flow ablation matches or exceeds full, the Step 11.6 win must not be claimed as a Flow-specific improvement.

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.reward.flow_reranker_ablation_attribution_v6 --predictions outputs/tables/11_v6_flow_predictions.csv --output outputs/tables/11_7_v6_flow_reranker_ablation_attribution.csv --grid-output outputs/tables/11_7_v6_flow_reranker_ablation_grid.csv --metrics outputs/metrics/11_7_v6_flow_reranker_ablation_attribution.json --status outputs/status/11_7_FLOW_RERANKER_ABLATION_ATTRIBUTION_V6.status.json --manifest outputs/manifests/11_7_FLOW_RERANKER_ABLATION_ATTRIBUTION_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_flow_reranker_ablation_attribution_v6.py
```

## Acceptance
```text
status JSON gate = PASS
full/no_flow/only_flow selected by train only
validation metrics reported against proxy
claim_allowed = false
flow_attribution_supported can only be true when full clearly beats no_flow
```

## Progress Update 2026-06-22
Status: `PASS`; diagnostic only.

Scale:
```text
train_rows: 2361
eval_rows: 639
feature_set_count: 7
```

Key validation comparison:
```text
proxy val rank: 0.1967
proxy val pair: 0.6164
proxy val top_decile: 0.002353

full selected: pairwise_logistic C=0.01
full val rank: 0.3890
full val pair: 0.6575
full val top_decile: 0.0000
full val wins vs proxy: 2/3

no_flow selected: ridge_utility alpha=100
no_flow val rank: 0.4267
no_flow val pair: 0.6849
no_flow val top_decile: 0.0000
no_flow val wins vs proxy: 2/3

only_flow selected: pairwise_logistic C=0.01
only_flow val rank: 0.3383
only_flow val pair: 0.5352
only_flow val top_decile: -0.000055
only_flow val wins vs proxy: 1/3
```

Attribution result:
```text
flow_attribution_supported: false
no_flow_matches_or_exceeds_full: true
only_flow_underperforms_full: true
flow_specific_edge_metric_count: 0

full_vs_no_flow_val_delta:
  rank: -0.0377
  pair: -0.0274
  top_decile: 0.0000
```

Interpretation:
```text
The Step 11.6 utility-reranker win is real as a diagnostic rank/pair signal, but it is not attributable to Flow. The no-flow ablation matches or exceeds the full feature set on held-out validation, while the only-flow model underperforms. Therefore Step 11.6 should not be used to claim Flow reward improvement. The next Flow fix must train an official checkpoint/objective whose gains survive no-flow and only-flow ablations.
```

