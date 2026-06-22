# 11.6 - Flow Utility Reranker Probe V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Test whether the Flow blocker is mainly an objective/reranking issue. Step 11.5 showed the MSE distribution-matching Flow model ranks global utility better than proxy, but misses pairwise and top-decile utility. This probe trains a small utility-aware reranker on the existing train split and evaluates it on the held-out Flow validation split.

Method basis:
```text
Bradley-Terry paired-comparison ranking: pairwise preferences can induce a latent ranking score.
Burges RankNet/LambdaRank/LambdaMART: ranking losses target ordering and top-of-list quality rather than pointwise distribution fit.
Scikit-learn Ridge: regularized regression provides a simple utility-score baseline.
```

## Outputs
```text
outputs/tables/11_6_v6_flow_utility_reranker_summary.csv
outputs/tables/11_6_v6_flow_utility_reranker_predictions.csv
outputs/metrics/11_6_v6_flow_utility_reranker_probe.json
outputs/status/11_6_FLOW_UTILITY_RERANKER_PROBE_V6.status.json
```

## Method
Use only train split labels to train candidate rerankers:

```text
features: flow_reward_score, proxy_average_reward, single_best_judge_reward,
          news_grounding_score, technical_grounding_score,
          unsupported_news_claim_rate
targets: raw_realized_utility for ridge utility regression
pairs: non-tied within-sample raw_realized_utility comparisons for pairwise logistic
```

`technical_rule_delta` and `raw_realized_utility` are never used as score features. They are only training/evaluation targets.

Select the reranker by train win count versus proxy, then train pair/rank/top-decile tie-breakers. Evaluate the selected score on Flow validation rows against the same core metrics as Step 11.

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.reward.flow_utility_reranker_probe_v6 --predictions outputs/tables/11_v6_flow_predictions.csv --summary-output outputs/tables/11_6_v6_flow_utility_reranker_summary.csv --predictions-output outputs/tables/11_6_v6_flow_utility_reranker_predictions.csv --metrics outputs/metrics/11_6_v6_flow_utility_reranker_probe.json --status outputs/status/11_6_FLOW_UTILITY_RERANKER_PROBE_V6.status.json --manifest outputs/manifests/11_6_FLOW_UTILITY_RERANKER_PROBE_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_flow_utility_reranker_probe_v6.py
```

## Acceptance
```text
status JSON gate = PASS
reranker selected on train only
validation metrics reported against proxy
claim_allowed = false unless reranker is promoted to an official Flow checkpoint with required ablations
```

## Progress Update 2026-06-22
Status: `PASS`; diagnostic only.

Selected reranker:
```text
selected_method: pairwise_logistic
selected_regularization: 0.01
train_rows: 2361
eval_rows: 639
pairwise_train_examples: 492
```

Train selection result:
```text
train_metric_wins_vs_proxy: rank=true, pair=true, top_decile=true
train_core_win_count_vs_proxy: 3
```

Held-out Flow validation result:
```text
eval_metric_wins_vs_proxy: rank=true, pair=true, top_decile=false
eval_core_win_count_vs_proxy: 2
eval_core_utility_win_vs_proxy: true

utility_reranker rank: 0.3890
proxy rank: 0.1967
utility_reranker pair accuracy: 0.6575
proxy pair accuracy: 0.6164
utility_reranker top-decile utility: 0.0000
proxy top-decile utility: 0.002353
```

Interpretation:
```text
The Flow blocker is likely objective-level, not impossible signal absence. A train-selected pairwise utility reranker beats proxy on rank and pair accuracy on validation, but still fails top-decile utility. The next real fix should train an official Flow/listwise reward checkpoint with top-decile/listwise utility terms and ablations, then rerun Step 11. This diagnostic alone cannot open the Flow claim.
```

Post-Step-11.7 attribution note: the reranker win should not be described as Flow-specific. Step 11.7 shows a no-flow ablation matches or exceeds the full feature set on validation, while only-flow underperforms. This keeps the Flow claim blocked and narrows the next fix to an official Flow objective whose gains survive attribution ablations.
