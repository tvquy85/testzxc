# Step 11.6 Story - Flow Utility Reranker Probe V6

Date: 2026-06-22

## What We Tested
Step 11.5 showed that the current Flow distribution-matching model has useful global rank signal but fails on pairwise and top-decile utility. We tested whether this is an objective issue by training a small utility-aware reranker on the existing Flow train split and evaluating on the held-out Flow validation split.

The probe follows paired/listwise ranking motivation:

- Bradley and Terry 1952: paired comparisons can estimate latent ranking strength.
- Burges 2010 RankNet/LambdaRank/LambdaMART: ranking objectives target ordering and top-of-list quality.
- Scikit-learn Ridge: regularized regression is a simple utility-score baseline.

## Result
The train-selected model was a pairwise logistic reranker:

```text
selected_method: pairwise_logistic
selected_regularization: 0.01
train_rows: 2361
eval_rows: 639
pairwise_train_examples: 492
```

It won all three core metrics against proxy on train:

```text
train_metric_wins_vs_proxy: rank=true, pair=true, top_decile=true
train_core_win_count_vs_proxy: 3
```

On held-out validation it won rank and pair accuracy but still failed top-decile utility:

```text
utility_reranker rank: 0.3890
proxy rank: 0.1967
utility_reranker pair accuracy: 0.6575
proxy pair accuracy: 0.6164
utility_reranker top-decile utility: 0.0000
proxy top-decile utility: 0.002353
eval_metric_wins_vs_proxy: rank=true, pair=true, top_decile=false
eval_core_win_count_vs_proxy: 2
```

## Paper Value
This gives a concrete next method direction: the Flow failure is not merely absence of reward signal. A utility-aware ranking objective can recover validation rank and pair wins. The remaining top-decile failure says the next official model must include a top-of-list/listwise utility term, not only pairwise ordering.

## Claim Boundary
This does not open the Flow claim. The reranker is a diagnostic model trained on existing scores, not the official Flow checkpoint. It also lacks the required ablation checkpoints and still misses top-decile utility. It should be used to justify the next Flow training objective, not as final evidence.
