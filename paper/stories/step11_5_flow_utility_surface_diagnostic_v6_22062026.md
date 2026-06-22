# Step 11.5 Story - Flow Rank Signal Is Real But Top-Decile Utility Is Misaligned

Step 11 showed that Flow beats the proxy on global rank correlation but fails pairwise preference and top-decile raw utility. Step 11.5 explains why this is not a simple threshold issue.

Key evidence:

```text
candidate_pair_count: 639
utility_varying_pair_count: 73
utility_varying_pair_rate: 0.1142
samples_with_utility_variation_rate: 0.1690
flow_rank_win_vs_proxy: true
flow_pair_accuracy_gap_vs_proxy: -0.0812
flow_top_decile_gap_vs_proxy: -0.00241
flow_top_decile_gap_vs_technical: -0.00305
flow_top_decile_overlap_with_technical: 0.0000
```

Interpretation:

The current Flow model learns a useful global ranking signal, but the data has sparse within-sample utility variation for pairwise preference learning. More importantly, Flow's top-decile region is almost disjoint from Technical_Rule's high-utility region, so it is not selecting the candidates that matter for decision utility.

Paper value:

- This gives a precise negative result instead of simply saying "Flow failed".
- It connects the failure to known learning-to-rank distinctions: global rank correlation, pairwise preference accuracy, and top-of-list utility are different objectives.
- The next model should use a pairwise/listwise utility-aware objective with explicit top-decile pressure, not another MSE distribution-matching run.

Potential paper wording:

```text
Flow improved global rank correlation with raw utility but failed top-decile utility. A utility-surface diagnostic showed that within-sample utility-varying pairs were sparse and Flow's top-decile selections had near-zero overlap with the technical-rule high-utility region. We therefore treat the current Flow result as a ranking-signal diagnostic and leave Flow-improvement claims blocked pending a listwise or pairwise utility-aware objective.
```
