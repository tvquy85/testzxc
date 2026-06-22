# Step 11.7 Story - Flow Reranker Attribution Boundary

Date: 2026-06-22

Step 11.6 found a useful diagnostic signal: a small train-selected utility reranker beat the proxy on validation rank and pair accuracy. Step 11.7 tested whether that gain was actually caused by Flow-specific signal.

The ablation result is negative but important. The no-flow feature set selected `ridge_utility alpha=100` and matched or exceeded the full feature set on held-out validation:

```text
full rank/pair/top: 0.3890 / 0.6575 / 0.0000
no_flow rank/pair/top: 0.4267 / 0.6849 / 0.0000
only_flow rank/pair/top: 0.3383 / 0.5352 / -0.000055
```

Therefore the Step 11.6 reranker win should be described as an objective/feature-reranking diagnostic, not as evidence that Flow reward improved over proxy. This is a useful paper boundary: the method found where utility signal exists, then the ablation prevented overclaiming. The next defensible Flow experiment must train an official Flow/listwise checkpoint and require gains to survive no-flow and only-flow ablations before reopening the Flow-improvement claim.

