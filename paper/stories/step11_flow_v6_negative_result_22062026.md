# Step11 Flow V6 Negative Result

Step11 completed a full current-data Flow V6 training/evaluation run on the
3000-row decision-distribution reward dataset. The result is scientifically
useful because it separates pipeline reproducibility from method claims.

Training succeeded:

- 3000 rows, 2361 train, 639 validation;
- 80 epochs on CUDA;
- best validation loss 0.2227;
- loss decreased over training;
- model checkpoint saved at `outputs/models/flow_reward_v6_decision.pt`.

Evaluation did not support a Flow claim:

- Flow beats proxy on rank correlation with raw realized utility;
- Flow loses on preference-pair accuracy;
- Flow loses on top-decile raw realized utility;
- distributional fidelity is not better than the proxy anchor;
- reliability/grounding ablation checkpoints were not run, so the stricter
  claim rule must remain false.

This is a strong paper-story guardrail: DFD-FlowReward-V6 is implemented and
auditable, but the current evidence says not to claim that it beats proxy on
current data. The next research action should investigate why the model learns
rank signal but fails pair/top-decile utility, especially whether the target
distribution is too neutral-heavy, whether reliability weighting suppresses
tail decisions, and whether the condition embedding is too weak.
