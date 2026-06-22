# Step 14.6 / 17.8 Story - Forecast Distribution Repair Boundary

Date: 2026-06-22

The DPO prediction artifact had 25 parse-ok outputs with all five forecast probabilities present, but the probability mass did not sum to 1.0. The repair normalizes only these label-free probability vectors and rederives labels/actions. It does not use target labels.

Observed effect:

```text
DPO original schema_ok_rate: 0.9167
DPO repaired schema_ok_rate: 1.0000
DPO Macro-F1/MCC before repair: 0.1124 / 0.0119
DPO Macro-F1/MCC after repair: 0.1305 / 0.0269
RWSFT Macro-F1/MCC: 0.1202 / -0.0052
Technical_Rule Macro-F1/MCC: 0.2277 / 0.0466
```

This creates a narrow positive alignment signal: repaired DPO beats RWSFT on Macro-F1 and MCC. It does not solve the main forecast claim because Technical_Rule remains much stronger. The paper-safe story is that probability-schema repair matters and has now been made canonical for deterministic downstream evaluation, but it is not enough to claim forecast superiority.

Promotion result: downstream baseline, backtest, counterfactual eligibility, ablation, statistical tests, and strict gate were rerun with repaired predictions. Step 19 opens `alignment_improves_over_sft` on point Macro-F1 and MCC, while Flow, forecast superiority, paper-level alpha, and counterfactual faithfulness remain blocked.
