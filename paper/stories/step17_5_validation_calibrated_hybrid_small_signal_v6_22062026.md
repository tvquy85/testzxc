# Step 17.5 Story - Validation-Calibrated Hybrid Finds a Small Forecast Signal

After the counterfactual eligibility audit showed DPO side-probability saturation, we tested a leak-safe post-hoc decision system:

```text
Tune DPO confidence threshold on validation only.
Use DPO prediction only when schema-valid and confidence >= threshold.
Otherwise fall back to Technical_Rule.
```

Validation selected threshold:

```text
best_threshold: 0.71
val_dpo_use_rate: 0.2833
```

On the held-out test set:

```text
test_dpo_use_rate: 0.2167
hybrid Macro-F1: 0.2321
Technical_Rule Macro-F1: 0.2277
hybrid MCC: 0.0503
Technical_Rule MCC: 0.0466
```

But paired bootstrap intervals do not support a paper-level claim:

```text
delta Macro-F1 CI95: [-0.02464, 0.03220]
delta MCC CI95: [-0.03249, 0.03761]
paired_ci_support: false
```

Paper value:

- The result is scientifically useful because it shows DPO contains some high-confidence signal, even though the raw aligned model is weaker than Technical_Rule.
- The effect is too small/noisy for a claim, so the correct framing is diagnostic/future-work.
- This points to a concrete next step: train/calibrate DPO to preserve high-confidence correct moves while fixing class-balance and down-side saturation.

Potential paper wording:

```text
A validation-calibrated confidence gate slightly improved point estimates over the technical rule baseline, but paired bootstrap intervals included zero. We therefore report it as evidence that aligned forecasts contain recoverable signal, while keeping forecast-superiority claims blocked.
```
