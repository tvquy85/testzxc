# Step 16.5 Story - Counterfactual Failure Comes From Prediction-Side Saturation

The Step 16 counterfactual faithfulness gate failed, but the follow-up eligibility audit shows why rerunning the same perturbations is unlikely to help.

Key evidence:

```text
eligible_side_signal_rate: 0.4229
up_expected_eligible_rate: 0.7179
down_expected_eligible_rate: 0.0516
down_expected_mean_down_side: 0.0239
down_expected_zero_down_side_rate: 0.8774
```

Interpretation:

Many `down_decreases` tasks ask whether removing/neutralizing negative evidence reduces bearish belief, but the DPO model assigns almost no bearish probability in the original prediction. This makes the directional expectation test weakly interpretable: there is little or no original down-side signal to reduce.

Paper value:

- This supports a stronger claim-boundary story: the counterfactual gate failed for a model-calibration reason, not a parser/schema reason.
- It connects the counterfactual blocker to the forecast blocker: DPO is under-using down-side classes and does not beat Technical_Rule.
- The next legitimate fix is prediction calibration/class-balance/counterfactual training, not lowering the counterfactual threshold or rewording the same perturbations.

Potential paper wording:

```text
We audited counterfactual directional eligibility and found that most down-decrease perturbations were applied to examples where the model assigned near-zero bearish probability before perturbation. We therefore interpret the counterfactual failure as evidence of prediction-side saturation and class-balance weakness, and keep faithfulness claims blocked pending calibrated contrastive training.
```
