# Step 16 - Counterfactual faithfulness remains blocked

Step 16 upgraded the counterfactual setup from weak headline/body perturbations
to evidence-level perturbations across seven task types:

- remove positive/negative company evidence;
- neutralize positive/negative company evidence;
- remove all company evidence;
- neutralize bearish/bullish technical signals.

The artifact quality gate passed. The final run used 350 balanced tasks, 50 per
task type, with generation-level schema_ok_rate 0.9771 and pair_schema_ok_rate
0.9543.

The scientific result is negative. Overall directional pass_rate was 0.3257,
below the 0.50 claim gate, and no_change_rate was 0.3657, above the 0.35 gate.
The model passed remove_negative_evidence at 0.54, but failed
remove_positive_evidence at 0.30. Neutralization was especially weak:
neutralize_positive_evidence and neutralize_negative_evidence had pass rates
0.18 and 0.14, both with no_change_rate 0.68.

Paper value:
- This is useful negative evidence. The model can produce schema-valid
  counterfactual forecasts, but its directional response is not reliably
  faithful to evidence-level perturbations.
- The paper should not claim evidence-level counterfactual faithfulness from
  the current DPO model.
- A future fix should target training data and loss design for contrastive
  evidence sensitivity, not only prompt wording. Neutralization tasks are the
  sharpest failure signal.
