# Step 15.6 Story - Validation-Selected Trading Policy V6

Step 15.5 found that sparse trading policies could look better than `Technical_Rule`, but the strongest row was post-hoc and therefore too easy to overclaim. Step 15.6 converted that observation into a stricter protocol: choose a DPO side-confidence threshold and daily position cap on validation only, lock the policy, and evaluate once on test.

The validation grid selected threshold `0.61` and cap `3`. On the held-out test calendar, the locked DPO policy reached Sharpe `1.6840` and mean daily return `0.001813`, with CI-supported positive deltas versus `Technical_Rule`. The important negative result is that absolute Sharpe and mean-return CIs still crossed zero, so the paper-level alpha claim remains blocked.

Paper angle: this is a useful "claim discipline" story. The model shows decision-useful signal under a preregistered validation rule, but the uncertainty interval is still too wide for an AAAI-ready alpha claim. The right next step is fresh-horizon or larger current-data evaluation, not stronger wording.
