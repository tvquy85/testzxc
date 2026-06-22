# Step09 Judge Calibration and Debias V6

Step09 turns the judge ensemble from a schema/consistency artifact into an
audited probabilistic signal. The key change before calibration was to separate
two facts that should not be collapsed:

- aggregate probabilities should remain usable when at least one label-order
  variant is valid;
- `judge_schema_ok` should still report whether all requested permutations
  passed.

This keeps downstream reward construction from receiving all-zero probability
vectors while preserving the audit trail for failed permutations. The resulting
Step08 artifact has 300/300 aggregate rows summing to one, while still showing
that 297 rows have all three valid permutations, 2 rows have two valid
permutations, and 1 row has one valid permutation.

Calibration uses Brier score, NLL, top-label ECE, one-vs-rest ECE, reliability
bins, and target-label breakdowns. The method is grounded in standard
calibration and proper-scoring-rule literature: Guo et al. (2017) for ECE and
reliability framing, Brier/proper scoring rules for probabilistic forecast
evaluation, and recent LLM option-order studies for the label-order robustness
motivation.

Full-scale result is a technical pass, not a strong judge claim:

- mean true-label probability is 0.2412, above the Step09 stop threshold 0.20;
- top-label accuracy is 0.257, only modestly above 5-class random;
- top-label ECE is 0.2396, so confidence is not well calibrated;
- strong_down is usable in the full-scale pass, but strong_up remains weak with
  zero top-label accuracy.

Paper angle: this is a useful negative-control story. The pipeline does not
pretend that a stable LLM judge is automatically calibrated. It explicitly
separates label-order stability from probabilistic calibration and blocks
standalone judge-quality claims unless downstream utility and held-out
prediction evidence improve.
