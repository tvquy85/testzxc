# Step10 Flow Decision Dataset Stop Gate V6

Step10 implements the DFD-FlowReward-V6 dataset contract. It initially failed
the canonical row-scale gate on the 300-row judge subset, then passed after
Step08/09 were expanded to full 3000 candidate-row coverage.

The useful result is methodological rather than a performance claim:

- primary Flow target is exactly the calibrated judge distribution over five
  movement classes;
- realized utility and technical-rule delta are stored only as auxiliary
  evidence, not as replacement labels;
- reliability weights combine schema/permutation validity, argmax consistency,
  normalized label-order KL, normalized disagreement entropy, and evidence
  quality;
- train/val split is present and source test rows are excluded;
- all target rows sum to one.

The row gate first blocked progression because only 300 candidate rows had
independent judge distributions, while Step10 requires at least 2700 rows. This
is important for the paper narrative: the pipeline refused to train Flow or DPO
from a small judge subset, even though all tensor-shape and target-contract
checks passed. After full judge scale-up, Step10 passed with 3000 rows, target
dimension 5, condition dimension 128, train/val split 2361/639, and target
sum-ok rate 1.0.

Next valid scientific action: run Step11 Flow training/evaluation against proxy
and raw realized utility. `flow_claim_allowed` and downstream alignment claims
must remain false until those held-out utility/fidelity gates pass.
