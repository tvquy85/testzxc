# Step 19 - V6 pipeline passes, claims remain restricted

Step 19 created the strict V6 science gate and claim matrix. The final decision
is:

- pipeline_decision: GO_V6_PIPELINE
- claim_decision: CLAIM_RESTRICTED
- strong_accept_ready: false

Allowed claims are limited to pipeline/diagnostic evidence: data/evidence
quality, judge quality, alignment artifacts, diagnostic alpha, comparable
baselines, and ablation table presence.

Blocked claims are the scientifically important ones:

- rationale quality remains limited by template/diversity preferred gates;
- Flow does not beat proxy on 2/3 core metrics;
- DPO does not improve over RWSFT/SFT;
- DPO does not beat Technical_Rule on Macro-F1 and MCC;
- paper-level alpha lacks confidence intervals;
- counterfactual faithfulness fails pass/no-change/news gates;
- statistical tests are missing.

Paper value:
- The project now has a clean claim-governance artifact. This is stronger than
  a vague negative result because every blocked claim has named evidence.
- The paper should frame V6 as a rigorous recovery and diagnostic pipeline, not
  as a strong-accept-ready result.
- Next work should prioritize prediction objective, evidence sensitivity, and
  statistical testing before any scale-up or paper-performance claim.
