# Step08 Judge Raw-vs-Usable Repair V6

Step08 V6 separates raw schema compliance from usable probabilistic evidence.
The judge output is audited under a strict raw probability-sum contract, but a
second usable distribution is allowed when the output is complete, non-negative,
near the probability simplex, and deterministic repair does not change the
argmax class.

The motivation came from smoke tests where Qwen3-4B-Instruct produced stable
label-order decisions but sometimes left probability mass unassigned. Treating
this as silent PASS would weaken the audit trail, while discarding otherwise
consistent judgments would waste useful signal. The implemented compromise keeps
`raw_schema_ok` for compliance reporting and uses `usable_schema_ok` for
downstream reward only after explicit, logged repair.

This distinction is useful for the paper because it makes the judge pipeline
reproducible and falsifiable: formatting arithmetic failures, label-order
stability, repair frequency, and true-label probability are all reported
separately instead of being collapsed into one pass/fail number.
