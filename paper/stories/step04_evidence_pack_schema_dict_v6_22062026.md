# Step 04 - Evidence pack schema mismatch fixed

Step 04 initially failed because the repair validator assumed an evidence pack
was a flat list. The actual V6 artifact uses a structured dict:

- company_evidence;
- context_evidence;
- technical_signals.

The repair code was updated to validate and preserve this schema. The final
Step 04 status is PASS with 3129 rows, 3 repaired rows, and 0 post-repair
failures.

Paper value:
- This supports a clear data-engineering story: evidence packs are structured
  by source and signal type rather than treated as flat text blobs.
- Validation should be schema-aware; otherwise a valid structured artifact can
  be misclassified as invalid.
