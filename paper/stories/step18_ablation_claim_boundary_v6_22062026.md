# Step 18 - Ablations reinforce the claim boundary

Step 18 created an ablation evidence table over the current V6 artifacts. The
table intentionally separates executed artifacts from diagnostic proxies and
reference-only rows. No row is marked NOT_RUN, but rows without a true
prediction artifact are labeled as DIAGNOSTIC_ONLY, DIAGNOSTIC_PROXY, or
REFERENCE_ONLY.

The result reinforces the negative forecast story:

- Full_V6 (Qwen_DPO_V6) Macro-F1 is 0.1124.
- No_News_Evidence / Technical_Only proxies map to Technical_Rule at Macro-F1
  0.2277.
- No_DPO_RWSFT_Only (Qwen_RWSFT_V6) remains below Technical_Rule.
- Base_Qwen_NoAlign is reference-only because no current-data base prediction
  artifact exists.

Paper value:
- This table is useful for transparent claim governance. It prevents the paper
  from implying that every ablation was a full inference run when some are
  diagnostic proxies.
- The ablation evidence supports a conservative framing: current V6 does not
  yet convert stronger evidence handling and alignment machinery into better
  directional prediction.
- The next scientific improvement should target prediction objective and
  evidence sensitivity, not only more pipeline scale.
