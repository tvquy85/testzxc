# Step 17 - Technical_Rule remains the strongest baseline

Step 17 created a current-data comparable baseline table on the same 300 V6
prediction contexts used by Step 14. The table includes six comparable rows and
two explicit reference-only rows:

- Technical_Rule;
- Qwen_DPO_V6;
- Qwen_RWSFT_V6;
- SEP_Style_Summarize_Explain;
- Policy_Style_Scalar_Proxy;
- Neutral_Hold_Majority;
- Qwen_Base_NoAlign as reference-only because no current-data base prediction
  artifact was produced;
- PEN_Reference_Only because no current-data PEN adapter/prediction artifact
  exists.

The result is negative for the forecast-superiority claim. Technical_Rule is
best by Macro-F1 at 0.2277 and MCC 0.0466. Qwen_DPO_V6 reaches Macro-F1 0.1124
and MCC 0.0119, so it fails both required comparisons. Qwen_RWSFT_V6 is also
below Technical_Rule.

Paper value:
- This supports an honest claim boundary: the current V6 aligned LLM should not
  be presented as a better forecaster than the deterministic technical rule.
- The stronger story remains methodology and diagnostics, not headline forecast
  dominance.
- Future work should target the prediction/action objective before scaling
  architecture, because the simple technical rule still captures more label
  signal than the current DPO/RWSFT adapters.
