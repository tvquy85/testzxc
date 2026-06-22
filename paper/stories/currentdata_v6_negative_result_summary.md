# Current-Data V6 Negative Result Summary

Final decision: `CLAIM_RESTRICTED`.

The current-data V6 pipeline improves reproducibility and evidence grounding, but does not yet establish strong forecast or trading improvement. Flow reward should be interpreted as a diagnostic experiment until it beats proxy. Repaired DPO now beats RWSFT on point metrics, but still does not beat the technical baseline under strict validation.

Blocked claims:
- flow_reward_improvement: Official Flow did not beat proxy on at least 2/3 core metrics; Step 11.6 reranker wins rank/pair, but Step 11.7 no-flow ablation matches or exceeds the full feature set while only-flow underperforms, so the diagnostic win is not attributable to Flow
- forecast_beats_technical_rule: Repaired DPO still did not beat Technical_Rule on both Macro-F1 and MCC; validation-stacked and supervised-ceiling diagnostics have no CI-supported test improvement
- trading_alpha_paper_level: requires Sharpe > 0, >=120 days, and confidence intervals that support positive alpha; Step 15.5 is post-hoc diagnostic only and Step 15.6 validation-selected DPO still has absolute Sharpe/mean-return CIs crossing zero
