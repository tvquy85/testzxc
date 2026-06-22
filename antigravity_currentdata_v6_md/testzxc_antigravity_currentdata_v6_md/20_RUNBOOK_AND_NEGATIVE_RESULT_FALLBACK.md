# 20 — Runbook and Negative-Result Fallback

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Create a final runbook. If V6 cannot support strong claims, produce an honest negative-result package.

## Outputs
```text
RUNBOOK_currentdata_v6.md
paper/stories/currentdata_v6_negative_result_summary.md
outputs/status/20_RUNBOOK_AND_NEGATIVE_RESULT_FALLBACK.status.json
```

## Required runbook sections
Environment/model paths; current-data scope; exact command order; artifacts by step; claim matrix; known failures; table reproduction; what can/cannot be claimed; next full-scale path.

## Negative-result fallback text
```text
The current-data V6 pipeline improves reproducibility and evidence grounding, but does not yet establish strong forecast or trading improvement. Flow reward should be interpreted as a diagnostic experiment until it beats proxy. Repaired DPO now beats RWSFT on point metrics, but still does not beat the technical baseline under strict validation.
```

## Test case
```python
from pathlib import Path

def test_runbook_mentions_claim_restriction():
    p=Path('RUNBOOK_currentdata_v6.md')
    assert p.exists()
    text=p.read_text(encoding='utf-8').lower()
    assert 'claim' in text and ('restricted' in text or 'blocked' in text)
```

## Commands
```bash
python -m src.repro.write_v6_runbook --gate outputs/repro/currentdata_v6_strong_accept_gate.json --claim-table outputs/tables/19_v6_claim_matrix.csv --output RUNBOOK_currentdata_v6.md --negative-summary paper/stories/currentdata_v6_negative_result_summary.md --status outputs/status/20_RUNBOOK_AND_NEGATIVE_RESULT_FALLBACK.status.json
python -m pytest -q tests/test_runbook_v6.py tests
```

## Progress Update 2026-06-22
Status: `PASS`; status file reports `next_step_allowed=true`.

Implementation notes:
```text
src/repro/write_v6_runbook.py
tests/test_runbook_v6.py
```

Artifacts verified:
```text
RUNBOOK_currentdata_v6.md
paper/stories/currentdata_v6_negative_result_summary.md
outputs/manifests/20_RUNBOOK_AND_NEGATIVE_RESULT_FALLBACK.manifest.json
outputs/status/20_RUNBOOK_AND_NEGATIVE_RESULT_FALLBACK.status.json
```

Final command run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.repro.write_v6_runbook --gate outputs/repro/currentdata_v6_strong_accept_gate.json --claim-table outputs/tables/19_v6_claim_matrix.csv --output RUNBOOK_currentdata_v6.md --negative-summary paper/stories/currentdata_v6_negative_result_summary.md --status outputs/status/20_RUNBOOK_AND_NEGATIVE_RESULT_FALLBACK.status.json --manifest outputs/manifests/20_RUNBOOK_AND_NEGATIVE_RESULT_FALLBACK.manifest.json
```

Final metrics:
```text
claim_decision: CLAIM_RESTRICTED
runbook_written: true
negative_summary_written: true
claim_rows: 14
pipeline_pass: true
claim_allowed: false
```

Gate context after Step 18.5:
```text
statistical_tests: allowed
trading_alpha_paper_level: blocked because CI does not support positive alpha
strong_accept_ready: false
```

Trading policy variant context after Step 15.5:
```text
trading_policy_variant_probe: PASS
strategy_count: 6
num_trading_days: 173
best_strategy_by_sharpe: Supervised_LogReg_TFIDF
best_strategy_sharpe: 1.4379
best_strategy_ci_support_vs_zero: false
best_strategy_ci_support_vs_technical: true
dpo_official_sharpe: 0.9667
repaired_dpo_official_sharpe: 1.2002
multiple_testing_warning: true
trading_alpha_paper_level: still blocked
```

Validation-selected trading context after Step 15.6:
```text
validation_selected_trading_policy: PASS
selected_threshold: 0.61
selected_position_cap: 3
val_sharpe: 5.4164
test_sharpe: 1.6840
test_sharpe_ci95: [-0.9523, 4.0931]
test_mean_daily_return: 0.001813
test_mean_daily_return_ci95: [-0.001080, 0.004889]
delta_sharpe_vs_technical: 3.1051, CI95 [0.7142, 5.4266]
delta_mean_return_vs_technical: 0.004138, CI95 [0.000959, 0.007316]
alpha_paper_level_supported: false
trading_alpha_paper_level: still blocked
```

Counterfactual root-cause context after Step 16.5:
```text
counterfactual_eligibility_audit: PASS
eligible_side_signal_rate: 0.4229
down_expected_eligible_rate: 0.0516
counterfactual_faithfulness: Step 16.5 explains the original failure; final claim is determined after Steps 16.6/16.7
```

Counterfactual quality-filtered context after Step 16.6:
```text
counterfactual_quality_filtered_tasks: PASS
candidate_tasks: 1495
quality_pass_tasks: 1217
selected_tasks: 192
counterfactual_quality_filtered_eval: PASS
pass_rate: 0.4844
no_change_rate: 0.2604
schema_ok_rate: 0.9974
remove_positive_evidence_pass_rate: 0.5000
remove_negative_evidence_pass_rate: 0.7143
neutralize_negative_evidence_pass_rate: 0.0952
news_faithfulness_claim_allowed: true
general_counterfactual_claim_allowed: false
counterfactual_faithfulness: still blocked at Step 16.6; resolved by Step 16.7 semantic-neutralized rerun
```

Counterfactual semantic-neutralized context after Step 16.7:
```text
counterfactual_semantic_neutralized_tasks: PASS
semantic_neutralization: true
selected_tasks: 192
semantic_neutralization_repaired_selected_rate: 0.2656
counterfactual_semantic_neutralized_eval: PASS
pass_rate: 0.5313
no_change_rate: 0.2083
schema_ok_rate: 0.9948
remove_positive_evidence_pass_rate: 0.5000
remove_negative_evidence_pass_rate: 0.7143
neutralize_negative_evidence_pass_rate: 0.5238
news_faithfulness_claim_allowed: true
general_counterfactual_claim_allowed: true
counterfactual_faithfulness: allowed
```

Forecast calibration context after Step 17.5:
```text
validation_calibrated_hybrid: PASS
best_threshold: 0.71
delta_macro_f1_vs_technical: 0.00436
delta_mcc_vs_technical: 0.00371
paired_ci_support: false
forecast_beats_technical_rule: still blocked
```

Forecast repair context after Steps 14.6 and 17.8:
```text
dpo_repaired_schema_ok_rate: 1.0000
dpo_repaired_macro_f1: 0.1305
dpo_repaired_mcc: 0.0269
rwsft_repaired_macro_f1: 0.1202
rwsft_repaired_mcc: -0.0052
technical_rule_macro_f1: 0.2277
technical_rule_mcc: 0.0466
alignment_improves_over_sft: allowed on point Macro-F1 and MCC
forecast_beats_technical_rule: still blocked
```

Forecast stacking context after Step 17.6:
```text
validation_stacked_forecast_probe: PASS
best_c: 10.0
best_class_weight: none
val_stacked_macro_f1: 0.3913
val_stacked_mcc: 0.2531
test_stacked_macro_f1: 0.2017
test_stacked_mcc: 0.0216
test_technical_macro_f1: 0.2277
test_technical_mcc: 0.0466
delta_macro_f1_vs_technical: -0.0260
delta_mcc_vs_technical: -0.0250
paired_ci_support: false
validation_overfit_warning: true
forecast_beats_technical_rule: still blocked
```

Supervised signal-ceiling context after Step 17.7:
```text
supervised_signal_ceiling_probe: PASS
best_c: 0.03
train_rows: 2453
train_max_date: 2018-12-17
val_supervised_macro_f1: 0.1469
val_technical_macro_f1: 0.2041
test_supervised_macro_f1: 0.1480
test_technical_macro_f1: 0.2277
test_supervised_mcc: 0.0776
test_technical_mcc: 0.0466
delta_macro_f1_vs_technical: -0.0797
delta_mcc_vs_technical: 0.0310
paired_ci_support: false
signal_ceiling_warning: true
forecast_beats_technical_rule: still blocked
```

Rationale quality context after Step 07.5:
```text
rationale_template_decomposition: PASS
news_plus_meta_mean_jaccard: 0.6447
news_plus_meta_template_cluster_rate: 0.2257
news_repeated_phrase_rate: 0.0000
rationale_quality: allowed
```

Flow root-cause context after Step 11.5:
```text
flow_utility_surface_diagnostic: PASS
utility_varying_pair_rate: 0.1142
flow_rank_win_vs_proxy: true
flow_pair_accuracy_gap_vs_proxy: -0.0812
flow_top_decile_gap_vs_proxy: -0.00241
flow_top_decile_overlap_with_technical: 0.0000
flow_reward_improvement: still blocked
```

Flow reranker context after Step 11.6:
```text
flow_utility_reranker_probe: PASS
selected_method: pairwise_logistic
selected_regularization: 0.01
train_core_win_count_vs_proxy: 3
eval_core_win_count_vs_proxy: 2
eval_rank: 0.3890 vs proxy 0.1967
eval_pair: 0.6575 vs proxy 0.6164
eval_top_decile: 0.0000 vs proxy 0.002353
flow_reward_improvement: still blocked for official checkpoint
```

Flow attribution context after Step 11.7:
```text
flow_reranker_ablation_attribution: PASS
flow_attribution_supported: false
no_flow_matches_or_exceeds_full: true
only_flow_underperforms_full: true
full val rank/pair: 0.3890/0.6575
no_flow val rank/pair: 0.4267/0.6849
only_flow val rank/pair: 0.3383/0.5352
flow_reward_improvement: still blocked because the reranker win is not Flow-attributed
```

Acceptance result:
```text
status JSON gate: PASS
runbook exists: PASS
negative-result summary exists: PASS
claim restriction text present: PASS
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m py_compile src\repro\write_v6_runbook.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/20_RUNBOOK_AND_NEGATIVE_RESULT_FALLBACK.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_runbook_v6.py tests/test_strong_accept_gate_v6.py
```

Final interpretation: V6 is now documented as a runnable, claim-restricted current-data recovery pipeline. Step 14.6 repairs probability-schema issues and opens the alignment-over-RWSFT point claim, while Step 16.7 semantic-neutralized counterfactual evaluation opens the counterfactual-faithfulness gate. Statistical tests/CI still block forecast superiority, paper-level alpha, and Flow. It is not strong-accept ready.
