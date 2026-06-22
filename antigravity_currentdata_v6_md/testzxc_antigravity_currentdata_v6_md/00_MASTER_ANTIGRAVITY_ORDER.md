# 00 — Master Antigravity Execution Order for Current-Data V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Execute in this order

1. `01_FREEZE_MEDIUM_BASELINE_AND_TESTS.md`
2. `02_HARD_EVENT_DATA_AUDIT.md`
3. `03_HARD_EVENT_FILTER_V6.md`
4. `04_EVIDENCE_PACK_REPAIR_V6.md`
5. `05_RATIONALE_PROMPT_NEWS_USAGE_V6.md`
6. `06_GENERATE_RATIONALES_1000X3_V6.md`
6.5. `06_5_POST_PROCESSING_AND_REPAIR_V6.md`
7. `07_RATIONALE_DIVERSITY_AND_TEMPLATE_GATE.md`
7.5. `07_5_RATIONALE_TEMPLATE_DECOMPOSITION_V6.md`
8. `08_INDEPENDENT_JUDGE_ENSEMBLE_V6.md`
9. `09_JUDGE_CALIBRATION_AND_DEBIAS_TESTS.md`
10. `10_FLOW_REWARD_V6_DECISION_TARGETS.md`
11. `11_FLOW_EVAL_RAW_UTILITY_AND_PROXY.md`
11.5. `11_5_FLOW_UTILITY_SURFACE_DIAGNOSTIC_V6.md`
11.6. `11_6_FLOW_UTILITY_RERANKER_PROBE_V6.md`
11.7. `11_7_FLOW_RERANKER_ABLATION_ATTRIBUTION_V6.md`
12. `12_ALIGNMENT_PAIRS_MIN_GAP_AND_DIVERSITY.md`
13. `13_TRAIN_RWSFT_DPO_V6.md`
14. `14_PREDICT_WITH_V6_ADAPTERS.md`
14.6. `14_6_FORECAST_DISTRIBUTION_REPAIR_V6.md`
15. `15_BACKTEST_TRACK_BASELINE_V6.md`
15.5. `15_5_TRADING_POLICY_VARIANT_PROBE_V6.md`
15.6. `15_6_VALIDATION_SELECTED_TRADING_POLICY_V6.md`
16. `16_COUNTERFACTUAL_NEWS_EVIDENCE_V6.md`
16.5. `16_5_COUNTERFACTUAL_ELIGIBILITY_AUDIT_V6.md`
16.6. `16_6_COUNTERFACTUAL_QUALITY_FILTERED_TASKS_V6.md`
16.7. `16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_V6.md`
17. `17_BASELINES_SEP_POLICY_TECH_RULE.md`
17.5. `17_5_VALIDATION_CALIBRATED_FORECAST_HYBRID_V6.md`
17.6. `17_6_VALIDATION_STACKED_FORECAST_PROBE_V6.md`
17.7. `17_7_SUPERVISED_SIGNAL_CEILING_PROBE_V6.md`
17.8. `17_8_REPAIRED_FORECAST_BASELINE_PROBE_V6.md`
18. `18_ABLATIONS_V6.md`
18.5. `18_5_STATISTICAL_TESTS_AND_CI_V6.md`
19. `19_STRICT_AAAI_STRONG_ACCEPT_GATE.md`
20. `20_RUNBOOK_AND_NEGATIVE_RESULT_FALLBACK.md`

## Intent
This plan addresses current Clean V4 Medium failures: Flow does not beat proxy enough, alpha remains negative, counterfactual faithfulness is blocked, rationales are template-heavy, judge signal is near random, Technical_Rule beats aligned LLM, and PEN/SEP/Policy are not comparable baselines yet.

## Test case
Create `tests/test_v6_file_order.py`:
```python
from pathlib import Path

def test_v6_goal_and_master_exist():
    assert Path('Goal.md').exists() or Path('testzxc_antigravity_currentdata_v6_md/Goal.md').exists()
```
Run `python -m pytest -q tests/test_v6_file_order.py`.
