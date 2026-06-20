# FIRE-Fin Current-Data Codex Upgrade Pack

Use this pack to upgrade the current `upgrade-aaai-reproducibility` branch **before scaling data**.

## Why this pack exists
The branch is reproducible but weak scientifically:
- Flow v2 does not beat proxy reward.
- Counterfactual pass rate is low.
- Daily backtest is negative and short.
- Inferability depends too much on generator `forecast_distribution`.
- Grounding is mostly lexical, with many unverified claims.
- Ablations are registered but not actually run.

## Feed files to Codex in order
0. `00_MASTER_CURRENTDATA_UPGRADE_ORDER.md`
1. `01_FREEZE_BASELINE_AND_BRANCH.md`
2. `02_CURRENT_DATA_QUALITY_AUDIT.md`
3. `03_ENTITY_EVENT_FILTER_CURRENT_DATA.md`
4. `04_TICKER_DATE_CONTEXT_AGGREGATION.md`
5. `05_LABEL_BACKTEST_TARGET_FIX_ABNORMAL_RETURN.md`
6. `06_RATIONALE_PROMPT_CLEAN_CONTEXT.md`
7. `07_INDEPENDENT_INFERABILITY_JUDGE.md`
8. `08_JUDGE_DEBIAS_LABEL_ORDER_RANDOMIZATION.md`
9. `09_CLAIM_EXTRACTION_GROUNDING_V2.md`
10. `10_FLOW_DATASET_V3_TARGETS_AND_EMBEDDINGS.md`
11. `11_FLOW_TRAIN_EVAL_FIX_VALID_SPLIT.md`
12. `12_RWSFT_DPO_REBUILD_FROM_INDEPENDENT_REWARDS.md`
13. `13_ALIGNMENT_REAL_RUN_CURRENT_DATA.md`
14. `14_DAILY_BACKTEST_ABNORMAL_AND_COSTS.md`
15. `15_COUNTERFACTUAL_DIRECTIONAL_FIX_CURRENT_DATA.md`
16. `16_MINIMUM_ABLATIONS_CURRENT_DATA.md`
17. `17_PAPER_TABLES_NEGATIVE_RESULTS_GATES.md`
18. `18_AAAI_SCIENCE_GATE_STRICT.md`
19. `19_CURRENTDATA_RUNBOOK.md`

See `SOURCES_AND_RATIONALE.md` for references.
