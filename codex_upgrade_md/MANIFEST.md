# MANIFEST — Codex upgrade task files

Use this manifest to feed files to Codex one by one. Do not paste the full folder into one Codex prompt.

## Files

1. `00_MASTER_CODEX_EXECUTION_ORDER.md`
2. `01_REPO_AUDIT_AND_SAFE_BRANCH.md`
3. `02_CONFIG_PATHS_AND_REPRO_ENV.md`
4. `03_DATA_SCALE_AND_SPLIT_LOCK.md`
5. `04_LEAKAGE_GUARDS_AND_UNIT_TESTS.md`
6. `05_LABELS_ABNORMAL_RETURN_AND_BALANCE.md`
7. `06_TECHNICAL_FEATURES_V2.md`
8. `07_TECH_EVENT_TOKENS_V2.md`
9. `08_STRICT_RATIONALE_SCHEMA_NO_AUTOFIX.md`
10. `09_RATIONALE_GENERATION_SCALEUP_TRAIN_ONLY.md`
11. `10_PROXY_JUDGES_MULTI_MODEL_DEBIASED.md`
12. `11_CLAIM_LEVEL_GROUNDING_JUDGES.md`
13. `12_FLOW_REWARD_MULTITARGET_V2.md`
14. `13_FLOW_REWARD_EVAL_VS_PROXY.md`
15. `14_RWSFT_DPO_DATASET_REBUILD.md`
16. `15_ALIGNMENT_TRAINING_REPRODUCIBLE.md`
17. `16_DAILY_PORTFOLIO_BACKTEST_V2.md`
18. `17_COUNTERFACTUAL_DIRECTIONAL_EVAL_V2.md`
19. `18_BASELINES_EXPANSION_AND_SEEDS.md`
20. `19_ABLATION_AND_STATISTICAL_TESTS.md`
21. `20_PAPER_TABLES_NO_DUMMY_GATES.md`
22. `21_FINAL_REPRO_PACKAGE_AND_AAAI_GATE.md`
23. `SOURCES_AND_REVIEW_NOTES.md`

## Suggested Codex prompt wrapper

Paste this before each task file:

```text
You are modifying the GitHub repo tvquy85/testzxc. Implement only the task below. Do not implement later tasks. Do not fabricate outputs. Create or modify the named files only unless necessary. After coding, run the verification commands or explain exactly why they cannot run. Mark status PASS only when acceptance criteria are satisfied.
```
