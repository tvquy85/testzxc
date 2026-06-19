# 00 — Master execution order for Codex upgrade

## Purpose

Upgrade the current `testzxc` prototype from an internal FIRE-Fin MVP into an AAAI-grade reproducible research codebase. Do **not** ask Codex to do all tasks at once. Feed Codex **one file at a time** in the order below.

## Non-negotiable rules

1. Never fabricate metrics, tables, or status files.
2. Never mark a task `PASS` if any required output is missing.
3. Never use test data for training, reward construction, DPO pair construction, prompt tuning, or model selection.
4. Preserve raw LLM outputs. Post-processing output must be saved separately.
5. No dummy tables, no `N/A` main ablation rows, no silent fallback to neutral labels.
6. Every script must be runnable from repo root.
7. Every generated artifact must have a manifest entry with path, row count, hash, and timestamp.

## Recommended execution order

1. `01_REPO_AUDIT_AND_SAFE_BRANCH.md`
2. `02_CONFIG_PATHS_AND_REPRO_ENV.md`
3. `03_DATA_SCALE_AND_SPLIT_LOCK.md`
4. `04_LEAKAGE_GUARDS_AND_UNIT_TESTS.md`
5. `05_LABELS_ABNORMAL_RETURN_AND_BALANCE.md`
6. `06_TECHNICAL_FEATURES_V2.md`
7. `07_TECH_EVENT_TOKENS_V2.md`
8. `08_STRICT_RATIONALE_SCHEMA_NO_AUTOFIX.md`
9. `09_RATIONALE_GENERATION_SCALEUP_TRAIN_ONLY.md`
10. `10_PROXY_JUDGES_MULTI_MODEL_DEBIASED.md`
11. `11_CLAIM_LEVEL_GROUNDING_JUDGES.md`
12. `12_FLOW_REWARD_MULTITARGET_V2.md`
13. `13_FLOW_REWARD_EVAL_VS_PROXY.md`
14. `14_RWSFT_DPO_DATASET_REBUILD.md`
15. `15_ALIGNMENT_TRAINING_REPRODUCIBLE.md`
16. `16_DAILY_PORTFOLIO_BACKTEST_V2.md`
17. `17_COUNTERFACTUAL_DIRECTIONAL_EVAL_V2.md`
18. `18_BASELINES_EXPANSION_AND_SEEDS.md`
19. `19_ABLATION_AND_STATISTICAL_TESTS.md`
20. `20_PAPER_TABLES_NO_DUMMY_GATES.md`
21. `21_FINAL_REPRO_PACKAGE_AND_AAAI_GATE.md`

## Global directory contract

Use these directories:

```text
data/raw/
data/processed/
data/labels/
data/indicators/
data/rationales/raw/
data/rationales/parsed/
data/judges/
data/reward/
data/alignment/
outputs/metrics/
outputs/tables/
outputs/figures/
outputs/status/
outputs/manifests/
outputs/audit/
checkpoints/
configs/
tests/
```

## Global verification command

After every step:

```bash
python -m src.utils.verify_status --status outputs/status/<STEP>.status.json
```

If `src.utils.verify_status` does not exist, create it in Step 01.
