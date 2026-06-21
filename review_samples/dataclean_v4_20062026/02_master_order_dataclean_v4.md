# 00 — Master Order: Current-Data DataClean V4 Upgrade

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Why V4 is needed
The `currentdata-aaai-fix-v2` branch is reproducible but not AAAI-ready. Reported small-scale results show: Flow v3.1 underperforms proxy (`0.0565` vs `0.5514` rank correlation), backtest Sharpe is negative (`-1.3467`), counterfactual faithfulness is partial (`0.422` pass), and samples show no-news / weak-news / multi-company contexts. The next upgrade must improve **current data quality first** before rerunning rationale, judges, flow, DPO, and backtest.

## Execution order
Run one file at a time:

1. `01_FREEZE_CURRENTDATA_V3_BASELINE.md`
2. `02_AUDIT_REVIEW_SAMPLES_AND_METRICS.md`
3. `03_ENTITY_ALIAS_MAP_V4.md`
4. `04_ENTITY_EVENT_SCORING_V4.md`
5. `05_ARTICLE_TYPE_AND_NOISE_FILTER_V4.md`
6. `06_DEDUP_NEWS_V4.md`
7. `07_BUILD_EVIDENCE_PACK_CONTEXTS_V4.md`
8. `08_RENDER_CONTEXT_EVIDENCE_V4.md`
9. `09_RATIONALE_PROMPT_EVIDENCE_ID_V4.md`
10. `10_STRICT_EVIDENCE_SCHEMA_VALIDATION.md`
11. `11_EVIDENCE_GROUNDING_JUDGE_V4.md`
12. `12_TRACK_SPLIT_AND_TRAIN_POOL_V4.md`
13. `13_REGENERATE_RATIONALES_V4.md`
14. `14_INDEPENDENT_JUDGE_RERUN_EVIDENCE_V4.md`
15. `15_FLOW_REWARD_REBUILD_CLEAN_V4.md`
16. `16_ALIGNMENT_REBUILD_CLEAN_V4.md`
17. `17_BACKTEST_COUNTERFACTUAL_ABLATION_V4.md`
18. `18_SCIENCE_GATE_AND_RUNBOOK_V4.md`

## Non-negotiable rules
- Train/reward/alignment use train only.
- Test is only for final evaluation.
- Keep `pipeline_pass` separate from `claim_allowed`.
- Do not zero-fill, dummy-fill, or hide negative results.
- Do not claim full-scale FNSPID or AAAI-readiness from this current-data run.

## Final artifact
`outputs/repro/currentdata_clean_v4_science_gate_report.json`
