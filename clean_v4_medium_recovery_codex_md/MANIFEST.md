# MANIFEST — Clean V4 Medium Recovery Codex Pack

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Purpose
This package converts the small-scale DataClean V4 diagnostic run into a **medium-scale current-data validation**. It directly targets the known weaknesses from the reviewer analysis:

- DataClean V4 is stronger but still small-scale.
- Flow V4 shows only small-slice signal; it needs semantic conditioning and true validation.
- Backtest is still negative and too short; alpha must remain blocked unless gates pass.
- News counterfactuals remain weaker than technical counterfactuals.
- Rationale outputs are schema-stable but may be template-heavy.
- V4 adapter was not trained/evaluated at sufficient medium scale.
- PEN/SEP/Policy exist in the repo but are not yet converted into competitive reference baselines.

## Execution order
1. `00_MASTER_CLEAN_V4_MEDIUM_ORDER.md`
2. `01_FREEZE_CLEAN_V4_SMALL_BASELINE.md`
3. `02_AUDIT_CLEAN_V4_FAILURE_MODES.md`
4. `03_BUILD_MEDIUM_SAMPLE_SELECTOR.md`
5. `04_EVIDENCE_PACK_QUALITY_GATES_V4_1.md`
6. `05_GENERATE_RATIONALES_MEDIUM_500X3.md`
7. `06_RATIONALE_DIVERSITY_TEMPLATE_AUDIT.md`
8. `07_INDEPENDENT_JUDGE_MEDIUM_FULL.md`
9. `08_LABEL_ORDER_DEBIAS_MULTI_PERMUTATION.md`
10. `09_GROUNDING_NEWS_NEGATIVE_FIX.md`
11. `10_FLOW_SEMANTIC_EMBEDDINGS_V5.md`
12. `11_FLOW_TARGET_NORMALIZATION_AND_SPLIT.md`
13. `12_FLOW_TRAIN_EVAL_MEDIUM.md`
14. `13_ALIGNMENT_DATASET_MEDIUM_RWSFT_DPO.md`
15. `14_ALIGNMENT_TRAIN_ADAPTER_V4_MEDIUM.md`
16. `15_PREDICT_WITH_ADAPTER_V4_MEDIUM.md`
17. `16_BACKTEST_TRACK_BREAKDOWN_MEDIUM.md`
18. `17_COUNTERFACTUAL_EVIDENCE_MEDIUM.md`
19. `18_MINIMUM_BASELINES_PEN_SEP_POLICY.md`
20. `19_ABLATION_SUITE_MEDIUM.md`
21. `20_STRICT_SCIENCE_GATE_MEDIUM.md`
22. `SOURCES_AND_REVIEW_BASIS.md`

## Final artifacts expected
```text
outputs/repro/currentdata_clean_v4_medium_science_gate_report.json
outputs/tables/medium_claim_matrix.csv
outputs/tables/medium_baseline_comparison.csv
outputs/tables/medium_ablation_results.csv
outputs/tables/medium_track_breakdown.csv
```

## Non-negotiable claim discipline
- Do not claim AAAI-ready.
- Do not claim trading alpha unless the alpha gate allows it.
- Do not claim Flow Reward superiority unless Flow V5 beats proxy on validation.
- Do not claim multimodal news+technical reasoning unless Track A beats appropriate ablations.
