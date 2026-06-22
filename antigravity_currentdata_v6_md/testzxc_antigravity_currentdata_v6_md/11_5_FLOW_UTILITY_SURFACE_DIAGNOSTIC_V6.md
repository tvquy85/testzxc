# 11.5 - Flow Utility Surface Diagnostic V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Diagnose why Step 11 Flow wins rank correlation but fails pair accuracy and top-decile utility. This step does not retrain Flow and does not allow the Flow claim; it identifies the objective mismatch to guide the next model fix.

## Outputs
```text
outputs/metrics/11_5_v6_flow_utility_diagnostic.json
outputs/tables/11_5_v6_flow_method_summary.csv
outputs/tables/11_5_v6_flow_top_decile_overlap.csv
outputs/tables/11_5_v6_flow_score_deciles.csv
review_samples/currentdata_v6/11_5_flow_pair_disagreement_examples.jsonl
outputs/status/11_5_FLOW_UTILITY_SURFACE_DIAGNOSTIC_V6.status.json
```

## Method
- Count within-sample candidate pairs with non-tied raw realized utility.
- Compare Flow/proxy/single-judge/technical-rule method summaries.
- Measure top-decile utility and top-decile overlap.
- Save pair disagreement examples where Flow is wrong and proxy/technical is right.

The motivation follows learning-to-rank evidence: rank correlation, pairwise preference accuracy, and top-of-list utility are different objectives. A model can have acceptable global rank correlation while selecting the wrong top-decile region.

## Command
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.reward.diagnose_flow_utility_v6 --predictions outputs/tables/11_v6_flow_predictions.csv --split val --metrics outputs/metrics/11_5_v6_flow_utility_diagnostic.json --summary-table outputs/tables/11_5_v6_flow_method_summary.csv --overlap-table outputs/tables/11_5_v6_flow_top_decile_overlap.csv --quantile-table outputs/tables/11_5_v6_flow_score_deciles.csv --examples review_samples/currentdata_v6/11_5_flow_pair_disagreement_examples.jsonl --status outputs/status/11_5_FLOW_UTILITY_SURFACE_DIAGNOSTIC_V6.status.json --manifest outputs/manifests/11_5_FLOW_UTILITY_SURFACE_DIAGNOSTIC_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_flow_utility_diagnostic_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/11_5_FLOW_UTILITY_SURFACE_DIAGNOSTIC_V6.status.json
```

## Acceptance
```text
status = PASS
eval_rows >= 300
candidate_pair_count is reported
utility_varying_pair_rate is reported
top-decile overlap table exists
```

## Progress Update 2026-06-22
Status: `PASS`; diagnostic explains the Step 11 negative result.

Key metrics:
```text
eval_rows: 639
candidate_pair_count: 639
utility_varying_pair_count: 73
utility_varying_pair_rate: 0.1142
samples_with_utility_variation_rate: 0.1690
flow_rank_win_vs_proxy: true
flow_pair_accuracy_gap_vs_proxy: -0.0812
flow_top_decile_gap_vs_proxy: -0.00241
flow_top_decile_gap_vs_technical: -0.00305
flow_top_decile_overlap_with_proxy: 0.0840
flow_top_decile_overlap_with_technical: 0.0000
claim_allowed: false
```

Interpretation:
```text
The current Flow model learns a useful global ranking signal, but pairwise utility supervision is sparse and top-decile selection is misaligned with technical/raw utility. The next legitimate Flow fix should use a pairwise/listwise utility-aware objective with explicit top-decile pressure, not another MSE-style distribution-matching run.
```
