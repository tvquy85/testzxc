# 17.8 - Repaired Forecast Baseline Probe V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Measure whether the Step 14.6 repaired DPO predictions change the alignment and forecast blockers. This is a diagnostic comparison table over original DPO, repaired DPO, repaired RWSFT, and Technical_Rule; Step 19 decides which claims can be opened after deterministic downstream reruns.

## Outputs
```text
outputs/tables/17_8_v6_repaired_forecast_baseline_probe.csv
outputs/metrics/17_8_v6_repaired_forecast_baseline_probe.json
outputs/status/17_8_REPAIRED_FORECAST_BASELINE_PROBE_V6.status.json
```

## Command
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.repaired_forecast_baseline_probe_v6 --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-original outputs/predictions/current_v6_dpo_predictions.parquet --dpo-repaired outputs/predictions/current_v6_dpo_predictions_repaired.parquet --rwsft-repaired outputs/predictions/current_v6_rwsft_predictions_repaired.parquet --output outputs/tables/17_8_v6_repaired_forecast_baseline_probe.csv --metrics outputs/metrics/17_8_v6_repaired_forecast_baseline_probe.json --status outputs/status/17_8_REPAIRED_FORECAST_BASELINE_PROBE_V6.status.json --manifest outputs/manifests/17_8_REPAIRED_FORECAST_BASELINE_PROBE_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_repaired_forecast_baseline_probe_v6.py
```

## Acceptance
```text
status JSON gate = PASS
report original DPO, repaired DPO, repaired RWSFT, and Technical_Rule metrics
claim_allowed = false for this probe; Step 19 decides downstream claims
```

## Progress Update 2026-06-22
Status: `PASS`; diagnostic only.

Metrics:
```text
evaluation_rows: 300
DPO original Macro-F1/MCC: 0.1124 / 0.0119
DPO repaired Macro-F1/MCC: 0.1305 / 0.0269
RWSFT repaired Macro-F1/MCC: 0.1202 / -0.0052
Technical_Rule Macro-F1/MCC: 0.2277 / 0.0466

DPO repaired beats RWSFT Macro-F1: true
DPO repaired beats RWSFT MCC: true
DPO repaired beats Technical_Rule Macro-F1: false
DPO repaired beats Technical_Rule MCC: false
```

Interpretation:
```text
Step 14.6 repair is enough to turn the alignment-over-RWSFT point estimate positive, but it is not enough for forecast superiority over Technical_Rule. The deterministic downstream reruns now use repaired predictions as the canonical Step 14 evaluation artifact. Step 19 allows `alignment_improves_over_sft` but keeps forecast superiority blocked.
```
