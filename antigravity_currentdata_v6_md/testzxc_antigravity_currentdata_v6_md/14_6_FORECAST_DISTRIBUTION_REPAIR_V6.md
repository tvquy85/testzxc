# 14.6 - Forecast Distribution Repair V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Repair parse-ok forecast JSON outputs whose only schema failure is probability mass not summing to 1.0. This is a deterministic probability-distribution repair: if all five forecast probabilities are present, finite, non-negative, and have positive total mass, normalize them and rederive label/action. The repair does not use target labels.

Method basis:
```text
Probabilistic forecasts should be evaluated as distributions.
Post-processing calibration/normalization is allowed only when it is label-free and fully auditable.
```

## Outputs
```text
outputs/predictions/current_v6_dpo_predictions_repaired.parquet
outputs/predictions/current_v6_rwsft_predictions_repaired.parquet
outputs/metrics/14_6_v6_dpo_forecast_distribution_repair.json
outputs/metrics/14_6_v6_rwsft_forecast_distribution_repair.json
outputs/status/14_6_DPO_FORECAST_DISTRIBUTION_REPAIR_V6.status.json
outputs/status/14_6_RWSFT_FORECAST_DISTRIBUTION_REPAIR_V6.status.json
```

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.repair_forecast_predictions_v6 --predictions outputs/predictions/current_v6_dpo_predictions.parquet --contexts data/processed/current_v6_prediction_contexts.parquet --output outputs/predictions/current_v6_dpo_predictions_repaired.parquet --metrics outputs/metrics/14_6_v6_dpo_forecast_distribution_repair.json --status outputs/status/14_6_DPO_FORECAST_DISTRIBUTION_REPAIR_V6.status.json --manifest outputs/manifests/14_6_DPO_FORECAST_DISTRIBUTION_REPAIR_V6.manifest.json --min-schema-ok-rate 0.99
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.repair_forecast_predictions_v6 --predictions outputs/predictions/current_v6_rwsft_predictions.parquet --contexts data/processed/current_v6_prediction_contexts.parquet --output outputs/predictions/current_v6_rwsft_predictions_repaired.parquet --metrics outputs/metrics/14_6_v6_rwsft_forecast_distribution_repair.json --status outputs/status/14_6_RWSFT_FORECAST_DISTRIBUTION_REPAIR_V6.status.json --manifest outputs/manifests/14_6_RWSFT_FORECAST_DISTRIBUTION_REPAIR_V6.manifest.json --min-schema-ok-rate 0.99
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_forecast_prediction_repair_v6.py
```

## Acceptance
```text
status JSON gate = PASS
repaired_schema_ok_rate >= 0.99
repair does not use target labels
claim_allowed = false for the repair step itself; downstream claim status is decided by Step 19 after reruns
```

## Progress Update 2026-06-22
Status: `PASS` for DPO and RWSFT repair artifacts.

DPO repair:
```text
rows: 300
original_schema_ok_rate: 0.9167
repairable_invalid_rows: 25
repaired_rows: 25
unrepaired_invalid_rows: 0
repaired_schema_ok_rate: 1.0000
repair_reason_counts: normalized_probability_sum=25

original Macro-F1/MCC: 0.1124 / 0.0119
repaired Macro-F1/MCC: 0.1305 / 0.0269
delta Macro-F1/MCC: +0.0181 / +0.0150
```

RWSFT repair:
```text
rows: 300
original_schema_ok_rate: 1.0000
repaired_rows: 0
repaired_schema_ok_rate: 1.0000
Macro-F1/MCC unchanged: 0.1202 / -0.0052
```

Interpretation:
```text
The original DPO artifact had 25 parse-ok distributions that failed only because probability mass did not sum to 1.0. Label-free normalization recovers them and improves DPO metrics enough to beat RWSFT on Macro-F1 and MCC in a follow-up probe, but it still does not beat Technical_Rule.

Promotion update:
```text
The repaired DPO/RWSFT prediction artifacts were promoted as canonical Step 14 evaluation inputs for deterministic reruns of Steps 15, 16.5, 17, 17.5, 17.6, 17.8, 18, 18.5, 19, and 20. Step 19 allows alignment_improves_over_sft on point Macro-F1 and MCC, but forecast superiority and paper-level alpha remain blocked.
```
```
