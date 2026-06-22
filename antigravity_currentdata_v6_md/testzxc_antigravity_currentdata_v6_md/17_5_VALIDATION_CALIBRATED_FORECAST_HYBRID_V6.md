# 17.5 - Validation-Calibrated Forecast Hybrid V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Test whether the DPO forecast distribution contains usable signal after leak-safe validation calibration. The method chooses a confidence threshold on validation data only, then applies it once to held-out test data:

```text
if DPO schema is valid and DPO side-confidence >= threshold:
    use DPO prediction
else:
    use Technical_Rule prediction
```

This is a diagnostic decision-system probe, not a replacement for the official DPO baseline claim.

## Outputs
```text
data/processed/current_v6_validation_contexts.parquet
outputs/predictions/current_v6_dpo_val_predictions.parquet
outputs/tables/17_5_v6_validation_calibrated_hybrid.csv
outputs/tables/17_5_v6_validation_threshold_search.csv
outputs/predictions/current_v6_validation_calibrated_hybrid_predictions.parquet
outputs/metrics/17_5_v6_validation_calibrated_hybrid.json
outputs/status/17_5_VALIDATION_CALIBRATED_FORECAST_HYBRID_V6.status.json
```

## Method
The threshold is tuned on validation Macro-F1 with MCC/accuracy tie-breakers. Test performance is then reported against Technical_Rule with paired bootstrap CIs for Accuracy, Macro-F1, and MCC deltas.

This follows the validation-set discipline used in post-hoc calibration/threshold tuning: tuning must happen outside the final test split, and a positive point estimate is not enough for a paper claim without uncertainty support.

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.build_prediction_contexts_v6 --source data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --fallback-contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet --split val --min-rows 300 --min-trading-days 20 --output data/processed/current_v6_validation_contexts.parquet --metrics outputs/metrics/14_5_v6_validation_contexts.json --samples review_samples/currentdata_v6/14_5_validation_context_samples.jsonl --status outputs/status/14_5_BUILD_V6_VALIDATION_CONTEXTS.status.json --manifest outputs/manifests/14_5_BUILD_V6_VALIDATION_CONTEXTS.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.generate_test_predictions_v2 --contexts data/processed/current_v6_validation_contexts.parquet --adapter outputs/models/qwen3_current_v6_dpo_adapter --config configs/local_paths.yaml --hf-home E:/huggingface --split val --allow-non-test-split --min-rows 300 --min-schema-ok-rate 0.90 --min-parse-ok-rate 0.90 --output outputs/predictions/current_v6_dpo_val_predictions.parquet --metrics outputs/metrics/14_5_v6_dpo_val_predictions.json --samples review_samples/currentdata_v6/14_5_dpo_val_prediction_samples.jsonl --status outputs/status/14_5_PREDICT_DPO_VAL_V6.status.json --manifest outputs/manifests/14_5_PREDICT_DPO_VAL_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.validation_calibrated_hybrid_v6 --val-contexts data/processed/current_v6_validation_contexts.parquet --val-predictions outputs/predictions/current_v6_dpo_val_predictions.parquet --test-contexts data/processed/current_v6_prediction_contexts.parquet --test-predictions outputs/predictions/current_v6_dpo_predictions.parquet --output outputs/tables/17_5_v6_validation_calibrated_hybrid.csv --threshold-table outputs/tables/17_5_v6_validation_threshold_search.csv --predictions-output outputs/predictions/current_v6_validation_calibrated_hybrid_predictions.parquet --metrics outputs/metrics/17_5_v6_validation_calibrated_hybrid.json --status outputs/status/17_5_VALIDATION_CALIBRATED_FORECAST_HYBRID_V6.status.json --manifest outputs/manifests/17_5_VALIDATION_CALIBRATED_FORECAST_HYBRID_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_prediction_min_rows_v6.py tests/test_validation_calibrated_hybrid_v6.py
```

## Acceptance
```text
validation contexts status = PASS
validation DPO prediction status = PASS
hybrid calibration status = PASS
threshold selected only on val
paired CI support is reported
```

## Progress Update 2026-06-22
Status: `PASS`; the probe found a small validation-calibrated improvement over Technical_Rule point estimates, but no CI-supported claim.

Validation prediction gate:
```text
rows: 300
split: val
schema_ok_rate: 0.98
parse_ok_rate: 0.9967
selected_trading_days: 24
```

Hybrid metrics:
```text
best_threshold: 0.71
test_dpo_use_rate: 0.2167
test_hybrid_macro_f1: 0.2321
test_hybrid_mcc: 0.0503
test_technical_macro_f1: 0.2277
test_technical_mcc: 0.0466
delta_macro_f1_vs_technical: 0.00436
delta_mcc_vs_technical: 0.00371
delta_macro_f1_ci95: [-0.02464, 0.03220]
delta_mcc_ci95: [-0.03249, 0.03761]
paired_ci_support: false
claim_allowed: false
```

Interpretation:
```text
DPO contains some usable high-confidence signal when selected by validation thresholding, but the test improvement is too small and uncertain for a forecast claim. This is a promising next-fix direction, not a strong-accept result.
```
