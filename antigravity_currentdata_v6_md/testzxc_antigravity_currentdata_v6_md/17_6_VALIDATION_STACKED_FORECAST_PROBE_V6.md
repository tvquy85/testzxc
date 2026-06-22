# 17.6 - Validation-Stacked Forecast Probe V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Test whether a validation-trained meta-classifier can combine DPO probabilities, DPO labels, Technical_Rule labels, evidence metadata, and track indicators into a stronger held-out forecast. This follows the stacked-generalization idea, but the final decision remains strict: validation gains do not support a claim unless the held-out test split beats Technical_Rule with paired CI support.

References used for method boundary:
```text
Wolpert, D.H. 1992. Stacked Generalization. Neural Networks 5(2):241-259. DOI: 10.1016/S0893-6080(05)80023-1
Scikit-learn StackingClassifier documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
Scikit-learn LogisticRegression documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
```

## Outputs
```text
outputs/tables/17_6_v6_validation_stacked_forecast_probe.csv
outputs/tables/17_6_v6_validation_stacked_grid.csv
outputs/predictions/current_v6_validation_stacked_forecast_predictions.parquet
outputs/metrics/17_6_v6_validation_stacked_forecast_probe.json
outputs/status/17_6_VALIDATION_STACKED_FORECAST_PROBE_V6.status.json
```

## Method
Train regularized multinomial `LogisticRegression` on validation rows only. The feature set includes:

```text
DPO probability distribution
DPO side confidence and up/down margin
DPO predicted-label indicators
Technical_Rule score and predicted-label indicators
news/evidence count and quality metadata
V6 track indicators
```

Select `(C, class_weight)` by validation Macro-F1, with MCC and accuracy tie-breakers. Apply the selected model once to the held-out test split and compare against Technical_Rule using paired bootstrap deltas for accuracy, Macro-F1, and MCC.

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.validation_stacked_forecast_probe_v6 --val-contexts data/processed/current_v6_validation_contexts.parquet --val-predictions outputs/predictions/current_v6_dpo_val_predictions.parquet --test-contexts data/processed/current_v6_prediction_contexts.parquet --test-predictions outputs/predictions/current_v6_dpo_predictions.parquet --output outputs/tables/17_6_v6_validation_stacked_forecast_probe.csv --grid-output outputs/tables/17_6_v6_validation_stacked_grid.csv --predictions-output outputs/predictions/current_v6_validation_stacked_forecast_predictions.parquet --metrics outputs/metrics/17_6_v6_validation_stacked_forecast_probe.json --status outputs/status/17_6_VALIDATION_STACKED_FORECAST_PROBE_V6.status.json --manifest outputs/manifests/17_6_VALIDATION_STACKED_FORECAST_PROBE_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_validation_stacked_forecast_probe_v6.py
```

## Acceptance
```text
status JSON gate = PASS
validation/test split separation is preserved
grid table reports validation and test metrics
paired bootstrap CI is reported
claim_allowed remains false unless test Macro-F1 and MCC beat Technical_Rule with CI support
```

## Progress Update 2026-06-22
Status: `PASS` for the diagnostic pipeline. Scientific claim remains blocked.

Best validation-selected stacker:
```text
best_c: 10.0
best_class_weight: none
feature_count: 33
val_rows: 300
test_rows: 300
```

Validation vs test result:
```text
Validation Stacked_Logistic_V6 Macro-F1: 0.3913
Validation Stacked_Logistic_V6 MCC: 0.2531
Test Stacked_Logistic_V6 Macro-F1: 0.2017
Test Stacked_Logistic_V6 MCC: 0.0216
Test Technical_Rule Macro-F1: 0.2277
Test Technical_Rule MCC: 0.0466
```

Paired test delta vs Technical_Rule:
```text
delta_accuracy: -0.0200, CI95 [-0.0800, 0.0400]
delta_macro_f1: -0.0260, CI95 [-0.0822, 0.0293]
delta_mcc: -0.0250, CI95 [-0.0976, 0.0462]
paired_ci_support: false
validation_overfit_warning: true
claim_allowed: false
```

Interpretation:
```text
The validation-trained stacker overfits the small validation split and does not generalize to the held-out test split. This rules out a simple meta-classifier fix for the forecast blocker. The next forecast improvement should target objective/data signal quality, temporal robustness, and calibration under larger validation coverage rather than adding a more flexible stacker on the current 300-row validation set.
```
