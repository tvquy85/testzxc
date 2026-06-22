# 17.7 - Supervised Signal Ceiling Probe V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Check whether the V6 train contexts contain enough leak-safe forecast signal for a simple supervised text/technical model to beat `Technical_Rule`. This is a signal-ceiling diagnostic, not an aligned-model claim. If a non-LLM supervised model cannot beat the rule under validation discipline, the next fix should target objective/data/temporal robustness before more LLM alignment compute.

Method basis:
```text
Scikit-learn model selection and cross-validation user guide: https://scikit-learn.org/stable/modules/cross_validation.html
Scikit-learn TimeSeriesSplit principle: time-ordered data should not train on future observations. URL: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
Lopez de Prado financial-ML validation principle: purging/embargoing reduce leakage when labels depend on future events. SSRN: https://ssrn.com/abstract=3257420
```

## Outputs
```text
outputs/tables/17_7_v6_supervised_signal_ceiling_probe.csv
outputs/tables/17_7_v6_supervised_signal_ceiling_grid.csv
outputs/predictions/current_v6_supervised_signal_ceiling_predictions.parquet
outputs/metrics/17_7_v6_supervised_signal_ceiling_probe.json
outputs/status/17_7_SUPERVISED_SIGNAL_CEILING_PROBE_V6.status.json
```

## Method
Train a regularized, class-balanced logistic regression probe on train-only V6 contexts strictly before the validation/test windows. Features include:

```text
TF-IDF over evidence/context text
TF-IDF over technical-event tokens
Technical_Rule score
evidence/news metadata
ticker and V6 track indicators
```

Hyperparameter `C` is selected by validation Macro-F1 with MCC/accuracy tie-breakers. The selected model is evaluated once on held-out current test rows and compared against `Technical_Rule` with paired bootstrap deltas.

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.supervised_signal_ceiling_probe_v6 --train-contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --val-contexts data/processed/current_v6_validation_contexts.parquet --test-contexts data/processed/current_v6_prediction_contexts.parquet --output outputs/tables/17_7_v6_supervised_signal_ceiling_probe.csv --grid-output outputs/tables/17_7_v6_supervised_signal_ceiling_grid.csv --predictions-output outputs/predictions/current_v6_supervised_signal_ceiling_predictions.parquet --metrics outputs/metrics/17_7_v6_supervised_signal_ceiling_probe.json --status outputs/status/17_7_SUPERVISED_SIGNAL_CEILING_PROBE_V6.status.json --manifest outputs/manifests/17_7_SUPERVISED_SIGNAL_CEILING_PROBE_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_supervised_signal_ceiling_probe_v6.py
```

## Acceptance
```text
status JSON gate = PASS
train rows are strictly before validation/test windows
validation-selected C is reported
test paired bootstrap deltas are reported
claim_allowed remains false unless validation and test superiority over Technical_Rule have CI support
```

## Progress Update 2026-06-22
Status: `PASS` for diagnostic execution. Scientific forecast claim remains blocked.

Leak-safe split:
```text
train_rows: 2453
train_min_date: 2018-03-29
train_max_date: 2018-12-17
val_rows: 300
val_min_date: 2022-01-03
test_rows: 300
test_min_date: 2023-01-03
best_c: 0.03
```

Validation result:
```text
Technical_Rule Macro-F1: 0.2041
Technical_Rule MCC: 0.0438
Supervised_LogReg_TFIDF_V6 Macro-F1: 0.1469
Supervised_LogReg_TFIDF_V6 MCC: 0.0283
val_beats_technical_macro_f1: false
val_beats_technical_mcc: false
```

Held-out test result:
```text
Technical_Rule Macro-F1: 0.2277
Technical_Rule MCC: 0.0466
Supervised_LogReg_TFIDF_V6 Macro-F1: 0.1480
Supervised_LogReg_TFIDF_V6 MCC: 0.0776
delta_accuracy: 0.0000, CI95 [-0.0700, 0.0700]
delta_macro_f1: -0.0797, CI95 [-0.1379, -0.0241]
delta_mcc: 0.0310, CI95 [-0.0492, 0.1086]
paired_ci_support: false
signal_ceiling_warning: true
claim_allowed: false
```

Interpretation:
```text
The supervised probe does not reveal an easy forecast ceiling above Technical_Rule under the strict V6 train/validation/test protocol. Macro-F1 is materially worse than the deterministic technical rule, and the point MCC gain lacks CI support. This reinforces that the next forecast step should improve labels/objective/temporal coverage rather than simply fitting a text classifier or stacker on the current small V6 contexts.
```
