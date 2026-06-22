# Step 17.7 Story - Supervised Signal Ceiling Probe V6

Date: 2026-06-22

## What We Tested
After validation gating and stacking failed to produce CI-supported forecast gains, we tested whether the current V6 train contexts contain enough leak-safe signal for a simple supervised model to beat `Technical_Rule`.

The probe uses train-only rows before the validation/test windows, selects logistic regression regularization on validation, and evaluates once on the 2023 current-data test split. This follows standard model-selection discipline and time-series leakage caution:

- Scikit-learn model selection and cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- Scikit-learn TimeSeriesSplit principle: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Lopez de Prado financial-ML validation/purging principle: https://ssrn.com/abstract=3257420

## Result
The selected model used `C=0.03` and trained on 2,453 train rows from 2018-03-29 to 2018-12-17, before the validation window starting 2022-01-03 and test window starting 2023-01-03.

Validation:

```text
Technical_Rule Macro-F1: 0.2041
Technical_Rule MCC: 0.0438
Supervised_LogReg_TFIDF_V6 Macro-F1: 0.1469
Supervised_LogReg_TFIDF_V6 MCC: 0.0283
```

Held-out test:

```text
Technical_Rule Macro-F1: 0.2277
Technical_Rule MCC: 0.0466
Supervised_LogReg_TFIDF_V6 Macro-F1: 0.1480
Supervised_LogReg_TFIDF_V6 MCC: 0.0776
delta_macro_f1_vs_technical: -0.0797, CI95 [-0.1379, -0.0241]
delta_mcc_vs_technical: 0.0310, CI95 [-0.0492, 0.1086]
paired_ci_support: false
claim_allowed: false
```

## Paper Value
This result is a useful ceiling check. If a supervised text/technical classifier trained on leak-safe V6 contexts cannot beat the deterministic technical rule on validation or Macro-F1 test, the failure is not merely that the LLM was under-aligned. The current feature/label protocol itself has weak or unstable directional signal.

## Reusable Lesson
Before spending GPU on additional DPO/RLHF/Flow variants, run a cheap supervised ceiling probe. If it fails under validation discipline, the next paper-worthy improvement should target data construction, label definition, horizon selection, and temporal robustness rather than larger alignment training.
