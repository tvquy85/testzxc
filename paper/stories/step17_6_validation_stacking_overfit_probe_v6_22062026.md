# Step 17.6 Story - Validation Stacking Overfit Probe V6

Date: 2026-06-22

## What We Tested
After Step 17.5 showed a tiny validation-calibrated hybrid improvement with confidence intervals crossing zero, we tested a more flexible stacked forecast probe. The stacker used DPO probabilities, DPO labels, Technical_Rule labels, evidence metadata, and V6 track indicators. It trained only on validation rows and was evaluated once on the held-out test split.

The design follows stacked generalization as a diagnostic ensemble idea, with regularized logistic regression as the final classifier:

- Wolpert, D.H. 1992. Stacked Generalization. Neural Networks 5(2):241-259. DOI: 10.1016/S0893-6080(05)80023-1
- Scikit-learn StackingClassifier documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
- Scikit-learn LogisticRegression documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

## Result
The best validation-selected model used `C=10.0`, `class_weight=none`, and 33 features.

```text
Validation Stacked_Logistic_V6 Macro-F1: 0.3913
Validation Stacked_Logistic_V6 MCC: 0.2531
Test Stacked_Logistic_V6 Macro-F1: 0.2017
Test Stacked_Logistic_V6 MCC: 0.0216
Test Technical_Rule Macro-F1: 0.2277
Test Technical_Rule MCC: 0.0466
```

Paired bootstrap deltas versus Technical_Rule were negative on point estimates and all confidence intervals crossed zero:

```text
delta_accuracy: -0.0200, CI95 [-0.0800, 0.0400]
delta_macro_f1: -0.0260, CI95 [-0.0822, 0.0293]
delta_mcc: -0.0250, CI95 [-0.0976, 0.0462]
paired_ci_support: false
validation_overfit_warning: true
claim_allowed: false
```

## Paper Value
This is useful negative evidence. The current forecast blocker is not solved by simply adding a meta-classifier on top of weak DPO and technical signals. The validation split is small enough that a flexible stacker can look strong in-sample while generalizing worse than Technical_Rule.

For the paper narrative, this supports a stricter claim boundary: V6 provides a reproducible diagnostic pipeline and honest failure localization, not forecast superiority. The next forecast work should improve objective/data signal and temporal robustness before increasing ensemble flexibility.

## Reusable Lesson
When a validation-calibrated method gives only tiny test gains, test a stronger but controlled meta-model as a falsification probe. If the meta-model overfits and fails the held-out test, do not promote the original smaller gain. Treat it as evidence that the signal is fragile.
