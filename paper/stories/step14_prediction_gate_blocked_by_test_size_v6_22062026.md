# Step 14 Story - Prediction Gate Blocked Before GPU Inference

Date: 2026-06-22

Step 14 intentionally stopped before adapter inference. The repaired current-data V6 context artifact has only 286 rows in `split == test`, but the Step 14 runbook requires `split == test` only and at least 300 predictions.

Evidence:

```text
context rows: 3129
train rows: 2453
val rows: 390
test rows: 286
required prediction rows: 300
selected_rows: 286
status: FAIL
next_step_allowed: false
```

This is a useful paper/process story because the gate caught a protocol mismatch before expensive inference and before any forecast claim could be made. The model artifacts from Step 13 exist, but prediction evidence cannot be produced under the current Step 14 acceptance rule.

The fix should not be ad hoc. We should not silently add validation rows to test, lower the threshold without explanation, or run prediction anyway and call it complete. The next decision should be documented as one of:

```text
1. Revise the split protocol leak-safely so the test holdout has at least 300 rows.
2. Keep the existing 286-row current-data test holdout and explicitly revise Step 14's threshold.
3. Treat this as a negative protocol finding and stop downstream forecast/backtest/counterfactual claims.
```

Potential wording:

```text
Although both V6 adapters were trained, the prediction gate exposed a holdout-size mismatch: the fixed current-data test split contains 286 examples, below the predeclared 300-prediction minimum. We therefore stopped before inference and did not report adapter forecast metrics. This preserves the integrity of the claim gate and forces the split-size decision to be explicit.
```

## Recovery note

The blocker was resolved without lowering the threshold. The existing current-data fallback context pool (`data/processed/ticker_date_contexts_h1_v2_targets.parquet`) contains 5,187 test rows across 248 trading days. A prediction-only V6 context artifact was built from that source:

```text
data/processed/current_v6_prediction_contexts.parquet
rows: 300
split: test only
trading days: 173
label distribution: 60 per class
v6_track_distribution: hard_event_news=201, company_general_news=99
```

The key methodological point is that test size was repaired by changing the prediction-context selection protocol, not by mixing validation rows into test and not by lowering the gate. This is consistent with temporal evaluation discipline: future/test periods stay separated from train/validation, and the expanded prediction context is used only for evaluation.

Final Step 14 metrics:

```text
DPO schema_ok_rate: 0.9167
DPO Macro-F1: 0.1124
DPO MCC: 0.0119
RWSFT schema_ok_rate: 1.0000
RWSFT Macro-F1: 0.1202
RWSFT MCC: -0.0052
```

This is a clean negative performance signal. Step 14 now passes the artifact/schema gate, but the forecast-quality claim is still blocked. The paper story should say that V6 alignment produced runnable adapters and valid predictions, yet prediction quality remains weak and must be judged against Technical_Rule/backtest gates before any forecast claim.

Updated wording:

```text
After repairing the prediction-context selection to meet the predeclared 300-row test gate, both V6 adapters produced valid deterministic predictions. However, Macro-F1 and MCC remained weak, so adapter training is not sufficient evidence of forecast improvement. The claim gate remains closed until baseline and trading evaluations pass.
```
