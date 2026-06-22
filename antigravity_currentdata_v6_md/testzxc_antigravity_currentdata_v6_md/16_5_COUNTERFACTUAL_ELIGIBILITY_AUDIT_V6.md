# 16.5 - Counterfactual Eligibility Audit V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Diagnose whether Step 16 counterfactual failures reflect model insensitivity, invalid directional tests, or original prediction bias. A directional expectation test is only interpretable when the original prediction has enough probability mass on the side expected to decrease.

## Outputs
```text
outputs/tables/16_5_v6_counterfactual_eligibility_by_type.csv
outputs/tables/16_5_v6_counterfactual_task_eligibility.csv
outputs/metrics/16_5_v6_counterfactual_eligibility_audit.json
outputs/status/16_5_COUNTERFACTUAL_ELIGIBILITY_AUDIT_V6.status.json
```

## Method
- For `up_decreases`, require original DPO `up_side = p_mild_up + p_strong_up >= 0.25`.
- For `down_decreases`, require original DPO `down_side = p_mild_down + p_strong_down >= 0.20`.
- Report eligibility by counterfactual type and compare against Step 16 pass/no-change rates.

This is grounded in directional expectation testing and contrast-set evaluation: a perturbation should test a local decision boundary where the model has a relevant original-side signal. If the model assigns zero down-side probability, a `down_decreases` perturbation is not strong evidence of news faithfulness; it is mostly a bias/saturation diagnostic.

## Command
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.audit_counterfactual_eligibility_v6 --tasks data/eval/current_v6_counterfactual_tasks.jsonl --predictions outputs/predictions/current_v6_dpo_predictions.parquet --breakdown outputs/tables/16_v6_counterfactual_breakdown.csv --output outputs/tables/16_5_v6_counterfactual_eligibility_by_type.csv --task-output outputs/tables/16_5_v6_counterfactual_task_eligibility.csv --metrics outputs/metrics/16_5_v6_counterfactual_eligibility_audit.json --status outputs/status/16_5_COUNTERFACTUAL_ELIGIBILITY_AUDIT_V6.status.json --manifest outputs/manifests/16_5_COUNTERFACTUAL_ELIGIBILITY_AUDIT_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_counterfactual_eligibility_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/16_5_COUNTERFACTUAL_ELIGIBILITY_AUDIT_V6.status.json
```

## Acceptance
```text
status = PASS
tasks >= 300
prediction_rows >= 300
schema_ok_rate_on_tasks >= 0.90
eligible_side_signal_rate is reported
```

## Progress Update 2026-06-22
Status: `PASS`; audit confirms a root cause rather than allowing a counterfactual claim.

Key metrics:
```text
tasks: 350
prediction_rows: 300
schema_ok_rate_on_tasks: 0.9286
eligible_side_signal_rate: 0.4229
up_expected_eligible_rate: 0.7179
down_expected_eligible_rate: 0.0516
down_expected_mean_down_side: 0.0239
down_expected_zero_down_side_rate: 0.8774
claim_allowed: false
```

Interpretation:
```text
Step 16 failure is partly confounded by DPO prediction-side saturation. Most down-decrease counterfactuals have almost no original down-side probability mass, so removing/neutralizing negative evidence cannot reliably show a down-side decrease. The next model fix should target prediction calibration/class balance and counterfactual task eligibility, not merely rerun the same perturbation set.
```
