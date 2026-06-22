# 15.6 - Validation-Selected Trading Policy V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Convert the Step 15.5 post-hoc trading-policy signal into a stricter validation-selected protocol: search DPO probability thresholds and daily position caps only on validation, lock the selected policy, and evaluate it once on the held-out test calendar with moving-block bootstrap confidence intervals.

Method basis:
```text
White 2000 Reality Check: repeated rule search on the same test period creates data-snooping risk.
Bailey and Lopez de Prado Deflated Sharpe Ratio: selection bias and backtest overfitting inflate Sharpe evidence.
Politis and Romano 1994 stationary bootstrap: dependent daily returns require dependence-aware resampling.
```

## Outputs
```text
outputs/tables/15_6_v6_validation_selected_trading_grid.csv
outputs/tables/15_6_v6_validation_selected_trading_summary.csv
outputs/tables/15_6_v6_validation_selected_trading_daily_returns.csv
outputs/metrics/15_6_v6_validation_selected_trading_policy.json
outputs/status/15_6_VALIDATION_SELECTED_TRADING_POLICY_V6.status.json
```

## Method
Grid-search DPO action side-confidence thresholds from `0.00` to `0.90` and daily position caps `{1,2,3,5,10,20}` on the 24-day validation calendar. Require at least 5 nonzero validation position days. Select by validation Sharpe, then validation mean return, validation active days, smaller cap, and lower threshold. The selected policy is then locked and evaluated on the 173-day test calendar against:

```text
Technical_Rule
Qwen_DPO_V6 default threshold 0.20, cap 20
```

The alpha claim remains blocked unless the selected test policy has >=120 test days and moving-block bootstrap CIs support positive absolute Sharpe, positive absolute mean daily return, and positive paired deltas versus `Technical_Rule`.

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.validation_selected_trading_policy_v6 --val-contexts data/processed/current_v6_validation_contexts.parquet --val-predictions outputs/predictions/current_v6_dpo_val_predictions_repaired.parquet --test-contexts data/processed/current_v6_prediction_contexts.parquet --test-predictions outputs/predictions/current_v6_dpo_predictions_repaired.parquet --thresholds 0.00:0.90:0.01 --position-caps 1,2,3,5,10,20 --min-val-nonzero-days 5 --grid-output outputs/tables/15_6_v6_validation_selected_trading_grid.csv --summary-output outputs/tables/15_6_v6_validation_selected_trading_summary.csv --daily-output outputs/tables/15_6_v6_validation_selected_trading_daily_returns.csv --metrics outputs/metrics/15_6_v6_validation_selected_trading_policy.json --status outputs/status/15_6_VALIDATION_SELECTED_TRADING_POLICY_V6.status.json --manifest outputs/manifests/15_6_VALIDATION_SELECTED_TRADING_POLICY_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_validation_selected_trading_policy_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/15_6_VALIDATION_SELECTED_TRADING_POLICY_V6.status.json
```

## Acceptance
```text
status JSON gate = PASS
selection_uses_test_returns = false
grid and selected policy artifacts are written
moving-block bootstrap CIs are reported on test
claim_allowed = true only if absolute and paired CI gates pass
```

## Progress Update 2026-06-22
Status: `PASS`; diagnostic-positive but paper-level alpha remains blocked.

Selected validation policy:
```text
threshold: 0.61
position_cap: 3
validation days: 24
validation nonzero days: 24
validation Sharpe: 5.4164
validation mean daily return: 0.004961
grid policies evaluated: 546
```

Locked test result:
```text
test days: 173
test nonzero position days: 54
test coverage: 0.3121
test Sharpe: 1.6840
test Sharpe CI95: [-0.9523, 4.0931]
test mean daily return: 0.001813
test mean daily return CI95: [-0.001080, 0.004889]
delta Sharpe vs Technical_Rule: 3.1051, CI95 [0.7142, 5.4266]
delta mean return vs Technical_Rule: 0.004138, CI95 [0.000959, 0.007316]
delta Sharpe vs default DPO: 0.4839
delta mean return vs default DPO: 0.000290
alpha_paper_level_supported: false
```

Interpretation:
```text
Validation selection reduces the Step 15.5 post-hoc risk and produces a stronger DPO trading diagnostic than the default policy. However, absolute Sharpe and mean-return confidence intervals still cross zero. This is useful evidence for the next full-scale/fresh-horizon trading protocol, not a paper-level alpha claim.
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m py_compile src\eval\validation_selected_trading_policy_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests\test_validation_selected_trading_policy_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/15_6_VALIDATION_SELECTED_TRADING_POLICY_V6.status.json
```
