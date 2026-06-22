# 15.5 - Trading Policy Variant Probe V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Evaluate whether existing V6 prediction variants produce after-cost trading utility beyond `Technical_Rule` on the same test calendar. This is a diagnostic policy-sensitivity probe, not a paper-level alpha claim, because multiple variants are compared after observing the held-out period.

Method basis:
```text
Lo 2002 Sharpe-ratio inference: Sharpe estimates have sampling error.
White 2000 Reality Check: repeated rule/model searches on the same data create data-snooping risk.
Bailey and Lopez de Prado Deflated Sharpe Ratio: selection bias, non-normality, and backtest overfitting can inflate performance.
```

## Outputs
```text
outputs/tables/15_5_v6_trading_policy_variant_summary.csv
outputs/tables/15_5_v6_trading_policy_variant_daily_returns.csv
outputs/metrics/15_5_v6_trading_policy_variant_probe.json
outputs/status/15_5_TRADING_POLICY_VARIANT_PROBE_V6.status.json
```

## Method
Build daily after-cost returns for:

```text
Technical_Rule
Qwen_DPO_V6_Official_Action
Qwen_DPO_V6_Label
Validation_Calibrated_Hybrid
Validation_Stacked_Logistic
Supervised_LogReg_TFIDF
```

All variants use the same 173-day test calendar, transaction cost, slippage, short borrow cost, and position cap. The probe reports moving-block bootstrap CIs for Sharpe and mean return, plus paired deltas versus `Technical_Rule`.

## Commands
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.trading_policy_variant_probe_v6 --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-predictions outputs/predictions/current_v6_dpo_predictions.parquet --hybrid-predictions outputs/predictions/current_v6_validation_calibrated_hybrid_predictions.parquet --stacked-predictions outputs/predictions/current_v6_validation_stacked_forecast_predictions.parquet --supervised-predictions outputs/predictions/current_v6_supervised_signal_ceiling_predictions.parquet --summary-output outputs/tables/15_5_v6_trading_policy_variant_summary.csv --daily-output outputs/tables/15_5_v6_trading_policy_variant_daily_returns.csv --metrics outputs/metrics/15_5_v6_trading_policy_variant_probe.json --status outputs/status/15_5_TRADING_POLICY_VARIANT_PROBE_V6.status.json --manifest outputs/manifests/15_5_TRADING_POLICY_VARIANT_PROBE_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_trading_policy_variant_probe_v6.py
```

## Acceptance
```text
status JSON gate = PASS
all variants use same test calendar and cost assumptions
moving-block bootstrap CIs are reported
multiple_testing_warning = true
claim_allowed = false
```

## Progress Update 2026-06-22
Status: `PASS`; diagnostic only.

Summary:
```text
strategy_count: 6
num_trading_days: 173
best_strategy_by_sharpe: Supervised_LogReg_TFIDF
best_strategy_sharpe: 1.4379
best_strategy_mean_daily_return: 0.001811
best_strategy_ci_support_vs_zero: false
best_strategy_ci_support_vs_technical: true
dpo_official_sharpe: 0.9667
repaired_dpo_official_sharpe: 1.2002
multiple_testing_warning: true
claim_allowed: false
```

Key rows:
```text
Supervised_LogReg_TFIDF Sharpe: 1.4379, CI95 [-0.8008, 3.4505]
Supervised_LogReg_TFIDF mean daily return: 0.001811, CI95 [-0.001012, 0.004623]
Supervised_LogReg_TFIDF delta Sharpe vs Technical: 2.8590, CI95 [0.1021, 5.4438]
Supervised_LogReg_TFIDF delta mean return vs Technical: 0.004136, CI95 [0.000386, 0.007864]

Qwen_DPO_V6_Official_Action Sharpe after repaired-DPO promotion: 1.2002, CI95 [-1.4061, 3.6628]
Qwen_DPO_V6_Official_Action delta Sharpe vs Technical: 2.6213, CI95 [0.0281, 5.2204]

Technical_Rule Sharpe: -1.4211, CI95 [-3.5464, 0.5454]
```

Interpretation:
```text
Sparse prediction policies can produce better trading returns than Technical_Rule despite weak classification Macro-F1, but this does not establish paper-level alpha. The best observed strategy is a non-LLM supervised diagnostic variant selected after seeing the same held-out test period; its absolute Sharpe and mean-return CIs still cross zero. A paper-level alpha claim needs preregistered selection on validation and fresh out-of-sample confirmation.
```
