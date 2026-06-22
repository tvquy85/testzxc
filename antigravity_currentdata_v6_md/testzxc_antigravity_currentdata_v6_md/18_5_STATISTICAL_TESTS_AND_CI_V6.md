# 18.5 - Statistical Tests And Confidence Intervals V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Add paper-facing uncertainty evidence before the strict claim gate. This step does not make a positive alpha or forecast claim by itself; it decides whether the existing point estimates have confidence-interval support.

## Outputs
```text
outputs/tables/19_v6_statistical_tests.csv
outputs/tables/19_v6_backtest_daily_comparison.csv
outputs/metrics/19_v6_statistical_tests.json
outputs/status/18_5_STATISTICAL_TESTS_AND_CI_V6.status.json
```

## Method
- Moving-block bootstrap CI for DPO daily net return Sharpe and mean return.
- Paired moving-block bootstrap CI for DPO versus Technical_Rule daily-return deltas.
- Paired row bootstrap CI for DPO versus Technical_Rule and DPO versus RWSFT forecast metrics.
- Exact one-sided McNemar test for paired directional accuracy discordances.

The design follows the finance/econometrics caution that Sharpe estimates have sampling error and serial correlation risk, and that model-selection/data-snooping can make naive p-values misleading. Therefore Step 19 may allow the `statistical_tests` claim when the evidence exists, while keeping alpha/forecast claims blocked unless their CIs support the claim.

## Command
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.repro.run_v6_statistical_tests --daily-returns outputs/tables/15_v6_daily_returns.csv --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-predictions outputs/predictions/current_v6_dpo_predictions.parquet --rwsft-predictions outputs/predictions/current_v6_rwsft_predictions.parquet --backtest-metrics outputs/metrics/15_v6_backtest_track_baseline.json --output outputs/tables/19_v6_statistical_tests.csv --daily-comparison-output outputs/tables/19_v6_backtest_daily_comparison.csv --metrics outputs/metrics/19_v6_statistical_tests.json --status outputs/status/18_5_STATISTICAL_TESTS_AND_CI_V6.status.json --manifest outputs/manifests/18_5_STATISTICAL_TESTS_AND_CI_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_statistical_tests_v6.py tests/test_strong_accept_gate_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/18_5_STATISTICAL_TESTS_AND_CI_V6.status.json
```

## Acceptance
```text
required_tests_present = true
confidence_interval_available = true
daily_return_rows >= 120
forecast_rows >= 250
status = PASS
```

## Claim Boundary
If CIs include zero or paired tests do not support positive deltas, the correct result is still `PASS` for evidence generation but `CLAIM_RESTRICTED` at Step 19.

## Progress Update 2026-06-22
Status: `PASS`; statistical evidence was generated on the current V6 artifacts and verified by status JSON.

Artifacts verified:
```text
outputs/tables/19_v6_statistical_tests.csv
outputs/tables/19_v6_backtest_daily_comparison.csv
outputs/metrics/19_v6_statistical_tests.json
outputs/manifests/18_5_STATISTICAL_TESTS_AND_CI_V6.manifest.json
outputs/status/18_5_STATISTICAL_TESTS_AND_CI_V6.status.json
```

Command run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.repro.run_v6_statistical_tests --daily-returns outputs/tables/15_v6_daily_returns.csv --contexts data/processed/current_v6_prediction_contexts.parquet --dpo-predictions outputs/predictions/current_v6_dpo_predictions.parquet --rwsft-predictions outputs/predictions/current_v6_rwsft_predictions.parquet --backtest-metrics outputs/metrics/15_v6_backtest_track_baseline.json --output outputs/tables/19_v6_statistical_tests.csv --daily-comparison-output outputs/tables/19_v6_backtest_daily_comparison.csv --metrics outputs/metrics/19_v6_statistical_tests.json --status outputs/status/18_5_STATISTICAL_TESTS_AND_CI_V6.status.json --manifest outputs/manifests/18_5_STATISTICAL_TESTS_AND_CI_V6.manifest.json
```

Key metrics:
```text
test_count: 12
required_tests_present: true
confidence_interval_available: true
daily_return_rows: 173
forecast_rows: 275
dpo_sharpe: 0.9667
dpo_sharpe_ci95: [-1.5791, 3.3879]
dpo_mean_daily_return_ci95: [-0.00203, 0.00457]
delta_sharpe_vs_technical_rule_ci95: [-0.3943, 5.1388]
delta_mean_return_vs_technical_rule_ci95: [-0.00023, 0.00745]
alpha_paper_level_supported: false
forecast_dpo_beats_technical_ci_support: false
forecast_dpo_beats_rwsft_ci_support: false
```

Acceptance result:
```text
status JSON gate: PASS
required tests present: PASS
confidence intervals available: PASS
paper-level alpha support: FAIL, correctly blocked by CI
forecast superiority support: FAIL, correctly blocked by paired tests
```
