# 15 — Daily Backtest V6 by Track and Baseline

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Evaluate decision utility without inflating Sharpe. Use daily portfolio returns and compare to Technical_Rule and technical-only/no-news baselines.

## Outputs
```text
outputs/metrics/15_v6_backtest_track_baseline.json
outputs/tables/15_v6_daily_returns.csv
outputs/tables/15_v6_track_breakdown.csv
outputs/status/15_BACKTEST_TRACK_BASELINE_V6.status.json
```

## Backtest rules
Aggregate date × ticker; use after-cost returns; include transaction cost, slippage, borrow cost; report by track; alpha claim blocked unless Sharpe >0 and days >=60 diagnostic.

## Test case
```python
from src.eval.backtest_daily_portfolio_v3 import compute_sharpe

def test_sharpe_zero_returns():
    assert compute_sharpe([0,0,0]) == 0
```

## Commands
```bash
python -m src.eval.backtest_daily_portfolio_v3 --predictions outputs/predictions/current_v6_dpo_predictions.parquet --output-daily outputs/tables/15_v6_daily_returns.csv --output-track outputs/tables/15_v6_track_breakdown.csv --metrics outputs/metrics/15_v6_backtest_track_baseline.json --status outputs/status/15_BACKTEST_TRACK_BASELINE_V6.status.json --cost-bps 5 --slippage-bps 2 --short-borrow-bps 1
python -m pytest -q tests/test_backtest_v6.py tests
```

## Progress Update 2026-06-22
Status: `PASS`; status file reports `next_step_allowed=true`.

Important method correction: an initial Step 15 run counted only active trading days (`70` days), which can inflate Sharpe. The runner was corrected before accepting the result: daily portfolio returns now use the full Step 14 prediction-context test calendar and fill non-trading/no-position days with zero return. Final accepted run uses `173` calendar trading days, with `70` nonzero-position days.

Artifacts verified:
```text
outputs/metrics/15_v6_backtest_track_baseline.json
outputs/tables/15_v6_daily_returns.csv
outputs/tables/15_v6_track_breakdown.csv
outputs/manifests/15_BACKTEST_TRACK_BASELINE_V6.manifest.json
outputs/status/15_BACKTEST_TRACK_BASELINE_V6.status.json
```

Final command run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.eval.backtest_daily_portfolio_v3 --predictions outputs/predictions/current_v6_dpo_predictions.parquet --contexts data/processed/current_v6_prediction_contexts.parquet --output-daily outputs/tables/15_v6_daily_returns.csv --output-track outputs/tables/15_v6_track_breakdown.csv --metrics outputs/metrics/15_v6_backtest_track_baseline.json --status outputs/status/15_BACKTEST_TRACK_BASELINE_V6.status.json --manifest outputs/manifests/15_BACKTEST_TRACK_BASELINE_V6.manifest.json --cost-bps 5 --slippage-bps 2 --short-borrow-bps 1 --min-trading-days 60 --min-schema-ok-rate 0.90
```

Final metrics:
```text
num_trading_days: 173
nonzero_position_days: 70
schema_ok_rate: 0.9167
sharpe_daily_annualized: 0.9667
sortino_daily_annualized: 1.1606
max_drawdown: 0.1474
mean_daily_return: 0.001222
avg_positions_per_day: 0.5087
total_turnover: 20.0
coverage: 0.4046
technical_rule_num_trading_days: 173
technical_rule_sharpe_annualized: -1.4211
technical_rule_mean_daily_return: -0.002325
delta_sharpe_vs_technical_rule: 2.3878
delta_mean_daily_return_vs_technical_rule: 0.003547
no_news_baseline_available: false
```

Acceptance result:
```text
status JSON gate: PASS
schema gate: PASS (0.9167 >= 0.90)
trading-day gate: PASS (173 >= 60)
turnover gate: PASS (20.0 > 0)
daily calendar anti-inflation check: PASS (173 rows, not 70 active-only rows)
Technical_Rule comparison: PASS as diagnostic (DPO Sharpe > Technical_Rule Sharpe)
unit/full tests: PASS with D:\LOBProj\LOBExp\.venv\Scripts\python.exe
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m py_compile src\eval\backtest_daily_portfolio_v3.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_backtest_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/15_BACKTEST_TRACK_BASELINE_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_backtest_v6.py tests/test_v6_predictions.py tests/test_prediction_contexts_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests
```

Test result:
```text
77 passed, 2 warnings
```

Scientific caution: Step 15 supports only the after-cost daily backtest diagnostic for the selected 300-row V6 prediction-context artifact. It does not override the weak Step 14 classification metrics, and it is not enough for an AAAI/paper alpha claim until later baseline, ablation, and statistical gates pass. The current Step 15 artifact also has no no-news subset (`no_news_baseline_available=false`) because the selected Step 14 contexts all contain company-event evidence.
