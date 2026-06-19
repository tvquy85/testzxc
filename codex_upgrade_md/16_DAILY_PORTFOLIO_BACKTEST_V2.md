# 16 — Daily portfolio backtest v2

## Goal

Replace per-news-sample backtest with a finance-valid daily portfolio simulator.

## Existing file to replace

```text
src/eval/backtest_long_short_hold.py
```

Do not delete the old file; create a v2 file:

```text
src/eval/backtest_daily_portfolio_v2.py
```

## Required behavior

1. Aggregate multiple news predictions for the same `date × ticker`.
2. Produce at most one position per `ticker × date`.
3. Use out-of-sample test predictions only.
4. Support:
   - equal weight;
   - confidence weight;
   - volatility scaling.
5. Include costs:
   - transaction cost bps;
   - optional slippage bps;
   - optional short borrow bps.
6. Compute daily portfolio returns.
7. Annualize Sharpe from daily returns only.
8. Report turnover and coverage.

## Outputs

```text
outputs/metrics/backtest_daily_portfolio_v2.json
outputs/tables/backtest_daily_returns_v2.csv
outputs/figures/equity_curve_v2.png
```

## Verification commands

```bash
python src/eval/backtest_daily_portfolio_v2.py \
  --predictions outputs/predictions/test_predictions.parquet \
  --labels data/labels/labels_h1_abnormal.parquet \
  --split test \
  --cost-bps 5 \
  --output-json outputs/metrics/backtest_daily_portfolio_v2.json \
  --daily-returns outputs/tables/backtest_daily_returns_v2.csv
```

## Acceptance criteria

- Daily returns index has one row per trading date, not one row per news item.
- No date outside test split.
- Sharpe uses daily portfolio return series.
- Report `num_trading_days`, `avg_positions_per_day`, `turnover`, `coverage`, `max_drawdown`.
- If fewer than 60 trading days, mark backtest as insufficient.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "16_DAILY_PORTFOLIO_BACKTEST_V2",
  "status": "PASS|FAIL",
  "inputs_checked": [],
  "outputs_created": [],
  "metrics": {},
  "failures": [],
  "next_step_allowed": true
}
```

Save it to:

```text
outputs/status/16_DAILY_PORTFOLIO_BACKTEST_V2.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
