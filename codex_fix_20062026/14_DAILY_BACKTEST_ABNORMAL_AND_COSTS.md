# 14 — Daily Portfolio Backtest with Abnormal Return and Costs

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Re-run backtest using abnormal return and realistic constraints.

## Inputs
- `outputs/predictions/current_v3_test_predictions.parquet`
- `data/processed/ticker_date_contexts_h1_v2_targets.parquet`

## Outputs
- `outputs/metrics/backtest_daily_portfolio_current_v3.json`
- `outputs/tables/backtest_daily_returns_current_v3.csv`
- `outputs/status/14_DAILY_BACKTEST_ABNORMAL_AND_COSTS.status.json`

## Required fixes
Create/update `src/eval/backtest_daily_portfolio_v3.py`:
- use `target_return`
- aggregate by date × ticker
- one position per ticker per date
- cap top 20 positions/day
- costs: transaction, slippage, borrow
- report turnover, coverage, Sharpe, Sortino, max drawdown
- if trading days < 60, `alpha_claim_allowed=false`

## Run
```bash
python -m src.eval.backtest_daily_portfolio_v3 \
  --predictions outputs/predictions/current_v3_test_predictions.parquet \
  --contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --return-col target_return \
  --max-positions-per-day 20 \
  --threshold 0.20 \
  --cost-bps 5 --slippage-bps 2 --short-borrow-bps 1 \
  --metrics outputs/metrics/backtest_daily_portfolio_current_v3.json \
  --daily-output outputs/tables/backtest_daily_returns_current_v3.csv \
  --status outputs/status/14_DAILY_BACKTEST_ABNORMAL_AND_COSTS.status.json
```

## Verify
```bash
python - <<'PY'
import json, pandas as pd
m=json.load(open("outputs/metrics/backtest_daily_portfolio_current_v3.json"))
d=pd.read_csv("outputs/tables/backtest_daily_returns_current_v3.csv")
assert len(d)==m["num_trading_days"]
assert "daily_return_net" in d.columns
assert m["return_column"]=="target_return"
print(m)
PY
```

## Acceptance
- Uses abnormal return.
- Daily returns file valid.
- Negative Sharpe blocks alpha claim but not pipeline.
