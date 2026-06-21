# 16 — Backtest Track Breakdown Medium

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Run daily portfolio backtest with track-level breakdown.

## Why this is needed
Trading alpha is currently blocked. Medium results must show whether news+technical, technical-only, or soft-news tracks are helping/hurting.

## Files to create or modify
- Modify only the files explicitly named in the command section. If a required file is missing, create it under the stated path.

## Inputs
```text
outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet
data/processed/medium_clean_v4_contexts_gated.parquet
```

## Outputs
```text
outputs/metrics/16_backtest_track_breakdown_medium.json
outputs/tables/medium_track_breakdown.csv
outputs/tables/medium_daily_returns.csv
outputs/status/16_BACKTEST_TRACK_BREAKDOWN_MEDIUM.status.json
```

## Commands
```bash
python -m src.eval.backtest_daily_portfolio_v3 \
  --predictions outputs/predictions/medium_clean_v4_dpo_test_predictions.parquet \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --return-col abnormal_return_h1 \
  --transaction-cost-bps 5 --slippage-bps 5 --borrow-cost-bps 2 --position-cap 0.05 \
  --output outputs/metrics/16_backtest_track_breakdown_medium.json \
  --daily-output outputs/tables/medium_daily_returns.csv \
  --track-output outputs/tables/medium_track_breakdown.csv \
  --status outputs/status/16_BACKTEST_TRACK_BREAKDOWN_MEDIUM.status.json
```

## Verification
```bash
python - <<'PY'
import json, pandas as pd
m=json.load(open('outputs/metrics/16_backtest_track_breakdown_medium.json'))
assert 'sharpe_annualized' in m
assert 'alpha_claim_allowed' in m
tr=pd.read_csv('outputs/tables/medium_track_breakdown.csv')
assert len(tr) >= 1
print('PASS backtest', m['sharpe_annualized'])
PY
```

## Acceptance criteria
- Uses daily date×ticker portfolio, not per-news pseudo-returns.
- Alpha claim false unless Sharpe > 0, mean daily return > 0, and test days >= 60.
- Track breakdown exists.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "16_BACKTEST_TRACK_BREAKDOWN_MEDIUM",
  "status": "PASS|FAIL",
  "pipeline_pass": true,
  "claim_allowed": false,
  "inputs": [],
  "outputs": [],
  "metrics": {},
  "failures": [],
  "warnings": []
}
```
