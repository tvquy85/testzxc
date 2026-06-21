# 17 — Backtest, Counterfactual, and Ablation V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Evaluate clean V4 by track and produce real, non-dummy ablations.

## Inputs
- `data/processed/ticker_date_evidence_contexts_h1_v4.parquet`
- `data/processed/current_track_assignments_v4.parquet`
- aligned model/checkpoint if available.

## Outputs
- `outputs/predictions/current_clean_v4_test_predictions.parquet`
- `outputs/metrics/backtest_daily_portfolio_current_clean_v4.json`
- `outputs/tables/backtest_daily_returns_current_clean_v4.csv`
- `data/counterfactual/current_clean_v4_cf_tasks.parquet`
- `outputs/metrics/counterfactual_directional_current_clean_v4.json`
- `outputs/tables/ablation_current_clean_v4.csv`
- `outputs/status/17_BACKTEST_COUNTERFACTUAL_ABLATION_V4.status.json`

## Backtest rules
- Report all_test, news_technical_only, technical_only.
- Use `abnormal_return_h1` if available.
- Include transaction cost, turnover, max drawdown, Sharpe.
- Trading alpha claim blocked if Sharpe <= 0.

## Counterfactual task types
`remove_positive_evidence`, `remove_negative_evidence`, `remove_all_company_evidence`, `neutralize_bearish_technical`, `neutralize_bullish_technical`.

## Ablations
`Full_Clean_V4`, `No_News_Evidence`, `No_Technical_Tokens`, `Technical_Only_Track`, `News_Technical_Track`, `No_Flow_Reward`, `SFT_Only_or_Base_Model`.

## Verification gates
- No `NOT_RUN` ablation rows.
- No dummy values.
- Counterfactual no-change rate reported.
- News counterfactual pass rate must be reported separately.
