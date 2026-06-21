# Sources and Review Basis

## Repo sources used
- `BaoCaoCodexFixSmallScale_20062026.md`: reports current V3.1 status, flow underperformance, negative backtest, partial counterfactual gains.
- `codex_fix_20062026/00_MASTER_CURRENTDATA_UPGRADE_ORDER.md`: defines current-data-first upgrade scope and bottlenecks.
- `src/llm/generate_rationales.py`: current generation code with split handling, render context, raw/parsed outputs.
- `src/eval/backtest_daily_portfolio_v3.py`: current daily portfolio simulator.
- `src/reward/evaluate_flow_vs_proxy_v3.py`: current flow/proxy evaluation.

## External research basis
- FNSPID: 29.7M stock price records and 15.7M time-aligned news records for 4,775 S&P500 companies, 1999–2023. Therefore current-data results must not be claimed as full-scale.
- SEP: Summarize-Explain-Predict shows stock-text prediction requires reducing chaotic text into useful signals before explanation/prediction.
- BenchStock: prediction accuracy does not necessarily correlate with portfolio return; realistic backtesting and transaction costs matter.
- AAAI reproducibility checklist: preprocessing code, hyperparameters, and reproducible artifacts are expected.
- Policy/flow paper in repo: motivates inferability-based rationale and distributional reward, but V4 must not claim flow improvement unless it empirically beats proxy.
