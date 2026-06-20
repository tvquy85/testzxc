# Sources and Rationale

## Research sources
- FNSPID: large-scale news/price dataset, 29.7M stock-price records and 15.7M time-aligned news records for 4,775 S&P500 companies, 1999–2023. Current repo uses a small subset, so do not claim full-scale results. https://arxiv.org/abs/2402.06698
- SEP: close explainable stock-prediction baseline using self-reflective LLM + PPO. https://arxiv.org/abs/2402.03659 and https://github.com/koa-fin/sep
- PEN: AAAI 2023 Prediction-Explanation Network for stock movement explanation. https://ojs.aaai.org/index.php/AAAI/article/view/25648
- Policy-to-language flow reward paper: inspiration for inferability and flow-matched reward. Use the local `policy/` folder as reference, but do not claim equivalent architecture unless implemented.
- LLM-as-judge bias: use label-order randomization and stability checks. https://arxiv.org/abs/2406.07791
- Realistic financial evaluation: use daily portfolio returns, costs, turnover, and market constraints. BenchStock: https://openreview.net/pdf?id=de87306fbdbad74acf00a17ebc34a3e3688998e3. FinTSB: https://arxiv.org/abs/2502.18834
- AAAI reproducibility: checklist and supplementary material may be inspected by reviewers. https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/

## Non-negotiable implementation implications
- Improve current subset first.
- Do not use generator `forecast_distribution` as independent judge score.
- Do not report trading alpha if daily portfolio return is negative or test horizon is too short.
- Do not use NOT_RUN ablations as evidence.
- Do not hide invalid LLM output with auto-fix.
