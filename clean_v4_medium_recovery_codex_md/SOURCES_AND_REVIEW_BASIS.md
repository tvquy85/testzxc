# Sources and Review Basis

## Repository artifacts used

- `BaoCaoCodexFixCleanData_20062026.md` on branch `currentdata-aaai-fix-v2`: reports DataClean V4 small-scale state, including `GO_SMALL`, `CLAIM_RESTRICTED`, evidence-pack metrics, Flow V4 small-slice signal, negative Sharpe, and next-step recommendation.
- `dataclean_v4_codex_md/00_MASTER_DATACLEAN_V4_ORDER.md`: defines current-data-only scope, no SN2, no full FNSPID expansion, no overwrite of V3 artifacts, status JSON for every step, and claim gating.
- `src/reward/build_flow_dataset_v4.py`: current Flow target vector and hash-based condition backend.
- `src/reward/evaluate_flow_vs_proxy_v4.py`: evaluates Flow vs proxy via rank correlation, pair accuracy, and top-decile utility.
- Existing folders: `PEN/`, `sep/`, `policy/`.

## External references to respect

- AAAI reviewer standards: novelty, significance, technical quality, empirical rigor, reproducibility, clarity, and responsible research practice.
- FNSPID: large-scale financial news/price benchmark; current pack does not expand to full scale.
- PEN: explainable stock movement forecasting with text-price alignment.
- SEP: self-reflective LLM framework for explainable stock predictions using summarize-explain-predict and PPO.
- Policy paper: inferability-based explanation training with proxy LLM logits and flow-matched reward.
- Financial backtesting benchmarks: accuracy alone is insufficient; use daily portfolio returns, transaction costs, turnover, drawdown, and multi-period validation.
- LLM-as-judge bias work: motivates label-order debias and judge stability tests.

## Claim discipline

Allowed after this pack only if gates pass:

```text
medium-scale current-data diagnostic result
evidence-grounded rationale pipeline
counterfactual improvement if supported
Flow improvement if validation wins
```

Not allowed at medium stage by default:

```text
AAAI main-ready
full-scale FNSPID result
profitable trading alpha
outperforming PEN/SEP/Policy
general superiority of Flow Reward
```
