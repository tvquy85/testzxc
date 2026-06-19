# Sources and Design Notes

> **Use with Antigravity / Gemini Pro 3.1 High**  
> This is a design note, not an execution step.

## Key sources used

### FNSPID
- Hugging Face dataset: `Zihan1004/FNSPID`
- Paper page: `https://huggingface.co/papers/2402.06698`
- Reported scale: 29.7M stock prices and 15.7M time-aligned financial news records for 4,775 S&P 500 companies, 1999-2023.

### Translate Policy to Language
- User-provided ICLR 2026 PDF.
- Key design borrowed:
  - Explanation LLM does not see the true decision.
  - Proxy LLMs judge whether the explanation implies the true decision.
  - Rectified flow learns a distributional reward model from proxy logits.
  - Dense sentence-level reward is based on changes in true-decision probability.

### Local model roles
- `Qwen2.5-3B-Instruct` / `Qwen3-4B`: rationale generation/alignment.
- `ProsusAI/finbert`: sentiment baseline.
- `cross-encoder/nli-deberta-v3-small`: grounding and contradiction checks.
- `amazon/chronos-bolt-small`: zero-shot time-series baseline.
- `ibm-granite/granite-timeseries-ttm-r2`: lightweight time-series baseline.
- `AutonLab/MOMENT-1-small`: frozen time-series embedding/baseline.

## Design choices for RTX 3090

- Use Qwen 3B/4B with QLoRA.
- Use local judges sequentially, not concurrently.
- Cache rationales, judge logits, embeddings, rewards.
- Use reward-weighted SFT and DPO before PPO.
- Use lightweight conditional MLP rectified flow, not the full cross-attention flow from the paper.

## Main novelty to preserve

1. Technical Indicator Semanticization.
2. Regime-conditioned Flow Reward.
3. Counterfactual Rationale Alignment.

## Financial leakage warnings

Never let the model see realized return label, future prices, post-event stock reaction language, or analyst commentary published after event date.

## Minimal claim discipline

Do not claim profitable trading until transaction-cost backtest passes. Do not claim causal faithfulness unless counterfactual tests pass. Do not claim flow recovers true market reward; claim it models noisy heterogeneous proxy rewards if empirically supported.
