# FIRE-Fin-Lite Execution Order

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Objective
Build a reproducible FIRE-Fin-Lite research pipeline for explainable multimodal stock forecasting using:

- FNSPID financial news + stock prices.
- Technical indicators converted into semantic event tokens.
- Local LLMs available on this machine.
- Lightweight flow-matched reward model.
- Reward-weighted SFT / DPO instead of full PPO at the first stage.

The pipeline is designed for a single RTX 3090. Do not attempt full 8B PPO in the first implementation.

## Source idea
The source paper trains an Explanation LLM to generate explanations without seeing the true decision, then uses Proxy LLMs and a rectified-flow reward model to score whether the explanation lets a third party infer the correct decision. FIRE-Fin-Lite adapts that idea to stock forecast labels.

## Execute files in this order

1. `01_ENVIRONMENT_AND_MODEL_INVENTORY.md`
2. `02_DATA_DOWNLOAD_AND_LOCAL_CHECKS.md`
3. `03_FNSPID_SCHEMA_AND_SAMPLE_BUILD.md`
4. `04_PRICE_NEWS_ALIGNMENT_AND_LABELS.md`
5. `05_TECHNICAL_INDICATORS_AND_EVENT_TOKENS.md`
6. `06_BASELINES_FINBERT_TECH_TS.md`
7. `07_RATIONALE_SCHEMA_AND_PROMPTS.md`
8. `08_GENERATE_RATIONALES_LOCAL_LLM.md`
9. `09_PROXY_JUDGES_AND_GROUNDING.md`
10. `10_FLOW_REWARD_MODEL_LITE.md`
11. `11_ALIGNMENT_RWSFT_DPO.md`
12. `12_COUNTERFACTUAL_EVAL_AND_BACKTEST.md`
13. `13_ABLATIONS_AND_PAPER_TABLES.md`
14. `14_FINAL_AAAI_REPRO_PACKAGE.md`

## Global directory layout
Create this layout exactly:

```text
firefin/
  configs/
  data/
    raw/
    interim/
    processed/
    samples/
    indicators/
    labels/
    rationales/
    judge_outputs/
    embeddings/
  docs/
  models/
  outputs/
    metrics/
    tables/
    figures/
    checkpoints/
    logs/
    status/
  prompts/
  src/
    data/
    features/
    baselines/
    llm/
    judges/
    reward/
    align/
    eval/
    utils/
  tests/
```

## Global rules

- Use time-based split only; never random split across dates.
- Avoid leakage: no future price, post-event headline, or “shares surged/plunged after...” text unless the timestamp proves it was known before prediction.
- Main local LLM for generation: prefer `Qwen2.5-3B-Instruct-cf-finflow` if available; otherwise `Qwen3-4B-Instruct-2507`.
- Judge models: `Qwen3-4B-Instruct-2507`, `DeepSeek-R1-Distill-Qwen-1.5B`, FinGPT LoRA model if loadable.
- Grounding model: `cross-encoder--nli-deberta-v3-small`.
- Sentiment baseline: `ProsusAI--finbert`.
- Time-series baselines: `MOMENT-1-small`, `chronos-bolt-small`, `granite-timeseries-ttm-r2`.
- All scripts must accept `--limit`, `--seed`, `--output`, and `--dry-run` whenever relevant.
- Every step must save `outputs/status/STEP_NAME.status.json`.

## Stop condition
After each file, stop and report:

```json
{
  "step": "...",
  "status": "PASS|FAIL",
  "created_files": [],
  "row_counts": {},
  "next_step": "..."
}
```

## Expected first MVP scale

- Tickers: 30 to 50 liquid stocks.
- Period: start with 2018-2023 if available.
- Samples: 30k to 80k aligned news-price samples.
- Rationales: 10k to 30k candidate rationales.
- Horizon: start with 1 trading day. Add 5 trading days later.
