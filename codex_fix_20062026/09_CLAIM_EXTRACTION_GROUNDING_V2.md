# 09 — Claim Extraction and Grounding V3

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Improve grounding beyond exact token mention.

## Inputs
- clean contexts
- parsed rationales
- technical tokens
- optional NLI model

## Outputs
- `data/judges/claim_grounding_v3.parquet`
- `outputs/metrics/claim_grounding_v3.json`
- `outputs/data_samples/claim_grounding_v3_examples.json`
- `outputs/status/09_CLAIM_EXTRACTION_GROUNDING_V2.status.json`

## Codex task
Create `src/judges/claim_level_grounding_v3.py`.

## Required logic
- Extract claims from `news_rationale`, `technical_rationale`, `conflict_resolution`, `risk_note`.
- Technical aliases:
  - RSI overbought -> `RSI_OVERBOUGHT`
  - MACD bearish -> `MACD_BEARISH`, `MACD_BEARISH_CROSS`
  - volume spike -> `VOLUME_SPIKE`, `HIGH_VOLUME`
  - price above SMA -> `PRICE_ABOVE_SMA20`
- News grounding uses aggregated headline/body; empty premise -> `unverified`.
- If NLI model exists, use entailment/contradiction; otherwise conservative keyword fallback.
- Save contradiction/unverified examples.

## Run
```bash
python -m src.judges.claim_level_grounding_v3 \
  --contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --rationales data/rationales/parsed/current_clean_train_qwen3_4b_v3.parquet \
  --tokens data/indicators/technical_event_tokens_h1_v2.parquet \
  --output data/judges/claim_grounding_v3.parquet \
  --metrics outputs/metrics/claim_grounding_v3.json \
  --status outputs/status/09_CLAIM_EXTRACTION_GROUNDING_V2.status.json \
  --examples outputs/data_samples/claim_grounding_v3_examples.json
```

## Verify
```bash
python - <<'PY'
import pandas as pd, json
df=pd.read_parquet("data/judges/claim_grounding_v3.parquet")
m=json.load(open("outputs/metrics/claim_grounding_v3.json"))
assert len(df)>0
assert "supported_rate" in m
assert ('supported', 'unverified', 'not_applicable') & set(df["status"].unique())
print(m)
PY
```

## Acceptance
- Supported/unverified/not_applicable reported.
- Supported rate >= 0.15 or warning recorded.
