# Step 05 — Technical Indicators and Semantic Event Tokens

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Compute technical indicators from price windows and convert them into LLM-readable event tokens. This is a core novelty component.

## Inputs

```text
data/labels/aligned_samples_h1.parquet
data/processed/prices_mvp.parquet
```

## Outputs

```text
data/indicators/technical_features_h1.parquet
data/indicators/technical_event_tokens_h1.parquet
outputs/status/05_TECHNICAL_INDICATORS_AND_EVENT_TOKENS.status.json
```

## Tasks

### 1. Create scripts

```text
src/features/compute_technical_indicators.py
src/features/compile_technical_event_tokens.py
src/features/technical_rules.py
```

### 2. Compute numeric indicators
For each `sample_id`, compute indicators using only prices up to `window_end_date`, never event date or future date.

Required indicators:

```text
ret_1d, ret_5d, ret_20d
volatility_10d, volatility_20d
RSI_14
MACD, MACD_signal, MACD_hist
SMA_5, SMA_20, SMA_60
price_vs_SMA20, price_vs_SMA60
Bollinger_position_20, Bollinger_width_20
ATR_14
volume_zscore_20
relative_strength_vs_market_5d
relative_strength_vs_market_20d
gap_pct_last_day
```

If `pandas-ta` is unavailable, implement manually.

### 3. Compile event tokens
Create tokens such as:

```text
[RSI_OVERBOUGHT: RSI14=74.2, strength=medium]
[RSI_OVERSOLD: RSI14=26.8, strength=medium]
[MACD_BULLISH: hist=0.18]
[MACD_BEARISH: hist=-0.21]
[PRICE_ABOVE_SMA20: distance=6.4%]
[PRICE_BELOW_SMA20: distance=-4.2%]
[BOLLINGER_UPPER_PRESSURE: position=0.92]
[BOLLINGER_LOWER_PRESSURE: position=0.08]
[VOLUME_SPIKE: zscore=2.3]
[VOLUME_DRY_UP: zscore=-1.4]
[SECTOR_OR_MARKET_UNDERPERFORMANCE_5D: value=-3.1%]
[HIGH_VOLATILITY_REGIME: vol20_pctile=0.84]
```

Store:

```text
sample_id, ticker, event_date, technical_event_tokens, technical_summary_text
```

`technical_event_tokens` should be a JSON list string.

### 4. Regime label
Create `regime_label`:

```text
low_vol | normal_vol | high_vol
```

Use rolling market proxy volatility percentile:

```text
low_vol: <= 33rd percentile
normal_vol: 33rd-66th
high_vol: >= 66th
```

## Verification
Run:

```bash
cd firefin
python src/features/compute_technical_indicators.py \
  --samples data/labels/aligned_samples_h1.parquet \
  --prices data/processed/prices_mvp.parquet \
  --output data/indicators/technical_features_h1.parquet

python src/features/compile_technical_event_tokens.py \
  --features data/indicators/technical_features_h1.parquet \
  --output data/indicators/technical_event_tokens_h1.parquet
```

Then:

```bash
python - <<'PYCHECK'
import pandas as pd, json
f = pd.read_parquet('data/indicators/technical_features_h1.parquet')
t = pd.read_parquet('data/indicators/technical_event_tokens_h1.parquet')
print(f.shape, t.shape)
assert f['RSI_14'].between(0, 100).mean() > 0.95
assert t['technical_event_tokens'].notna().mean() > 0.95
sample_tokens = json.loads(t.iloc[0]['technical_event_tokens'])
assert isinstance(sample_tokens, list)
PYCHECK
```

## Acceptance criteria
PASS only if:

- Numeric indicators exist for at least 95% of aligned samples.
- Event tokens exist for at least 95% of aligned samples.
- No indicator uses event-day or future prices.
- `regime_label` exists and has at least 2 categories.

## Status JSON

```json
{
  "step": "05_TECHNICAL_INDICATORS_AND_EVENT_TOKENS",
  "status": "PASS|FAIL",
  "feature_rows": 0,
  "token_rows": 0,
  "missing_rate": 0.0,
  "regime_distribution": {},
  "example_tokens": []
}
```
