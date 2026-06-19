# 07 — Technical event tokens v2

## Goal

Upgrade technical event tokens from simple threshold labels to finance-grounded semantic tokens with direction and strength.

## Existing file to inspect

```text
src/features/compile_technical_event_tokens.py
```

## Codex task

Create `src/features/compile_technical_event_tokens_v2.py`.

Each token must include:

```json
{
  "token": "RSI_OVERBOUGHT",
  "value": 74.2,
  "direction_prior": "bearish_reversal_risk",
  "strength": "medium",
  "evidence_column": "RSI_14",
  "rule": "RSI_14 >= 70"
}
```

Support tokens for:
- RSI overbought / oversold;
- MACD bullish / bearish;
- price above/below SMA20;
- Bollinger upper/lower pressure;
- volume spike / dry-up;
- market outperformance / underperformance;
- high / normal / low volatility regime.

## Outputs

```text
data/indicators/technical_event_tokens_h1_v2.parquet
outputs/manifests/technical_token_rules.json
```

## Verification commands

```bash
python src/features/compile_technical_event_tokens_v2.py \
  --features data/indicators/technical_features_h1_v2.parquet \
  --output data/indicators/technical_event_tokens_h1_v2.parquet

python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/indicators/technical_event_tokens_h1_v2.parquet")
print(df[["sample_id","technical_event_tokens_json"]].head())
assert "technical_event_tokens_json" in df.columns
PY
```

## Acceptance criteria

- Tokens are JSON arrays, not only plain strings.
- Each token has `token`, `value`, `direction_prior`, `strength`, `rule`.
- Missing-token rate is reported.
- Do not mark no-token samples as perfect grounding later; no-token is a separate case.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "07_TECH_EVENT_TOKENS_V2",
  "status": "PASS|FAIL",
  "inputs_checked": [],
  "outputs_created": [],
  "metrics": {},
  "failures": [],
  "next_step_allowed": true
}
```

Save it to:

```text
outputs/status/07_TECH_EVENT_TOKENS_V2.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
