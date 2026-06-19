# 06 — Technical features v2 with validation

## Goal

Make technical indicators reliable, leakage-safe, and testable.

## Existing files to inspect

```text
src/features/compute_technical_indicators.py
```

## Codex task

1. Refactor indicator computation into pure functions.
2. Add unit tests for:
   - RSI bounds between 0 and 100;
   - MACD columns are finite after warmup;
   - rolling windows do not use future rows;
   - Bollinger width is non-negative;
   - volume z-score handles zero std.
3. Add feature manifest with:
   - feature names;
   - lookback window;
   - whether the feature is price-only or market-relative;
   - leakage safety note.

## Outputs

```text
data/indicators/technical_features_h1_v2.parquet
outputs/manifests/technical_feature_manifest.json
outputs/status/06_TECHNICAL_FEATURES_V2.status.json
```

## Verification commands

```bash
pytest -q tests/test_technical_features.py
python src/features/compute_technical_indicators.py \
  --samples data/labels/labels_h1_abnormal.parquet \
  --prices data/processed/prices_mvp.parquet \
  --output data/indicators/technical_features_h1_v2.parquet
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/indicators/technical_features_h1_v2.parquet")
print(df.shape)
print(df.isna().mean().sort_values(ascending=False).head())
PY
```

## Acceptance criteria

- Warmup missingness is explicitly reported, not hidden.
- No future-looking feature is present.
- Feature file row count matches labeled sample count after documented warmup filtering.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "06_TECHNICAL_FEATURES_V2",
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
outputs/status/06_TECHNICAL_FEATURES_V2.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
