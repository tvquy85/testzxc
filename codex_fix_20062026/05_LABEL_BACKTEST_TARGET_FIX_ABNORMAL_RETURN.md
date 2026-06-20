# 05 — Enforce Abnormal Return as Target

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Make all current-data prediction and backtest steps use abnormal return, not raw stock return.

## Inputs
- `data/processed/ticker_date_contexts_h1_v2.parquet`
- `data/labels/labels_h1_abnormal.parquet`

## Outputs
- `data/processed/ticker_date_contexts_h1_v2_targets.parquet`
- `outputs/metrics/target_integrity_h1_v2.json`
- `outputs/status/05_LABEL_BACKTEST_TARGET_FIX_ABNORMAL_RETURN.status.json`

## Codex task
Create `src/data/ensure_abnormal_targets_v2.py`.

## Required normalized columns
- `target_return = abnormal_return_h1`
- `target_label_5 = label_5`
- `target_direction = down|neutral|up`

## Run
```bash
python -m src.data.ensure_abnormal_targets_v2 \
  --contexts data/processed/ticker_date_contexts_h1_v2.parquet \
  --labels data/labels/labels_h1_abnormal.parquet \
  --output data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --metrics outputs/metrics/target_integrity_h1_v2.json \
  --status outputs/status/05_LABEL_BACKTEST_TARGET_FIX_ABNORMAL_RETURN.status.json
```

## Verify
```bash
python - <<'PY'
import pandas as pd, json
df=pd.read_parquet("data/processed/ticker_date_contexts_h1_v2_targets.parquet")
assert "target_return" in df.columns
assert "target_label_5" in df.columns
assert df["target_return"].notna().mean() > 0.95
print(json.load(open("outputs/metrics/target_integrity_h1_v2.json")))
PY
```

## Acceptance
- `target_return` must be abnormal return.
- Missing target rate < 5%.
