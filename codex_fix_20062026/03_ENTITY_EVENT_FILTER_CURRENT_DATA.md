# 03 — Entity/Event Filter on Current Data

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Create a filtered subset from current data without full-scale expansion.

## Inputs
- `data/labels/labels_h1_abnormal.parquet`
- `data/quality/current_data_quality_v2.parquet`
- `data/processed/split_membership.parquet`

## Outputs
- `data/processed/current_filtered_samples_v2.parquet`
- `outputs/metrics/current_filtered_samples_v2.json`
- `outputs/status/03_ENTITY_EVENT_FILTER_CURRENT_DATA.status.json`

## Filter rule
Keep rows with:
- `news_quality_score >= 0.45`
- not `low_quality_text_flag`
- text exists
- ticker mentioned OR not strongly multi-ticker generic

## Codex task
Create `src/data/filter_current_samples_v2.py`.

## Run
```bash
python -m src.data.filter_current_samples_v2 \
  --labels data/labels/labels_h1_abnormal.parquet \
  --quality data/quality/current_data_quality_v2.parquet \
  --splits data/processed/split_membership.parquet \
  --min-quality 0.45 \
  --output data/processed/current_filtered_samples_v2.parquet \
  --metrics outputs/metrics/current_filtered_samples_v2.json \
  --status outputs/status/03_ENTITY_EVENT_FILTER_CURRENT_DATA.status.json
```

## Verify
```bash
python - <<'PY'
import pandas as pd, json
df=pd.read_parquet("data/processed/current_filtered_samples_v2.parquet")
m=json.load(open("outputs/metrics/current_filtered_samples_v2.json"))
assert len(df)==m["kept_rows"]
assert set(df["split"].unique()) <= ('train', 'val', 'test')
assert m["kept_rows"] > 10000
assert m["drop_rate"] < 0.80
print(m)
PY
```

## Acceptance
- At least 10k rows remain.
- All splits remain represented.
- Label distributions before/after are reported.
