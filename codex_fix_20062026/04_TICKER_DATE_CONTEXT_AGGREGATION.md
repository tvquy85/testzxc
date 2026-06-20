# 04 — Aggregate to One Context per Ticker-Date

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Reduce per-news noise by aggregating rows to `ticker × event_date × horizon × split`.

## Inputs
- `data/processed/current_filtered_samples_v2.parquet`
- `data/indicators/technical_event_tokens_h1_v2.parquet`

## Outputs
- `data/processed/ticker_date_contexts_h1_v2.parquet`
- `outputs/metrics/ticker_date_contexts_h1_v2.json`
- `outputs/status/04_TICKER_DATE_CONTEXT_AGGREGATION.status.json`

## Codex task
Create `src/data/build_ticker_date_contexts_v2.py`.

## Output columns
- `sample_id`: deterministic SHA256 of ticker/date/horizon
- `source_sample_ids`: JSON list
- `ticker`, `event_date`, `split`, `horizon`
- `aggregated_headlines`, `aggregated_body`
- `context_news_count`, `mean_news_quality_score`
- `label_5`, `abnormal_return_h1`, `label_conflict_flag`
- `technical_event_tokens_json`

## Run
```bash
python -m src.data.build_ticker_date_contexts_v2 \
  --input data/processed/current_filtered_samples_v2.parquet \
  --tokens data/indicators/technical_event_tokens_h1_v2.parquet \
  --output data/processed/ticker_date_contexts_h1_v2.parquet \
  --metrics outputs/metrics/ticker_date_contexts_h1_v2.json \
  --status outputs/status/04_TICKER_DATE_CONTEXT_AGGREGATION.status.json
```

## Verify
```bash
python - <<'PY'
import pandas as pd, json
x=pd.read_parquet("data/processed/ticker_date_contexts_h1_v2.parquet")
assert len(x)>0
assert x.duplicated(["ticker","event_date","horizon","split"]).sum()==0
assert x["context_news_count"].min()>=1
print(json.load(open("outputs/metrics/ticker_date_contexts_h1_v2.json")))
PY
```

## Acceptance
- No duplicate ticker-date-horizon-split rows.
- Label conflict rate is reported.
