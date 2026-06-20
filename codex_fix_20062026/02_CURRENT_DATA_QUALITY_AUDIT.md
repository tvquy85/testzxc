# 02 — Current Data Quality Audit

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Quantify data noise on the current dataset: empty body, generic macro news, multi-company articles, ticker mismatch, and weak text.

## Inputs
- `data/labels/labels_h1_abnormal.parquet`
- `data/processed/split_membership.parquet`

## Outputs
- `data/quality/current_data_quality_v2.parquet`
- `outputs/metrics/current_data_quality_v2.json`
- `outputs/status/02_CURRENT_DATA_QUALITY_AUDIT.status.json`

## Codex task
Create `src/data/current_data_quality_audit.py`.

## Required columns
For each row compute:
- `headline_len`, `body_len`
- `body_empty_flag`, `text_empty_flag`
- `ticker_in_headline_flag`, `ticker_in_body_flag`
- `multi_ticker_like_flag` = 3+ uppercase ticker-like tokens
- `generic_macro_flag`
- `low_quality_text_flag`
- `news_quality_score` in [0,1]

## Suggested scoring
```python
def quality_score(row):
    score = 1.0
    if row["body_empty_flag"]: score -= 0.25
    if row["low_quality_text_flag"]: score -= 0.30
    if row["multi_ticker_like_flag"]: score -= 0.20
    if row["generic_macro_flag"]: score -= 0.20
    if not (row["ticker_in_headline_flag"] or row["ticker_in_body_flag"]): score -= 0.20
    return max(0.0, min(1.0, score))
```

## Run
```bash
python -m src.data.current_data_quality_audit \
  --input data/labels/labels_h1_abnormal.parquet \
  --splits data/processed/split_membership.parquet \
  --output data/quality/current_data_quality_v2.parquet \
  --metrics outputs/metrics/current_data_quality_v2.json \
  --status outputs/status/02_CURRENT_DATA_QUALITY_AUDIT.status.json
```

## Verify
```bash
python - <<'PY'
import pandas as pd, json
q=pd.read_parquet("data/quality/current_data_quality_v2.parquet")
m=json.load(open("outputs/metrics/current_data_quality_v2.json"))
assert len(q)>0
assert "news_quality_score" in q.columns
assert q["news_quality_score"].between(0,1).all()
assert m["row_count"]==len(q)
print(m)
PY
```

## Acceptance
- Row count equals label dataset row count.
- Metrics include quality quantiles by split.
- No rows are dropped in this step.
