# 12 — Track Split and Training Pool V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Separate true news+technical contexts from technical-only/no-news contexts so training and claims are honest.

## Inputs
- `data/processed/ticker_date_evidence_contexts_h1_v4.parquet`

## Outputs
- `data/processed/current_clean_train_pool_v4.parquet`
- `data/processed/current_track_assignments_v4.parquet`
- `outputs/metrics/current_track_assignments_v4.json`
- `outputs/status/12_TRACK_SPLIT_AND_TRAIN_POOL_V4.status.json`

## Track labels
`news_technical`, `technical_only`, `context_only`, `noise_excluded`.

## Code
```python
def assign_track(row):
    if row['has_company_event_news'] and row['num_company_event_evidence'] > 0:
        return 'news_technical'
    if not row['has_company_event_news'] and row.get('technical_event_tokens_json'):
        return 'technical_only'
    if row['num_context_only_evidence'] > 0:
        return 'context_only'
    return 'noise_excluded'

def training_weight(track, mean_quality):
    if track == 'news_technical': return min(1.0, 0.5 + 0.5 * float(mean_quality or 0))
    if track == 'technical_only': return 0.35
    if track == 'context_only': return 0.15
    return 0.0
```

## Command
```bash
python -m src.data.build_current_train_pool_v4 \
  --contexts data/processed/ticker_date_evidence_contexts_h1_v4.parquet \
  --output data/processed/current_clean_train_pool_v4.parquet \
  --track-output data/processed/current_track_assignments_v4.parquet \
  --metrics outputs/metrics/current_track_assignments_v4.json \
  --status outputs/status/12_TRACK_SPLIT_AND_TRAIN_POOL_V4.status.json
```

## Gates
- Track distribution reported by split.
- Train `news_technical` rows > 0.
- `noise_excluded` has weight 0.
