# 03 — Build Hard-Event Track V6 Filter

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Create a stronger current-data subset for news+technical reasoning. Do not delete old data. Add a V6 hard-event track to prevent technical-only template learning.

## Inputs
```text
data/processed/ticker_date_evidence_contexts_h1_v4.parquet
outputs/metrics/02_v6_hard_event_data_audit.json
```

## Outputs
```text
data/processed/ticker_date_evidence_contexts_h1_v6.parquet
outputs/metrics/03_v6_hard_event_filter.json
outputs/tables/03_v6_track_distribution.csv
review_samples/currentdata_v6/03_hard_event_context_examples.jsonl
outputs/status/03_HARD_EVENT_FILTER_V6.status.json
```

## Logic
Create `src/data/filter_hard_event_v6.py`:
```python
def assign_v6_track(row):
    if row.get('num_hard_event_evidence',0) >= 1 and row.get('mean_evidence_quality_score',0) >= 0.55:
        return 'hard_event_news'
    if row.get('has_company_event_news',False) and row.get('mean_evidence_quality_score',0) >= 0.45:
        return 'company_general_news'
    return 'weak_or_context_only'

def training_weight(track):
    return {'hard_event_news':1.0,'company_general_news':0.55,'weak_or_context_only':0.10}[track]
```

## Test case
```python
from src.data.filter_hard_event_v6 import assign_v6_track

def test_hard_event_track():
    row={'num_hard_event_evidence':1,'mean_evidence_quality_score':0.70,'has_company_event_news':True}
    assert assign_v6_track(row)=='hard_event_news'

def test_weak_track():
    row={'num_hard_event_evidence':0,'mean_evidence_quality_score':0.30,'has_company_event_news':False}
    assert assign_v6_track(row)=='weak_or_context_only'
```

## Commands
```bash
python -m src.data.filter_hard_event_v6 --input data/processed/ticker_date_evidence_contexts_h1_v4.parquet --output data/processed/ticker_date_evidence_contexts_h1_v6.parquet --metrics outputs/metrics/03_v6_hard_event_filter.json --table outputs/tables/03_v6_track_distribution.csv --samples-dir review_samples/currentdata_v6 --status outputs/status/03_HARD_EVENT_FILTER_V6.status.json
python -m pytest -q tests/test_hard_event_filter_v6.py tests
```

## Acceptance
`hard_event_news` count >=300 preferred. Exclude `weak_or_context_only` from multimodal training pool.
