# 02 — Audit Hard-Event Failure Modes in Current Evidence Packs

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Quantify how much current data is genuinely company-specific hard-event news versus weak/listicle/multi-company/general news.

## Inputs
```text
data/processed/*evidence*.parquet
review_samples/clean_v4_medium_21062026/03_medium_context_and_evidence_pack_samples.jsonl
```

## Outputs
```text
outputs/metrics/02_v6_hard_event_data_audit.json
outputs/tables/02_v6_article_type_distribution.csv
review_samples/currentdata_v6/02_weak_evidence_examples.jsonl
outputs/status/02_HARD_EVENT_DATA_AUDIT.status.json
```

## Implementation
Create `src/data/audit_hard_event_v6.py`. Define:
```python
HARD_TYPES={'earnings_or_guidance','analyst_rating','company_event','merger_acquisition','regulatory_or_lawsuit','product_or_contract'}
BAD_TYPES={'empty_or_weak','opinion_listicle','multi_company_roundup','sector_etf','macro_market'}

def classify_context_hardness(evidence_pack_json):
    import json
    pack=json.loads(evidence_pack_json) if isinstance(evidence_pack_json,str) else evidence_pack_json
    hard=sum(1 for x in pack if x.get('article_type') in HARD_TYPES)
    bad=sum(1 for x in pack if x.get('article_type') in BAD_TYPES)
    n=max(len(pack),1)
    return {'num_hard_event_evidence':hard,'num_bad_evidence':bad,'hard_event_ratio':hard/n,'weak_evidence_ratio':bad/n,'has_hard_event_news':hard>0}
```

## Required metrics
```json
{
  "contexts": 0,
  "hard_event_context_rate": 0.0,
  "weak_or_bad_context_rate": 0.0,
  "multi_company_context_rate": 0.0,
  "empty_or_weak_context_rate": 0.0,
  "recommended_hard_event_rows": 0
}
```

## Test case
```python
import json
from src.data.audit_hard_event_v6 import classify_context_hardness

def test_classify_hard_event_context():
    pack=json.dumps([{'evidence_id':'N1','article_type':'earnings_or_guidance','evidence_quality_score':0.85},{'evidence_id':'M1','article_type':'macro_market'}])
    out=classify_context_hardness(pack)
    assert out['num_hard_event_evidence']==1
    assert out['has_hard_event_news'] is True
```

## Commands
```bash
python -m src.data.audit_hard_event_v6 --input data/processed/ticker_date_evidence_contexts_h1_v4.parquet --output-metrics outputs/metrics/02_v6_hard_event_data_audit.json --output-table outputs/tables/02_v6_article_type_distribution.csv --samples review_samples/currentdata_v6/02_weak_evidence_examples.jsonl --status outputs/status/02_HARD_EVENT_DATA_AUDIT.status.json
python -m pytest -q tests/test_hard_event_audit_v6.py tests
```
