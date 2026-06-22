# 04 — Repair Evidence Packs and Enforce Evidence ID Consistency

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Ensure every evidence pack has valid IDs, valid article type, non-empty text, and no invalid references.

## Outputs
```text
data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet
outputs/metrics/04_v6_evidence_repair.json
outputs/status/04_EVIDENCE_PACK_REPAIR_V6.status.json
```

## Helper code
Create `src/data/repair_evidence_pack_v6.py` with:
```python
def validate_evidence_pack(pack):
    ids=set(); failures=[]
    for item in pack:
        eid=item.get('evidence_id')
        if not eid or eid in ids: failures.append(f'duplicate_or_missing_id:{eid}')
        ids.add(eid)
        if eid and eid.startswith('N') and item.get('article_type') in {'empty_or_weak','macro_market','sector_etf'}:
            failures.append(f'invalid_news_id_type:{eid}')
        if not (item.get('headline') or item.get('body_excerpt')):
            failures.append(f'empty_text:{eid}')
    return failures
```

## Test case
```python
from src.data.repair_evidence_pack_v6 import validate_evidence_pack

def test_reject_empty_news_evidence():
    pack=[{'evidence_id':'N1','article_type':'empty_or_weak','headline':'','body_excerpt':''}]
    assert validate_evidence_pack(pack)
```

## Commands
```bash
python -m src.data.repair_evidence_pack_v6 --input data/processed/ticker_date_evidence_contexts_h1_v6.parquet --output data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --metrics outputs/metrics/04_v6_evidence_repair.json --samples review_samples/currentdata_v6/04_repaired_context_examples.jsonl --status outputs/status/04_EVIDENCE_PACK_REPAIR_V6.status.json
python -m pytest -q tests/test_evidence_pack_repair_v6.py tests
```

## Acceptance
Invalid evidence ID rate = 0; no `empty_or_weak` item appears with an `N*` ID.

## Progress Update 2026-06-22
Status: `PASS`; status file reports `next_step_allowed=true`.

Repair note: the original Step 04 run failed because the validator expected `evidence_pack_json` to decode to a list. The actual V6 evidence-pack schema is a dict with `company_evidence`, `context_evidence`, and `technical_signals`. The repair code now validates and preserves that schema.

Final metrics:
```text
total_rows: 3129
repaired_rows: 3
rows_with_failures: 0
split_distribution: train=2453, val=390, test=286
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_evidence_pack_repair_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.data.repair_evidence_pack_v6 --input data/processed/ticker_date_evidence_contexts_h1_v6.parquet --output data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --metrics outputs/metrics/04_v6_evidence_repair.json --samples review_samples/currentdata_v6/04_repaired_context_examples.jsonl --status outputs/status/04_EVIDENCE_PACK_REPAIR_V6.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/04_EVIDENCE_PACK_REPAIR_V6.status.json
```
