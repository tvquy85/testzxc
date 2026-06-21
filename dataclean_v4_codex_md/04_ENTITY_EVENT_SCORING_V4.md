# 04 — Entity-Event Evidence Scoring V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Score each row for target-entity relevance and event specificity. This directly addresses noisy macro/multi-company/no-news contexts.

## Inputs
- `data/quality/current_data_quality_v2.parquet`
- `data/quality/ticker_alias_map_v4.json`

## Outputs
- `data/quality/current_entity_event_scores_v4.parquet`
- `outputs/metrics/current_entity_event_scores_v4.json`
- `outputs/status/04_ENTITY_EVENT_SCORING_V4.status.json`

## Required output columns
`target_entity_score`, `event_specificity_score`, `text_quality_score`, `evidence_quality_score`, `quality_tier`, `entity_event_keep`, `entity_event_drop_reason`.

## Core scoring code
Create `src/data/evidence_entity_event_score_v4.py` and include equivalent logic:

```python
EVENT_TERMS = {
    'earnings':1.0,'revenue':0.9,'eps':0.9,'guidance':1.0,
    'profit':0.8,'margin':0.7,'upgrade':0.9,'downgrade':0.9,
    'price target':0.8,'buyback':0.8,'dividend':0.7,
    'merger':1.0,'acquisition':1.0,'lawsuit':0.8,
    'regulatory':0.8,'recall':0.8,'contract':0.7,
}

def event_specificity_score(text: str) -> float:
    t = normalize_text(text)
    score = sum(w for term, w in EVENT_TERMS.items() if term in t)
    return min(1.0, score / 2.0)

def evidence_quality_score(row) -> float:
    return 0.40*row['target_entity_score'] + 0.30*row['event_specificity_score'] + 0.20*row.get('article_type_score',0.5) + 0.10*row['text_quality_score']

def quality_tier(score, target, article_type):
    if score >= 0.70 and target >= 0.50: return 'A_company_event'
    if score >= 0.55 and target >= 0.35: return 'B_relevant'
    if article_type in {'macro_market','sector_etf'}: return 'C_context_only'
    return 'D_noise'
```

## Command
```bash
python -m src.data.evidence_entity_event_score_v4 \
  --input data/quality/current_data_quality_v2.parquet \
  --aliases data/quality/ticker_alias_map_v4.json \
  --output data/quality/current_entity_event_scores_v4.parquet \
  --metrics outputs/metrics/current_entity_event_scores_v4.json \
  --status outputs/status/04_ENTITY_EVENT_SCORING_V4.status.json
```

## Verification gates
- Scores are in [0,1].
- `quality_tier` has no nulls.
- A+B rows >= 10% total rows; otherwise FAIL and report current data too noisy.
- D_noise rows are never `entity_event_keep=True`.
