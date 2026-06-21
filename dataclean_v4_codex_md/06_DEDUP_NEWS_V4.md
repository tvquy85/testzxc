# 06 — Deduplicate News V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Remove exact and near-duplicate news within `ticker × event_date` so repeated articles do not dominate rationale generation.

## Inputs
- `data/quality/current_article_type_scores_v4.parquet`

## Outputs
- `data/processed/current_deduped_news_v4.parquet`
- `outputs/metrics/current_deduped_news_v4.json`
- `outputs/status/06_DEDUP_NEWS_V4.status.json`

## Required columns
`normalized_text_hash`, `near_dup_cluster_id`, `cluster_size`, `is_cluster_representative`, `dedup_keep`.

## Code snippet
```python
def normalized_for_dedup(headline, body):
    import re
    text = f'{headline or ""} {body or ""}'.lower()
    text = re.sub(r'https?://\S+', ' url ', text)
    text = re.sub(r'[^a-z0-9$%.\s-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def jaccard_tokens(a, b):
    A, B = set(a.split()), set(b.split())
    return 0.0 if not A or not B else len(A & B) / len(A | B)
```

Representative priority: highest evidence quality, highest event specificity, longest non-empty body, earliest timestamp.

## Command
```bash
python -m src.data.deduplicate_news_v4 \
  --input data/quality/current_article_type_scores_v4.parquet \
  --output data/processed/current_deduped_news_v4.parquet \
  --metrics outputs/metrics/current_deduped_news_v4.json \
  --status outputs/status/06_DEDUP_NEWS_V4.status.json
```

## Verification gates
- Output rows <= input rows.
- No ticker-date loses all A/B evidence if it originally had A/B evidence.
- Dedup drop rate reported.
