# 05 — Article Type and Noise Filter V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Classify each news row into article types to separate company evidence from macro/context/noise.

## Inputs
- `data/quality/current_entity_event_scores_v4.parquet`

## Outputs
- `data/quality/current_article_type_scores_v4.parquet`
- `outputs/metrics/current_article_type_scores_v4.json`
- `outputs/status/05_ARTICLE_TYPE_AND_NOISE_FILTER_V4.status.json`

## Article labels
Use exactly: `earnings_or_guidance`, `analyst_rating`, `company_event`, `macro_market`, `sector_etf`, `multi_company_roundup`, `opinion_listicle`, `technical_backtest_article`, `empty_or_weak`.

## Rule snippet
```python
def classify_article_type(headline: str, body: str) -> str:
    text = normalize_text(f'{headline or ""} {body or ""}')
    if len(text.split()) < 20: return 'empty_or_weak'
    if any(x in text for x in ['earnings','revenue','eps','guidance','profit','margin']): return 'earnings_or_guidance'
    if any(x in text for x in ['upgrade','downgrade','price target','analyst']): return 'analyst_rating'
    if any(x in text for x in ['covered call','backtest','technical analysis','chart setup']): return 'technical_backtest_article'
    if any(x in text for x in ['etf','sector fund','index fund']): return 'sector_etf'
    if any(x in text for x in ['fed','cpi','inflation','interest rate','treasury yield','recession']): return 'macro_market'
    if count_ticker_like_tokens(text) >= 6: return 'multi_company_roundup'
    if any(x in text for x in ['why we sold','top stocks','best stocks','stocks to buy']): return 'opinion_listicle'
    return 'company_event'
```

## Command
```bash
python -m src.data.article_type_classifier_v4 \
  --input data/quality/current_entity_event_scores_v4.parquet \
  --output data/quality/current_article_type_scores_v4.parquet \
  --metrics outputs/metrics/current_article_type_scores_v4.json \
  --status outputs/status/05_ARTICLE_TYPE_AND_NOISE_FILTER_V4.status.json
```

## Verification gates
- Output row count equals input row count.
- `article_type` has no nulls.
- Metrics report distribution by split.
