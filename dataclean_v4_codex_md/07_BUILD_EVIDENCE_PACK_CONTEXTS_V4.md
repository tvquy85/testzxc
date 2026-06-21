# 07 — Build Ticker-Date Evidence Pack Contexts V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Replace raw concatenated headlines/body with compact evidence packs. Later rationales must cite `N1/N2/M1/T1` IDs.

## Inputs
- `data/processed/current_deduped_news_v4.parquet`
- `data/processed/ticker_date_contexts_h1_v2_targets.parquet`
- `data/indicators/technical_event_tokens_h1_v2.parquet`

## Outputs
- `data/processed/ticker_date_evidence_contexts_h1_v4.parquet`
- `outputs/metrics/ticker_date_evidence_contexts_h1_v4.json`
- `outputs/status/07_TICKER_DATE_EVIDENCE_PACK_V4.status.json`

## Required schema
`sample_id`, `ticker`, `event_date`, `split`, `horizon`, `label_5`, `abnormal_return_h1`, `evidence_pack_json`, `technical_event_tokens_json`, `clean_context_text`, `num_company_event_evidence`, `num_context_only_evidence`, `mean_evidence_quality_score`, `has_company_event_news`, `no_news_context_flag`.

## Evidence selection
- At most 3 A/B company evidence items: `N1`, `N2`, `N3`.
- At most 2 C context-only items: `M1`, `M2`.
- Exclude D_noise.
- `body_excerpt` max 600 chars.

## Command
```bash
python -m src.data.build_ticker_date_evidence_pack_v4 \
  --deduped-news data/processed/current_deduped_news_v4.parquet \
  --base-contexts data/processed/ticker_date_contexts_h1_v2_targets.parquet \
  --tokens data/indicators/technical_event_tokens_h1_v2.parquet \
  --output data/processed/ticker_date_evidence_contexts_h1_v4.parquet \
  --metrics outputs/metrics/ticker_date_evidence_contexts_h1_v4.json \
  --status outputs/status/07_TICKER_DATE_EVIDENCE_PACK_V4.status.json
```

## Verification gates
- No duplicate `sample_id`.
- `evidence_pack_json` parse rate >= 0.99.
- Mean evidence per context <= 5.
- At least 1,000 train contexts with `has_company_event_news=True`, otherwise FAIL with `data_insufficient`.
