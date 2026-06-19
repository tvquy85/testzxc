# Step 03 — FNSPID Schema Inspection and MVP Sample Build

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Create a small, clean MVP sample from FNSPID without solving labels yet. Output normalized news and price tables for 30-50 tickers.

## Inputs

- `data/raw/fnspid_file_manifest.csv`
- Preview files from Step 02.

## Tasks

### 1. Create scripts
Create:

```text
src/data/infer_fnspid_schema.py
src/data/build_mvp_unlabeled_sample.py
src/data/normalize_columns.py
```

### 2. Infer schema
`infer_fnspid_schema.py` must inspect preview files and infer column roles.

Target normalized news schema:

```text
news_id, ticker, timestamp_utc, headline, body, source, url, author, raw_file
```

If `body` is missing, set `body = headline`.

Target normalized price schema:

```text
ticker, date, open, high, low, close, adj_close, volume, raw_file
```

If `adj_close` is missing, set `adj_close = close` and record warning.

### 3. Select MVP tickers
Choose 30-50 tickers that have both news and price. Prefer high-coverage tickers.

Save:

```text
data/samples/mvp_tickers.txt
```

If no ticker overlap can be detected automatically, start with this fallback list and keep only available ones:

```text
AAPL,MSFT,AMZN,GOOGL,META,NVDA,TSLA,JPM,BAC,WMT,DIS,NFLX,INTC,AMD,IBM,ORCL,CSCO,CRM,BA,GE,XOM,CVX,PFE,MRK,JNJ,KO,PEP,NKE,MCD,HD,COST,V,MA,ADBE,QCOM,AVGO,TXN,UNH,PG
```

### 4. Build normalized sample
For selected tickers and years 2018-2023 if available, create:

```text
data/processed/news_mvp.parquet
data/processed/prices_mvp.parquet
```

If 2018-2023 is unavailable, use the latest continuous 3-year range available and record this in status.

### 5. Basic cleaning
News cleaning rules:

- Drop rows with empty ticker or timestamp.
- Drop exact duplicate `(ticker, timestamp_utc, headline)`.
- Strip HTML if present.
- Keep original text in `headline_raw` if cleaning modifies it.

Price cleaning rules:

- Sort by `ticker, date`.
- Drop rows with invalid close/volume.
- Enforce date type.

## Verification
Run:

```bash
cd firefin
python src/data/infer_fnspid_schema.py --manifest data/raw/fnspid_file_manifest.csv --out outputs/status/fnspid_schema_report.json
python src/data/build_mvp_unlabeled_sample.py --config configs/local_paths.yaml --tickers-out data/samples/mvp_tickers.txt --limit-tickers 50
```

Then run:

```bash
python - <<'PYCHECK'
import pandas as pd
news = pd.read_parquet('data/processed/news_mvp.parquet')
prices = pd.read_parquet('data/processed/prices_mvp.parquet')
print(news.shape, prices.shape)
print(news.columns.tolist())
print(prices.columns.tolist())
assert news['ticker'].nunique() >= 10
assert prices['ticker'].nunique() >= 10
assert news['headline'].notna().mean() > 0.95
PYCHECK
```

## Acceptance criteria
PASS only if:

- `news_mvp.parquet` and `prices_mvp.parquet` exist.
- At least 10 tickers have both news and price in MVP.
- News has timestamp and ticker.
- Prices have OHLCV and date.
- Row counts are saved.

## Status file
Create:

```text
outputs/status/03_FNSPID_SCHEMA_AND_SAMPLE_BUILD.status.json
```

Include:

```json
{
  "step": "03_FNSPID_SCHEMA_AND_SAMPLE_BUILD",
  "status": "PASS|FAIL",
  "news_rows": 0,
  "price_rows": 0,
  "ticker_count": 0,
  "date_min": "...",
  "date_max": "...",
  "warnings": []
}
```
