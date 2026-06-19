# Step 04 — Align News with Prices and Build Forecast Labels

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Create supervised samples where each news item is aligned to a prediction timestamp, prior price window, and future abnormal return label.

## Inputs

```text
data/processed/news_mvp.parquet
data/processed/prices_mvp.parquet
```

## Output

```text
data/labels/aligned_samples_h1.parquet
data/labels/label_distribution_h1.csv
outputs/status/04_PRICE_NEWS_ALIGNMENT_AND_LABELS.status.json
```

## Tasks

### 1. Create script
Create:

```text
src/data/build_aligned_labels.py
```

### 2. Define event date
For each news item:

- Convert `timestamp_utc` to date.
- If timestamp is before market close and date is a trading day, event date = same trading day.
- If timestamp is after market close or on non-trading day, event date = next trading day.

If exact intraday timestamp quality is uncertain, use conservative rule:

```text
event_date = next trading day after news timestamp date
```

Record which rule was used.

### 3. Build prior window
For each sample, require at least 60 prior trading days of price data ending at `event_date - 1 trading day`.

Save prior window compactly as feature rows later; for now store:

```text
window_start_date, window_end_date, event_date
```

### 4. Compute future return
For horizon `h=1` trading day:

```text
stock_return_h1 = adj_close[event_date + 1] / adj_close[event_date] - 1
market_return_h1 = mean cross-sectional return across MVP tickers for same horizon
abnormal_return_h1 = stock_return_h1 - market_return_h1
```

If SPY exists in data, prefer SPY; otherwise use equal-weight MVP market proxy.

### 5. Build 5-class label
Use abnormal return:

```text
Strong Down: <= -0.03
Mild Down: > -0.03 and <= -0.0075
Neutral: > -0.0075 and < 0.0075
Mild Up: >= 0.0075 and < 0.03
Strong Up: >= 0.03
```

Columns:

```text
sample_id, ticker, timestamp_utc, event_date, horizon, headline, body,
stock_return_h1, market_return_h1, abnormal_return_h1, label_5,
window_start_date, window_end_date
```

### 6. Add leakage flags
Create boolean columns:

```text
leakage_phrase_flag
post_market_uncertain_flag
```

`leakage_phrase_flag = true` if headline/body contains phrases like:

```text
shares surged, shares plunged, stock rises after, stock falls after, trading higher after, trading lower after
```

Do not drop them yet; just flag.

## Verification
Run:

```bash
cd firefin
python src/data/build_aligned_labels.py \
  --news data/processed/news_mvp.parquet \
  --prices data/processed/prices_mvp.parquet \
  --horizon 1 \
  --output data/labels/aligned_samples_h1.parquet
```

Then:

```bash
python - <<'PYCHECK'
import pandas as pd
s = pd.read_parquet('data/labels/aligned_samples_h1.parquet')
print(s.shape)
print(s['label_5'].value_counts(normalize=True))
assert s['sample_id'].is_unique
assert s['abnormal_return_h1'].notna().mean() > 0.95
assert s['label_5'].nunique() >= 3
PYCHECK
```

## Acceptance criteria
PASS only if:

- Output parquet exists.
- At least 5,000 aligned samples for MVP, or status explains why fewer.
- At least 3 label classes appear.
- `label_distribution_h1.csv` exists.
- Leakage flags are created.

## Status file
Create:

```json
{
  "step": "04_PRICE_NEWS_ALIGNMENT_AND_LABELS",
  "status": "PASS|FAIL",
  "aligned_rows": 0,
  "label_distribution": {},
  "leakage_flag_rate": 0.0,
  "date_min": "...",
  "date_max": "...",
  "notes": "..."
}
```
