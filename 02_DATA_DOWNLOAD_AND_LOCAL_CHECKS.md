# Step 02 — FNSPID Data Download and Local Checks

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Locate or download FNSPID safely, without loading the full dataset into memory. Produce a file manifest and identify usable news and price files.

## Research basis
FNSPID contains large-scale time-aligned financial news and stock prices for S&P 500 firms over 1999-2023. It is too large for naive full-memory loading, so this step must stream or inspect files lazily.

## Inputs

- `configs/local_paths.yaml` from Step 01.
- Local cache may include `datasets--Zihan1004--FNSPID`.

## Tasks

### 1. Create scripts
Create:

```text
src/data/inspect_fnspid_files.py
src/data/download_fnspid_minimal.py
```

### 2. Inspect local cache first
`inspect_fnspid_files.py` must:

- Read `configs/local_paths.yaml`.
- Search `datasets.fnspid_cache` and common HF cache snapshots.
- Recursively list files with suffix:
  - `.csv`
  - `.parquet`
  - `.jsonl`
  - `.zip`
  - `.gz`
- Save manifest:

```text
data/raw/fnspid_file_manifest.csv
outputs/status/fnspid_file_manifest_summary.json
```

Manifest columns:

```text
path, filename, suffix, size_mb, parent_dir, guessed_role
```

`guessed_role` should be one of:

```text
news | price | metadata | unknown
```

Use filename heuristics: `news`, `external`, `nasdaq`, `nyse`, `history`, `price`, `stock`.

### 3. Download only if not found
If local cache is missing, create `download_fnspid_minimal.py` using `huggingface_hub.snapshot_download`.

Important:

- Do not download every file by default.
- Start with file listing or allow patterns.
- Save to `data/raw/fnspid_hf_snapshot/`.

Command template:

```bash
python src/data/download_fnspid_minimal.py   --repo-id Zihan1004/FNSPID   --repo-type dataset   --output data/raw/fnspid_hf_snapshot   --allow-patterns "*.csv" "*.parquet" "*.zip"   --max-files 20
```

If the hub requires large downloads, stop and write instructions in status JSON.

### 4. Create tiny previews
For the 5 largest likely news/price files, read only first 100 rows and save:

```text
data/raw/previews/<safe_filename>.head100.csv
```

If compressed zip, inspect names inside the zip without extracting all.

## Verification
Run:

```bash
cd firefin
python src/data/inspect_fnspid_files.py --config configs/local_paths.yaml --output data/raw/fnspid_file_manifest.csv
```

Check:

```bash
python - <<'PYCHECK'
import pandas as pd
m = pd.read_csv('data/raw/fnspid_file_manifest.csv')
print(m.head())
print(m['guessed_role'].value_counts(dropna=False))
assert len(m) > 0
PYCHECK
```

## Acceptance criteria
PASS only if:

- `data/raw/fnspid_file_manifest.csv` exists and has at least 1 row.
- At least one file is guessed as `news` or `price`.
- Preview files exist in `data/raw/previews/` unless all files are inaccessible.
- No script attempts to load the full FNSPID dataset into RAM.

## Status file
Create:

```text
outputs/status/02_DATA_DOWNLOAD_AND_LOCAL_CHECKS.status.json
```

Include:

```json
{
  "step": "02_DATA_DOWNLOAD_AND_LOCAL_CHECKS",
  "status": "PASS|FAIL",
  "news_file_candidates": [],
  "price_file_candidates": [],
  "needs_download": false,
  "notes": "..."
}
```
