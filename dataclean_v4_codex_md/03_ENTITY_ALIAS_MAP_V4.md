# 03 — Build Ticker Alias Map V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Create aliases so later scripts can check whether a news row is about the target ticker/company.

## Inputs
- `data/quality/current_data_quality_v2.parquet`
- optional columns: `ticker`, `company`, `company_name`, `name`

## Outputs
- `data/quality/ticker_alias_map_v4.json`
- `outputs/metrics/ticker_alias_map_v4.json`
- `outputs/status/03_ENTITY_ALIAS_MAP_V4.status.json`

## Implementation
Create `src/data/build_ticker_alias_map_v4.py`.

```python
def normalize_company_name(name: str) -> list[str]:
    import re
    if not name or str(name).lower() == 'nan':
        return []
    raw = str(name).strip()
    base = re.sub(r'\b(inc|inc\.|corp|corp\.|corporation|co|co\.|ltd|plc|class a|class b)\b', '', raw, flags=re.I)
    base = re.sub(r'\s+', ' ', base).strip(' ,.-')
    out = [raw]
    if base and base.lower() != raw.lower():
        out.append(base)
    return sorted(set(out), key=len, reverse=True)
```

If company names are unavailable, each ticker must still map to `[ticker]`.

## Command
```bash
python -m src.data.build_ticker_alias_map_v4 \
  --input data/quality/current_data_quality_v2.parquet \
  --output data/quality/ticker_alias_map_v4.json \
  --metrics outputs/metrics/ticker_alias_map_v4.json \
  --status outputs/status/03_ENTITY_ALIAS_MAP_V4.status.json
```

## Verification gates
- Valid JSON.
- Ticker coverage >= 95% of unique tickers in current data.
- Every ticker has at least one alias.
