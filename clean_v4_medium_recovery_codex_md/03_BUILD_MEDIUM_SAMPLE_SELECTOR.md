# 03 — Build Medium Sample Selector

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Select a stratified medium subset from existing Clean V4 evidence contexts.

## Why this is needed
Small V4 used too few samples. Medium validation needs enough contexts for judge/flow/alignment without expanding full FNSPID.

## Files to create or modify

Create `src/data/select_medium_clean_v4_samples.py`.

Core stratified sampling logic:
```python
def stratified_sample(df, split, n, seed):
    part=df[df['split'].eq(split)].copy()
    keys=[c for c in ['label_5','track','news_reasoning_track'] if c in part.columns]
    if not keys:
        return part.sample(min(n,len(part)), random_state=seed)
    out=[]
    per=max(1, n // max(1, part.groupby(keys).ngroups))
    for _,g in part.groupby(keys, dropna=False):
        out.append(g.sample(min(per,len(g)), random_state=seed))
    res=pd.concat(out).drop_duplicates('sample_id')
    if len(res)<n:
        extra=part[~part.sample_id.isin(res.sample_id)].sample(min(n-len(res), len(part)-len(res)), random_state=seed)
        res=pd.concat([res,extra])
    return res.head(n)
```


## Inputs
```text
data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet
```

## Outputs
```text
data/processed/medium_clean_v4_sample_ids.parquet
data/processed/medium_clean_v4_contexts.parquet
outputs/metrics/03_medium_sample_selector.json
outputs/status/03_BUILD_MEDIUM_SAMPLE_SELECTOR.status.json
```

## Commands
```bash
python -m src.data.select_medium_clean_v4_samples \
  --contexts data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet \
  --train-n 1000 --val-n 300 --test-n 300 \
  --seed 42 \
  --output data/processed/medium_clean_v4_contexts.parquet \
  --ids-output data/processed/medium_clean_v4_sample_ids.parquet \
  --metrics outputs/metrics/03_medium_sample_selector.json \
  --status outputs/status/03_BUILD_MEDIUM_SAMPLE_SELECTOR.status.json
```

## Verification
```bash
python - <<'PY'
import pandas as pd, json
x=pd.read_parquet('data/processed/medium_clean_v4_contexts.parquet')
assert x.sample_id.is_unique
assert {'train','val','test'}.issubset(set(x.split.unique()))
assert len(x[x.split=='train']) >= 500
print('PASS medium selector', len(x))
PY
```

## Acceptance criteria
- No duplicate `sample_id`.
- Train/val/test distributions are reported.
- Uses only current Clean V4 data; no external dataset.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "03_BUILD_MEDIUM_SAMPLE_SELECTOR",
  "status": "PASS|FAIL",
  "pipeline_pass": true,
  "claim_allowed": false,
  "inputs": [],
  "outputs": [],
  "metrics": {},
  "failures": [],
  "warnings": []
}
```
