# 01 — Freeze Clean V4 Small Baseline

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Preserve all Clean V4 small-scale artifacts before medium-scale changes.

## Why this is needed
Reviewer-facing work needs an immutable baseline. We must be able to compare medium results to the small V4 report and reproduce old metrics.

## Files to create or modify

Create `src/repro/freeze_clean_v4_small_baseline.py`.

Core code idea:
```python
from pathlib import Path
import hashlib, shutil, json
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
```


## Inputs
```text
BaoCaoCodexFixCleanData_20062026.md
outputs/repro/currentdata_clean_v4_science_gate_report.json
outputs/status/
outputs/metrics/
outputs/tables/
review_samples/dataclean_v4_20062026/
```

## Outputs
```text
outputs/freeze/clean_v4_small_before_medium/
outputs/manifests/01_FREEZE_CLEAN_V4_SMALL_BASELINE.manifest.json
outputs/status/01_FREEZE_CLEAN_V4_SMALL_BASELINE.status.json
```

## Commands
```bash
python -m src.repro.freeze_clean_v4_small_baseline \
  --output-dir outputs/freeze/clean_v4_small_before_medium \
  --manifest outputs/manifests/01_FREEZE_CLEAN_V4_SMALL_BASELINE.manifest.json \
  --status outputs/status/01_FREEZE_CLEAN_V4_SMALL_BASELINE.status.json
```

## Verification
```bash
python - <<'PY'
import json, pathlib
s=json.load(open('outputs/status/01_FREEZE_CLEAN_V4_SMALL_BASELINE.status.json'))
assert s['status']=='PASS'
assert pathlib.Path('outputs/freeze/clean_v4_small_before_medium').exists()
print('PASS freeze')
PY
python -m pytest -q tests
```

## Acceptance criteria
- Freeze folder exists.
- Manifest contains checksums.
- No large model weights copied by default.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "01_FREEZE_CLEAN_V4_SMALL_BASELINE",
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
