# 01 — Freeze Clean V4 Medium Baseline and Current Tests

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Freeze the current medium result before V6 changes. This prevents accidental claim inflation and makes before/after comparisons auditable.

## Inputs
```text
BaoCaoclean_v4_medium_21062026.md
review_samples/clean_v4_medium_21062026/24_science_gate_report.json
outputs/metrics/*.json
outputs/tables/*.csv
outputs/status/*.json
```

## Outputs
```text
outputs/repro/v6_freeze_medium_baseline.json
outputs/status/01_FREEZE_MEDIUM_BASELINE_AND_TESTS.status.json
```

## Implementation
Create `src/repro/freeze_v6_medium_baseline.py`:
```python
from pathlib import Path
import json, hashlib
KEY_FILES = [
 'BaoCaoclean_v4_medium_21062026.md',
 'review_samples/clean_v4_medium_21062026/24_science_gate_report.json',
 'outputs/tables/medium_ablation_results.csv',
 'outputs/tables/medium_baseline_comparison.csv',
]
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def main():
    failures=[p for p in KEY_FILES if not Path(p).exists()]
    report={'files':[], 'failures':failures}
    for p in KEY_FILES:
        if Path(p).exists(): report['files'].append({'path':p,'sha256':sha256(p),'bytes':Path(p).stat().st_size})
    Path('outputs/repro').mkdir(parents=True, exist_ok=True); Path('outputs/status').mkdir(parents=True, exist_ok=True)
    Path('outputs/repro/v6_freeze_medium_baseline.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    status={'step':'01_FREEZE_MEDIUM_BASELINE_AND_TESTS','status':'PASS' if not failures else 'FAIL','inputs':KEY_FILES,'outputs':['outputs/repro/v6_freeze_medium_baseline.json'],'metrics':{'file_count':len(report['files'])},'failures':failures,'next_step_allowed':not failures}
    Path('outputs/status/01_FREEZE_MEDIUM_BASELINE_AND_TESTS.status.json').write_text(json.dumps(status,indent=2), encoding='utf-8')
    if failures: raise SystemExit(1)
if __name__=='__main__': main()
```

## Commands
```bash
python -m src.repro.freeze_v6_medium_baseline
python -m src.utils.verify_status outputs/status/01_FREEZE_MEDIUM_BASELINE_AND_TESTS.status.json
python -m pytest -q tests
```

## Test case
Create `tests/test_v6_freeze_baseline.py`:
```python
import json
from pathlib import Path

def test_v6_freeze_report_exists():
    p=Path('outputs/repro/v6_freeze_medium_baseline.json')
    assert p.exists()
    data=json.loads(p.read_text(encoding='utf-8'))
    assert not data['failures']
    assert len(data['files'])>=3
```
