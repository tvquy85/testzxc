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
    status={'step':'01_FREEZE_MEDIUM_BASELINE_AND_TESTS','status':'PASS' if not failures else 'FAIL','inputs_checked':KEY_FILES,'outputs_created':['outputs/repro/v6_freeze_medium_baseline.json'],'metrics':{'file_count':len(report['files'])},'failures':failures,'next_step_allowed':not failures}
    Path('outputs/status/01_FREEZE_MEDIUM_BASELINE_AND_TESTS.status.json').write_text(json.dumps(status,indent=2), encoding='utf-8')
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
