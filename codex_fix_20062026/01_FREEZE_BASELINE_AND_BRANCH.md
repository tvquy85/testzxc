# 01 — Freeze Baseline and Create Safe Branch

> Scope: upgrade `tvquy85/testzxc` branch `upgrade-aaai-reproducibility` on the **current already-built dataset/artifacts first**. Do not expand to full FNSPID until the current-data gates pass.
> Codex rule: implement only this task, run verification, write/update status JSON, and do not silently PASS if required outputs are missing.


## Goal
Freeze all current artifacts before code changes.

## Tasks
1. Run:
```bash
git checkout upgrade-aaai-reproducibility
git checkout -b currentdata-aaai-fix-v2
```

2. Create `src/utils/freeze_current_baseline.py`.

## Exact script core
```python
from pathlib import Path
import argparse, hashlib, json, shutil, time

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/baseline_freeze")
    args = ap.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out_root) / f"currentdata_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    src_dirs = ["outputs/metrics","outputs/status","outputs/tables","outputs/repro","outputs/data_samples"]
    rows = []
    for src in src_dirs:
        p = Path(src)
        if not p.exists():
            continue
        dst = out / src
        shutil.copytree(p, dst)
        for fp in dst.rglob("*"):
            if fp.is_file():
                rows.append({"path": str(fp), "bytes": fp.stat().st_size, "sha256": sha256_file(fp)})
    (out / "manifest.json").write_text(json.dumps({"frozen_at": ts, "files": rows}, indent=2), encoding="utf-8")
    Path("outputs/status").mkdir(parents=True, exist_ok=True)
    status = {"step":"01_FREEZE_BASELINE_CURRENTDATA","status":"PASS","outputs_created":[str(out)],"metrics":{"file_count":len(rows)}}
    Path("outputs/status/01_FREEZE_BASELINE_CURRENTDATA.status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()
```

## Run
```bash
python -m src.utils.freeze_current_baseline
```

## Verify
```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path("outputs/status/01_FREEZE_BASELINE_CURRENTDATA.status.json")
s=json.loads(p.read_text())
assert s["status"]=="PASS"
assert s["metrics"]["file_count"]>0
print(s)
PY
```

## Acceptance
- Baseline folder exists.
- Manifest exists.
- At least 10 files frozen.
