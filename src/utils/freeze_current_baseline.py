import argparse, hashlib, json, shutil, time
from pathlib import Path

from src.utils.artifacts import write_manifest, write_status

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/baseline_freeze")
    ap.add_argument("--manifest", default="outputs/manifests/01_FREEZE_BASELINE_CURRENTDATA.manifest.json")
    ap.add_argument("--status", default="outputs/status/01_FREEZE_BASELINE_CURRENTDATA.status.json")
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
    write_manifest(args.manifest, [str(out), str(out / "manifest.json")], "01_FREEZE_BASELINE_CURRENTDATA")
    write_status(
        args.status,
        "01_FREEZE_BASELINE_CURRENTDATA",
        "PASS",
        inputs_checked=src_dirs,
        outputs_created=[str(out), str(out / "manifest.json"), args.manifest, args.status],
        metrics={"file_count": len(rows), "freeze_dir": str(out)},
        failures=[],
        next_step_allowed=True,
    )

if __name__ == "__main__":
    main()
