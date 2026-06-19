from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import sha256_file, utc_now_iso, write_json, write_manifest, write_status


STEP = "01_REPO_AUDIT_AND_SAFE_BRANCH"
DEFAULT_STATUS = f"outputs/status/{STEP}.status.json"
DEFAULT_HASHES = "outputs/audit/repo_file_hashes_initial.csv"
DEFAULT_MANIFEST = f"outputs/manifests/{STEP}.manifest.json"


KEY_OUTPUTS = [
    "outputs/status",
    "outputs/metrics",
    "outputs/tables",
    "data/labels/aligned_samples_h1.parquet",
    "data/processed/split_h1.parquet",
    "data/rationales/candidate_rationales_h1.jsonl",
    "outputs/tables/table_1_prediction_main.csv",
    "outputs/tables/table_4_ablation.csv",
]


RISK_PATTERNS = {
    "hard_coded_windows_path": re.compile(r"(?i)(e:/|e:\\|c:\\users|d:\\)"),
    "dummy_table": re.compile(r"(?i)(dummy|placeholder|zero-filled|zero filled)"),
    "auto_fix_distribution": re.compile(r"(?i)(auto-fix|auto fix|needs_fix|No rationale provided)"),
    "neutral_fallback": re.compile(r"(?i)(default neutral|neutral.*fallback|dist\\[\"neutral\"\\]\\s*=\\s*1\\.0)"),
    "per_news_backtest": re.compile(r"(?i)(per-news|per news|news item|sample return|mean\\(\\))"),
}


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def tracked_files() -> list[str]:
    try:
        files = run_git(["ls-files", "src", "configs", "prompts"]).splitlines()
    except Exception:
        files = []
    return [f for f in files if Path(f).is_file()]


def write_hash_csv(path: str, files: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sha256", "size_bytes"])
        writer.writeheader()
        for file_path in files:
            p = Path(file_path)
            writer.writerow(
                {
                    "path": file_path,
                    "sha256": sha256_file(p),
                    "size_bytes": p.stat().st_size,
                }
            )


def scan_text_risks(files: list[str]) -> dict[str, list[dict[str, Any]]]:
    risks = {name: [] for name in RISK_PATTERNS}
    for file_path in files:
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for name, pattern in RISK_PATTERNS.items():
            matches = list(pattern.finditer(text))
            if matches:
                risks[name].append({"path": file_path, "count": len(matches)})
    return risks


def scan_status_risks() -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for path in sorted(Path("outputs/status").glob("*.json")) if Path("outputs/status").exists() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            risks.append({"path": str(path), "error": f"parse_error: {exc}"})
            continue
        if data.get("status") == "PASS":
            outputs = data.get("outputs_created", [])
            missing = [out for out in outputs if not Path(out).exists()]
            if missing:
                risks.append({"path": str(path), "missing_outputs": missing})
    return risks


def parse_tonghop() -> dict[str, Any] | None:
    path = Path("TongHop.md")
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "num_lines": text.count("\n") + 1,
        "mentions_pass": len(re.findall(r"\\bPASS\\b", text)),
        "mentions_fail": len(re.findall(r"\\bFAIL\\b", text)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/audit/repo_audit_initial.json")
    parser.add_argument("--hashes", default=DEFAULT_HASHES)
    parser.add_argument("--status", default=DEFAULT_STATUS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    files = tracked_files()
    write_hash_csv(args.hashes, files)
    risk_flags = scan_text_risks(files)
    status_risks = scan_status_risks()
    try:
        branch = run_git(["branch", "--show-current"])
        head = run_git(["rev-parse", "HEAD"])
        dirty = bool(run_git(["status", "--porcelain"]))
    except Exception:
        branch = None
        head = None
        dirty = None

    key_outputs = {
        path: {"exists": Path(path).exists(), "is_file": Path(path).is_file(), "is_dir": Path(path).is_dir()}
        for path in KEY_OUTPUTS
    }
    audit = {
        "step": STEP,
        "created_at": utc_now_iso(),
        "git": {"branch": branch, "head": head, "dirty_worktree": dirty},
        "tracked_file_count": len(files),
        "tracked_files": files,
        "key_outputs": key_outputs,
        "tonghop": parse_tonghop(),
        "risk_flags": risk_flags,
        "status_pass_missing_outputs": status_risks,
    }
    write_json(args.output, audit)
    write_manifest(args.manifest, [args.output, args.hashes], STEP)

    outputs = [args.output, args.hashes, args.manifest, args.status]
    write_status(
        args.status,
        step=STEP,
        status="PASS",
        inputs_checked=files + ["outputs/status/*.json", "outputs/metrics/*.json", "outputs/tables/*.csv"],
        outputs_created=outputs,
        metrics={
            "tracked_file_count": len(files),
            "risk_flag_count": sum(len(v) for v in risk_flags.values()) + len(status_risks),
            "branch": branch,
            "dirty_worktree": dirty,
        },
        failures=[],
        next_step_allowed=True,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
