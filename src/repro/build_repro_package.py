from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "21_FINAL_REPRO_PACKAGE_AND_AAAI_GATE"
INCLUDE_ROOTS = [
    "configs",
    "prompts",
    "src",
    "tests",
    "outputs/tables/final",
    "outputs/metrics",
    "outputs/manifests",
]
EXCLUDE_SUFFIXES = {".safetensors", ".pt", ".bin", ".parquet", ".zip"}
MAX_FILE_BYTES = 50_000_000


def load_gate(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"decision": "NO_GO", "blocking_issues": [f"gate report missing: {path}"], "warnings": []}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"decision": "NO_GO", "blocking_issues": [f"invalid gate report: {path}"], "warnings": []}


def write_readme(path: str, gate_report: dict[str, Any]) -> None:
    decision = gate_report.get("decision", "NO_GO")
    blockers = gate_report.get("blocking_issues", [])
    blocker_text = "\n".join(f"- {item}" for item in blockers) if blockers else "- none"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        f"""# FIRE-Fin Reproducibility Package

This package contains scripts, configs, prompts, tests, metrics, manifests, and final table artifacts for the FIRE-Fin AAAI upgrade audit.

Current gate decision: `{decision}`.

Blocking issues:

{blocker_text}

One-command smoke test from repo root:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest -q tests
```

Expected smoke output for this snapshot:

```text
25 passed
```

Gate check:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe src/repro/aaai_gate_check.py --output outputs/repro/aaai_gate_report.json
```

Expected gate output is `GO` only when all status files pass and the final blocker checks pass.

Large raw datasets and model weights are intentionally excluded. Use the scripts and local `$HF_HOME` configuration to recreate data/model-dependent artifacts.
""",
        encoding="utf-8",
    )


def write_checklist(path: str, gate_report: dict[str, Any]) -> None:
    blockers = gate_report.get("blocking_issues", [])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "# Reproducibility Checklist Draft\n\n"
        f"- Gate decision: `{gate_report.get('decision', 'NO_GO')}`\n"
        f"- Blocking issue count: {len(blockers)}\n"
        "- Raw datasets excluded: yes\n"
        "- Model weights excluded: yes\n"
        "- Configs included: yes\n"
        "- Prompts included: yes\n"
        "- Source and tests included: yes\n"
        "- Final tables, metrics, manifests included: yes\n",
        encoding="utf-8",
    )


def should_include(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    if path.stat().st_size > MAX_FILE_BYTES:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/repro/firefin_repro_package.zip")
    parser.add_argument("--gate-report", default="outputs/repro/aaai_gate_report.json")
    parser.add_argument("--readme", default="outputs/repro/README_REPRODUCE.md")
    parser.add_argument("--checklist", default="outputs/repro/REPRODUCIBILITY_CHECKLIST_DRAFT.md")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    gate_report = load_gate(args.gate_report)
    write_readme(args.readme, gate_report)
    write_checklist(args.checklist, gate_report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    included = [args.readme, args.checklist, args.gate_report]
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_name in included:
            if Path(file_name).exists():
                zf.write(file_name, Path(file_name).name)
        for root in INCLUDE_ROOTS:
            base = Path(root)
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if should_include(path):
                    zf.write(path, str(path).replace("\\", "/"))
                    included.append(str(path))
    outputs = [args.output, args.readme, args.checklist, args.gate_report, args.manifest, args.status]
    write_manifest(args.manifest, [args.output, args.readme, args.checklist, args.gate_report], STEP)
    blockers = list(gate_report.get("blocking_issues", []))
    if not Path(args.output).exists():
        blockers.append(f"package missing after build: {args.output}")
    metrics = {
        "decision": gate_report.get("decision", "NO_GO"),
        "blocker_count": len(blockers),
        "included_file_count": len(set(included)),
        "package_size_bytes": Path(args.output).stat().st_size if Path(args.output).exists() else 0,
    }
    status = "PASS" if not blockers else "FAIL"
    write_status(args.status, STEP, status, [args.gate_report, "configs", "prompts", "src", "tests", "outputs/tables/final", "outputs/metrics", "outputs/manifests"], outputs, metrics, blockers, status == "PASS")
    print(json.dumps({"status": status, "metrics": metrics, "failures": blockers, "output": args.output}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
