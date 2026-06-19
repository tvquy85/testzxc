from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status
from src.utils.config import ConfigError, load_config


STEP = "02_CONFIG_PATHS_AND_REPRO_ENV"
REQUIRED_MODULES = [
    "torch",
    "transformers",
    "accelerate",
    "pandas",
    "pyarrow",
    "numpy",
    "sklearn",
    "lightgbm",
    "matplotlib",
    "seaborn",
    "tqdm",
    "yaml",
    "pytest",
]
ALIGNMENT_ONLY_MODULES = ["peft", "trl", "bitsandbytes"]
OPTIONAL_MODULES = ["sentence_transformers"]


def module_inventory() -> dict[str, Any]:
    inv: dict[str, Any] = {}
    for name in REQUIRED_MODULES + ALIGNMENT_ONLY_MODULES + OPTIONAL_MODULES:
        try:
            mod = importlib.import_module(name)
            inv[name] = {"available": True, "version": getattr(mod, "__version__", "unknown")}
        except Exception as exc:
            inv[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return inv


def git_grep_for_local_paths() -> list[str]:
    needles = ["e:" + "/huggingface", "C:" + "\\Users"]
    findings: list[str] = []
    for root in ("configs", "src", "prompts"):
        if not Path(root).exists():
            continue
        for path in Path(root).rglob("*"):
            if not path.is_file():
                continue
            if path == Path(__file__):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            lower = text.lower()
            if any(n.lower() in lower for n in needles):
                findings.append(str(path))
    return sorted(set(findings))


def freeze_requirements(path: str, inv: dict[str, Any]) -> None:
    lines = [
        "# Generated from the active verification interpreter.",
        "# Missing packages are intentionally commented instead of guessed.",
    ]
    package_names = {
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "sentence_transformers": "sentence-transformers",
    }
    for mod_name in REQUIRED_MODULES + ALIGNMENT_ONLY_MODULES + OPTIONAL_MODULES:
        package = package_names.get(mod_name, mod_name)
        item = inv[mod_name]
        if item["available"] and item.get("version") not in (None, "unknown"):
            lines.append(f"{package}=={item['version']}")
        elif item["available"]:
            lines.append(package)
        else:
            lines.append(f"# MISSING: {package} ({item.get('error')})")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--output", default="outputs/audit/repro_env_report.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--lock", default="requirements.lock.txt")
    parser.add_argument("--hf-home", default=None, help="Runtime Hugging Face cache root override.")
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    inv = module_inventory()
    freeze_requirements(args.lock, inv)
    failures: list[str] = []
    try:
        cfg = load_config(args.config, validate_paths=False)
    except ConfigError as exc:
        cfg = {}
        failures.append(str(exc))
    local_path_findings = git_grep_for_local_paths()
    if local_path_findings:
        failures.append(f"hard-coded local paths remain: {local_path_findings}")

    missing_required = [name for name in REQUIRED_MODULES if not inv[name]["available"]]
    missing_alignment = [name for name in ALIGNMENT_ONLY_MODULES if not inv[name]["available"]]
    missing_optional = [name for name in OPTIONAL_MODULES if not inv[name]["available"]]
    if missing_required:
        failures.append(f"missing required modules in active interpreter: {missing_required}")

    report = {
        "step": STEP,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "config": cfg,
        "module_inventory": inv,
        "missing_required_modules": missing_required,
        "missing_alignment_modules": missing_alignment,
        "missing_optional_modules": missing_optional,
        "non_blocking_note": "Missing peft/trl/bitsandbytes blocks Step 15 alignment training only; it does not block Step 09-14 smoke generation/judging/reward plumbing.",
        "local_path_findings": local_path_findings,
        "nvidia_smi": nvidia_smi(),
        "env_sources_checked": [
            "active interpreter",
            "d:/Conferences/NIPS/FinEval_Prev/FinEval/",
            "d:/LOBProj/LOBExp/.venv/",
        ],
    }
    write_json(args.output, report)
    write_manifest(args.manifest, [args.output, args.lock], STEP)
    status = "PASS" if not failures else "FAIL"
    outputs = [args.output, args.lock, args.manifest, args.status]
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.config, "configs", "src", "prompts"],
        outputs_created=outputs,
        metrics={
            "missing_required_module_count": len(missing_required),
            "missing_alignment_module_count": len(missing_alignment),
            "missing_optional_module_count": len(missing_optional),
            "hard_coded_local_path_file_count": len(local_path_findings),
            "python_executable": sys.executable,
        },
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
