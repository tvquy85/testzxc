from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import artifact_entry, write_json, write_manifest, write_status

STEP = "01_FREEZE_CLEAN_V4_SMALL_BASELINE"

DEFAULT_INPUTS = [
    "BaoCaoCodexFixCleanData_20062026.md",
    "outputs/repro/currentdata_clean_v4_science_gate_report.json",
    "outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json",
    "outputs/metrics/flow_vs_proxy_clean_v4_stage0_combined.json",
    "outputs/metrics/current_clean_v4_step17_metrics.json",
    "outputs/tables/ablation_current_clean_v4.csv",
    "review_samples/dataclean_v4_20062026/README.md",
    "review_samples/dataclean_v4_20062026/sample_manifest.json",
]


def should_copy(path: Path, max_copy_mb: int) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size <= max_copy_mb * 1024 * 1024


def copy_tree_small_files(source_dir: Path, target_dir: Path, max_copy_mb: int) -> list[str]:
    copied: list[str] = []
    if not source_dir.exists():
        return copied
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        if not should_copy(source, max_copy_mb):
            continue
        target = target_dir / source.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(target).replace("\\", "/"))
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/freeze/clean_v4_small_before_medium")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--max-copy-mb", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    referenced: list[dict[str, Any]] = []
    copied: list[str] = []

    for source_text in DEFAULT_INPUTS:
        source = Path(source_text)
        if not source.exists():
            failures.append(f"missing input: {source_text}")
            continue
        referenced.append(artifact_entry(source, STEP, "clean_v4_small_before_medium"))
        if should_copy(source, args.max_copy_mb):
            target = output_dir / source_text.replace("/", "__").replace("\\", "__")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(target).replace("\\", "/"))

    copied.extend(
        copy_tree_small_files(
            Path("review_samples/dataclean_v4_20062026"),
            output_dir / "review_samples__dataclean_v4_20062026",
            args.max_copy_mb,
        )
    )

    summary_path = output_dir / "freeze_summary.json"
    summary = {
        "step": STEP,
        "copied_files": copied,
        "referenced_inputs": referenced,
        "max_copy_mb": args.max_copy_mb,
        "model_weights_copied": False,
    }
    write_json(summary_path, summary)
    write_manifest(args.manifest, [*copied, str(summary_path)], STEP, run_id="clean_v4_small_before_medium", extra={"referenced_inputs": referenced})

    metrics = {
        "pipeline_pass": not failures,
        "claim_allowed": False,
        "copied_files": len(copied),
        "referenced_inputs": len(referenced),
        "model_weights_copied": False,
    }
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=DEFAULT_INPUTS,
        outputs_created=[str(summary_path), args.manifest, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
