from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import artifact_entry, write_json, write_manifest, write_status

STEP = "01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4"

DEFAULT_INPUTS = [
    "BaoCaoCodexFixSmallScale_20062026.md",
    "outputs/repro/currentdata_science_gate_report_v2.json",
    "outputs/metrics/flow_vs_proxy_v3_1_eval.json",
    "outputs/metrics/backtest_daily_portfolio_current_v3.json",
    "outputs/metrics/counterfactual_directional_current_v3.json",
    "outputs/metrics/current_v3_claim_matrix.json",
    "data/processed/ticker_date_contexts_h1_v2_targets.parquet",
]


def should_copy(path: Path, max_copy_mb: int) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size <= max_copy_mb * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="currentdata_v3_before_clean_v4")
    parser.add_argument("--output-dir", default="outputs/baseline_freeze/currentdata_v3_before_clean_v4")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--max-copy-mb", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    copied: list[str] = []
    referenced: list[dict[str, Any]] = []
    for source_text in DEFAULT_INPUTS:
        source = Path(source_text)
        if not source.exists():
            failures.append(f"missing input: {source_text}")
            continue
        referenced.append(artifact_entry(source, STEP, args.tag))
        if should_copy(source, args.max_copy_mb):
            target = output_dir / source.name
            shutil.copy2(source, target)
            copied.append(str(target).replace("\\", "/"))

    summary_path = output_dir / "baseline_freeze_summary.json"
    write_json(
        summary_path,
        {
            "tag": args.tag,
            "copied_files": copied,
            "referenced_inputs": referenced,
            "max_copy_mb": args.max_copy_mb,
            "large_files_referenced_not_copied": [
                item["path"] for item in referenced if item["path"] not in [Path(path).name for path in copied]
            ],
        },
    )
    manifest_payload = write_manifest(args.manifest, [*copied, str(summary_path)], STEP, run_id=args.tag, extra={"referenced_inputs": referenced})
    metrics = {
        "tag": args.tag,
        "inputs_checked": len(DEFAULT_INPUTS),
        "copied_files": len(copied),
        "referenced_inputs": len(referenced),
        "manifest_artifacts": len(manifest_payload.get("artifacts", [])),
        "model_weights_copied": False,
    }
    if not referenced:
        failures.append("no baseline inputs were found")
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
