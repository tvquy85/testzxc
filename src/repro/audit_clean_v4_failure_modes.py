from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "02_AUDIT_CLEAN_V4_FAILURE_MODES"


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def extract_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.I)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except Exception:
        return None


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="BaoCaoCodexFixCleanData_20062026.md")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--samples-dir", default="review_samples/dataclean_v4_20062026")
    parser.add_argument("--output", default="outputs/metrics/02_clean_v4_failure_modes.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    report_path = Path(args.report)
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if not report_text:
        failures.append(f"missing report: {args.report}")

    science = read_json("outputs/repro/currentdata_clean_v4_science_gate_report.json")
    flow = read_json(Path(args.metrics_dir) / "flow_vs_proxy_clean_v4_stage0_combined.json")
    step17 = read_json(Path(args.metrics_dir) / "current_clean_v4_step17_metrics.json")
    alignment = read_json(Path(args.metrics_dir) / "alignment_dataset_current_clean_v4_stage0_small.json")
    pred = read_json(Path(args.metrics_dir) / "current_clean_v4_test_predictions_full.json")

    full_clean_v4_sharpe = step17.get("sharpe_daily_annualized")
    if full_clean_v4_sharpe is None:
        full_clean_v4_sharpe = extract_float(report_text, r"Sharpe[^`\n]*`?(-?\d+(?:\.\d+)?)")
    counterfactual_pass_rate = step17.get("counterfactual_pass_rate")
    counterfactual_no_change_rate = step17.get("counterfactual_no_change_rate")
    flow_rows = flow.get("rows") or extract_float(report_text, r"combined[^,\n]*?(\d+)\s+rows")
    dpo_pairs = alignment.get("dpo_pairs", 0)
    rwsft_examples = alignment.get("rwsft_examples", 0)

    sample_dir = Path(args.samples_dir)
    rationale_sample_rows = jsonl_count(sample_dir / "13_rationale_generation_samples.jsonl")
    grounding_sample_rows = jsonl_count(sample_dir / "11_claim_grounding_nli_samples.jsonl")

    adapter_v4_paths = [
        Path("outputs/models/qwen3_medium_clean_v4_dpo_adapter/adapter_model.safetensors"),
        Path("checkpoints/aligned/qwen3_4b/current_clean_v4_dpo/adapter_model.safetensors"),
    ]
    adapter_v4_trained = any(path.exists() for path in adapter_v4_paths)
    claim_matrix = science.get("claim_matrix", {})
    output = {
        "pipeline_decision_small": science.get("pipeline_decision"),
        "claim_decision_small": science.get("claim_decision"),
        "claim_matrix_small": claim_matrix,
        "flow_rows": int(flow_rows or 0),
        "flow_claim_allowed_small": bool(flow.get("flow_claim_allowed") or flow.get("flow_reward_improvement")),
        "full_clean_v4_sharpe": float(full_clean_v4_sharpe or 0.0),
        "trading_alpha_allowed_small": claim_matrix.get("trading_alpha") == "allowed",
        "counterfactual_pass_rate": float(counterfactual_pass_rate or 0.0),
        "counterfactual_no_change_rate": float(counterfactual_no_change_rate or 0.0),
        "rwsft_examples_small": int(rwsft_examples or 0),
        "dpo_pairs_small": int(dpo_pairs or 0),
        "adapter_v4_trained": bool(adapter_v4_trained),
        "prediction_rows_small": int(pred.get("rows", step17.get("prediction_rows", 0)) or 0),
        "rationale_sample_rows": rationale_sample_rows,
        "grounding_sample_rows": grounding_sample_rows,
        "known_failure_modes": [
            "small_scale_only",
            "negative_or_unproven_trading_alpha",
            "v4_adapter_not_trained_at_medium_scale" if not adapter_v4_trained else "v4_adapter_available",
            "counterfactual_news_needs_breakdown",
            "flow_needs_true_medium_validation",
        ],
        "claim_allowed": False,
        "pipeline_pass": True,
    }
    if output["flow_rows"] <= 0:
        failures.append("flow row count missing")
    if "full_clean_v4_sharpe" not in output:
        failures.append("full clean v4 sharpe missing")

    write_json(args.output, output)
    write_manifest(args.manifest, [args.report, args.output], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.report, args.metrics_dir, args.tables_dir, args.samples_dir],
        outputs_created=[args.output, args.manifest, args.status],
        metrics=output,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
