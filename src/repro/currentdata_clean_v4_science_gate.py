from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "18_SCIENCE_GATE_AND_RUNBOOK_V4"


def read_json(path: str | Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest_status(status_dir: str, prefix: str) -> dict[str, Any]:
    candidates = sorted(glob.glob(str(Path(status_dir) / f"{prefix}*.status.json")))
    if not candidates:
        return {"path": None, "payload": None}
    def priority(path: str) -> tuple[int, float, str]:
        name = Path(path).name
        score = 0
        if "STAGE0_COMBINED" in name:
            score += 100
        if "STAGE0_SMALL" in name:
            score += 50
        if "_SMALL" in name:
            score += 25
        if "DATASET" in name or "TRAIN" in name:
            score -= 20
        return score, Path(path).stat().st_mtime, path

    path = max(candidates, key=priority)
    return {"path": path, "payload": read_json(path)}


def status_pass(item: dict[str, Any]) -> bool:
    payload = item.get("payload") or {}
    return payload.get("status") == "PASS" and payload.get("next_step_allowed") is True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-dir", default="outputs/status")
    parser.add_argument("--flow-metrics", default="outputs/metrics/flow_vs_proxy_clean_v4_stage0_small.json")
    parser.add_argument("--step17-metrics", default="outputs/metrics/current_clean_v4_step17_metrics.json")
    parser.add_argument("--audit-metrics", default="outputs/metrics/current_v3_failure_diagnosis_for_clean_v4.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default="outputs/manifests/18_SCIENCE_GATE_AND_RUNBOOK_V4.manifest.json")
    args = parser.parse_args()

    prefixes = {
        "01": "01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4",
        "02": "02_AUDIT_CURRENT_V3_FOR_CLEAN_V4",
        "03": "03_ENTITY_ALIAS_MAP_V4",
        "04": "04_ENTITY_EVENT_SCORING_V4",
        "05": "05_ARTICLE_TYPE_AND_NOISE_FILTER_V4",
        "06": "06_DEDUP_NEWS_V4",
        "07": "07_TICKER_DATE_EVIDENCE_PACK_V4",
        "08": "08_RENDER_CONTEXT_EVIDENCE_V4",
        "09": "09_RATIONALE_PROMPT_EVIDENCE_ID_V4",
        "10": "10_STRICT_EVIDENCE_SCHEMA_VALIDATION_V4",
        "11": "11_EVIDENCE_GROUNDING_JUDGE_V4",
        "12": "12_TRACK_SPLIT_AND_TRAIN_POOL_V4",
        "13": "13_REGENERATE_RATIONALES_V4",
        "14": "14_INDEPENDENT_JUDGE_RERUN_EVIDENCE_V4",
        "15": "15_FLOW_REWARD_REBUILD_CLEAN_V4",
        "16": "16_ALIGNMENT_REBUILD_CLEAN_V4",
        "17": "17_BACKTEST_COUNTERFACTUAL_ABLATION_V4",
    }
    statuses = {key: latest_status(args.status_dir, prefix) for key, prefix in prefixes.items()}
    missing_or_failed = [
        {"step": key, "path": item.get("path"), "status": (item.get("payload") or {}).get("status")}
        for key, item in statuses.items()
        if not status_pass(item)
    ]
    flow = read_json(args.flow_metrics)
    step17 = read_json(args.step17_metrics)
    audit = read_json(args.audit_metrics)

    data_quality_allowed = True
    if audit and float(audit.get("no_news_rationale_rate", 1.0) or 1.0) >= 1.0:
        # The old V3 audit was bad; V4 is allowed only if later data-clean statuses passed.
        data_quality_allowed = all(status_pass(statuses[key]) for key in ["03", "04", "05", "06", "07"])
    flow_allowed = bool(flow.get("flow_reward_improvement", False))
    trading_allowed = bool(step17.get("trading_alpha_claim_allowed", False))
    cf_pass = step17.get("counterfactual_pass_rate")
    cf_no_change = step17.get("counterfactual_no_change_rate")
    cf_allowed = cf_pass is not None and (float(cf_pass) > 0.16 or float(cf_no_change or 1.0) < 0.696)
    ablation_allowed = int(step17.get("ablation_rows", 0) or 0) >= 7
    multimodal_allowed = data_quality_allowed and cf_allowed and ablation_allowed
    pipeline_go = not missing_or_failed
    claim_matrix = {
        "data_cleaning_improved_context_quality": "allowed" if data_quality_allowed else "blocked",
        "counterfactual_faithfulness": "allowed" if cf_allowed else "blocked",
        "flow_reward_improvement": "allowed" if flow_allowed else "blocked",
        "trading_alpha": "allowed" if trading_allowed else "blocked",
        "multimodal_news_technical_reasoning": "allowed" if multimodal_allowed else "blocked",
        "aaai_main_ready": "allowed" if pipeline_go and trading_allowed and flow_allowed and multimodal_allowed else "blocked",
    }
    report = {
        "pipeline_decision": "GO_SMALL" if pipeline_go else "BLOCKED",
        "claim_decision": "CLAIM_RESTRICTED" if "blocked" in claim_matrix.values() else "CLAIM_READY",
        "claim_matrix": claim_matrix,
        "status_evidence": statuses,
        "missing_or_failed_statuses": missing_or_failed,
        "flow_metrics": flow,
        "step17_metrics": step17,
        "runbook_next_steps": [
            "Scale rationale candidates beyond stage0 only after Step16-18 small pass.",
            "Do not claim flow reward improvement until flow beats proxy in at least 2 of 3 metrics.",
            "Do not claim trading alpha while Sharpe is non-positive or trading days are insufficient.",
            "Keep V4 artifacts separate from V3 artifacts until medium-scale gates pass.",
        ],
    }
    write_json(args.output, report)
    failures = []
    if not pipeline_go:
        failures.append(f"missing or failed statuses: {missing_or_failed}")
    write_manifest(args.manifest, [args.output, args.flow_metrics, args.step17_metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [item["path"] for item in statuses.values() if item.get("path")] + [args.flow_metrics, args.step17_metrics],
        [args.output, args.manifest, args.status],
        {
            "pipeline_decision": report["pipeline_decision"],
            "claim_decision": report["claim_decision"],
            "claim_matrix": claim_matrix,
            "missing_or_failed_count": len(missing_or_failed),
        },
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
