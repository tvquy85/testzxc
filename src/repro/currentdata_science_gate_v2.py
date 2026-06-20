from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import write_json, write_manifest, write_status
from src.utils.verify_status import validate_status

STEP = "18_AAAI_SCIENCE_GATE_STRICT"

REQUIRED_STATUSES = [
    "outputs/status/01_FREEZE_BASELINE_CURRENTDATA.status.json",
    "outputs/status/02_CURRENT_DATA_QUALITY_AUDIT.status.json",
    "outputs/status/03_ENTITY_EVENT_FILTER_CURRENT_DATA.status.json",
    "outputs/status/04_TICKER_DATE_CONTEXT_AGGREGATION.status.json",
    "outputs/status/05_LABEL_BACKTEST_TARGET_FIX_ABNORMAL_RETURN.status.json",
    "outputs/status/06_RATIONALE_PROMPT_CLEAN_CONTEXT.status.json",
    "outputs/status/07_INDEPENDENT_INFERABILITY_JUDGE.status.json",
    "outputs/status/08_JUDGE_DEBIAS_LABEL_ORDER_RANDOMIZATION.status.json",
    "outputs/status/09_CLAIM_EXTRACTION_GROUNDING_V2.status.json",
    "outputs/status/10_FLOW_DATASET_V3_TARGETS_AND_EMBEDDINGS.status.json",
    "outputs/status/11_FLOW_TRAIN_EVAL_FIX_VALID_SPLIT.status.json",
    "outputs/status/12_RWSFT_DPO_REBUILD_FROM_INDEPENDENT_REWARDS.status.json",
    "outputs/status/13_ALIGNMENT_REAL_RUN_CURRENT_DATA.status.json",
    "outputs/status/14_DAILY_BACKTEST_ABNORMAL_AND_COSTS.status.json",
    "outputs/status/15_COUNTERFACTUAL_DIRECTIONAL_FIX_CURRENT_DATA.status.json",
    "outputs/status/16_MINIMUM_ABLATIONS_CURRENT_DATA.status.json",
    "outputs/status/17_PAPER_TABLES_NEGATIVE_RESULTS_GATES.status.json",
]


def read_json(path: str) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def status_payload(path: str) -> dict[str, Any]:
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-matrix", default="outputs/metrics/current_v3_claim_matrix.json")
    parser.add_argument("--backtest-metrics", default="outputs/metrics/backtest_daily_portfolio_current_v3.json")
    parser.add_argument("--counterfactual-metrics", default="outputs/metrics/counterfactual_directional_current_v3.json")
    parser.add_argument("--flow-eval", default="outputs/metrics/flow_vs_proxy_v3_1_eval.json")
    parser.add_argument("--output", default="outputs/repro/currentdata_science_gate_report_v2.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--required-status", action="append", default=[])
    args = parser.parse_args()

    required_statuses = args.required_status or REQUIRED_STATUSES
    blocking_issues: list[str] = []
    warnings: list[str] = []
    evidence_files: list[str] = []
    status_audit: dict[str, Any] = {}
    for status_path in required_statuses:
        ok, failures = validate_status(status_path)
        payload = status_payload(status_path)
        status_audit[status_path] = {
            "valid_contract": ok,
            "status": payload.get("status"),
            "next_step_allowed": payload.get("next_step_allowed"),
            "failures": failures or payload.get("failures", []),
        }
        if not ok:
            blocking_issues.append(f"invalid status contract {status_path}: {failures}")
        elif payload.get("status") != "PASS" or payload.get("next_step_allowed") is not True:
            blocking_issues.append(f"required status not PASS/allowed: {status_path}")
        evidence_files.append(status_path)

    claim_matrix = read_json(args.claim_matrix)
    backtest = read_json(args.backtest_metrics)
    cf = read_json(args.counterfactual_metrics)
    flow = read_json(args.flow_eval)
    for source, metrics in [
        (args.claim_matrix, claim_matrix),
        (args.backtest_metrics, backtest),
        (args.counterfactual_metrics, cf),
        (args.flow_eval, flow),
    ]:
        if not metrics:
            blocking_issues.append(f"missing evidence metrics: {source}")
        else:
            evidence_files.append(source)

    allowed_claims: list[str] = []
    blocked_claims: list[dict[str, str]] = []
    alpha_allowed = bool(backtest.get("alpha_claim_allowed", False))
    flow_allowed = bool(flow.get("flow_claim_allowed", False))
    cf_allowed = bool((cf.get("pass_rate") or 0) > 0.16 or (cf.get("no_change_rate") or 1.0) < 0.696)
    if alpha_allowed:
        allowed_claims.append("trading_alpha")
    else:
        blocked_claims.append({"claim": "trading_alpha", "reason": "backtest alpha gate not reached"})
    if flow_allowed:
        allowed_claims.append("flow_reward_improvement")
    else:
        blocked_claims.append({"claim": "flow_reward_improvement", "reason": "flow did not beat proxy on >=2 metrics"})
    if cf_allowed:
        allowed_claims.append("counterfactual_faithfulness")
    else:
        blocked_claims.append({"claim": "counterfactual_faithfulness", "reason": "counterfactual directional gate not reached"})

    if not allowed_claims:
        warnings.append("Pipeline may be mechanically complete, but scientific claims remain restricted.")
    if claim_matrix and claim_matrix.get("blocked_claims"):
        warnings.append("Paper tables include blocked claims; report negative results explicitly.")

    pipeline_decision = "GO" if not blocking_issues else "NO_GO"
    claim_decision = "CLAIM_ALLOWED" if allowed_claims and pipeline_decision == "GO" else "CLAIM_RESTRICTED"
    report = {
        "pipeline_decision": pipeline_decision,
        "claim_decision": claim_decision,
        "allowed_claims": allowed_claims if pipeline_decision == "GO" else [],
        "blocked_claims": blocked_claims,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "status_audit": status_audit,
        "evidence_files": evidence_files,
        "recommendation": "Scale up only after NO_GO issues are resolved" if pipeline_decision != "GO" else "Scale medium before making paper claims",
    }

    write_json(args.output, report)
    write_manifest(args.manifest, [args.output, args.claim_matrix, args.backtest_metrics, args.counterfactual_metrics, args.flow_eval, *required_statuses], STEP)
    status = "PASS" if pipeline_decision == "GO" else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.claim_matrix, args.backtest_metrics, args.counterfactual_metrics, args.flow_eval, *required_statuses],
        [args.output, args.manifest, args.status],
        report,
        blocking_issues,
        status == "PASS",
    )
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
