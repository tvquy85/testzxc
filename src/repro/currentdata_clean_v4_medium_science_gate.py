from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status
from src.utils.verify_status import validate_status

STEP = "20_STRICT_SCIENCE_GATE_MEDIUM"

REQUIRED_STATUSES = [
    "01_FREEZE_CLEAN_V4_SMALL_BASELINE.status.json",
    "02_AUDIT_CLEAN_V4_FAILURE_MODES.status.json",
    "03_BUILD_MEDIUM_SAMPLE_SELECTOR.status.json",
    "04_EVIDENCE_PACK_QUALITY_GATES_V4_1.status.json",
    "05_GENERATE_RATIONALES_MEDIUM_500X3.status.json",
    "06_RATIONALE_DIVERSITY_TEMPLATE_AUDIT.status.json",
    "07_INDEPENDENT_JUDGE_MEDIUM_FULL.status.json",
    "08_LABEL_ORDER_DEBIAS_MULTI_PERMUTATION.status.json",
    "09_GROUNDING_NEWS_NEGATIVE_FIX.status.json",
    "10_FLOW_SEMANTIC_EMBEDDINGS_V5.status.json",
    "11_FLOW_TARGET_NORMALIZATION_AND_SPLIT.status.json",
    "12_FLOW_TRAIN_EVAL_MEDIUM.status.json",
    "13_ALIGNMENT_DATASET_MEDIUM_RWSFT_DPO.status.json",
    "14_ALIGNMENT_TRAIN_ADAPTER_V4_MEDIUM.status.json",
    "15_PREDICT_WITH_ADAPTER_V4_MEDIUM.status.json",
    "16_BACKTEST_TRACK_BREAKDOWN_MEDIUM.status.json",
    "17_COUNTERFACTUAL_EVIDENCE_MEDIUM.status.json",
    "18_MINIMUM_BASELINES_PEN_SEP_POLICY.status.json",
    "19_ABLATION_SUITE_MEDIUM.status.json",
]


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def status_audit(status_dir: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for name in REQUIRED_STATUSES:
        path = Path(status_dir) / name
        ok, contract_failures = validate_status(path)
        payload = read_json(path)
        row = {
            "path": str(path).replace("\\", "/"),
            "valid_contract": ok,
            "status": payload.get("status"),
            "next_step_allowed": payload.get("next_step_allowed"),
            "failures": contract_failures or payload.get("failures", []),
        }
        rows.append(row)
        if not ok:
            failures.append(f"invalid status contract {path}: {contract_failures}")
        elif payload.get("status") != "PASS" or payload.get("next_step_allowed") is not True:
            failures.append(f"required status not PASS/allowed: {path}")
    return rows, failures


def claim_row(claim: str, allowed: bool, evidence: str, block_reason: str = "") -> dict[str, Any]:
    return {
        "claim": claim,
        "claim_allowed": bool(allowed),
        "claim_block_reason": "" if allowed else block_reason,
        "evidence": evidence,
    }


def metric_file(metrics_dir: str, name: str) -> str:
    return str(Path(metrics_dir) / name)


def table_file(tables_dir: str, name: str) -> str:
    return str(Path(tables_dir) / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--status-dir", default="outputs/status")
    parser.add_argument("--output", default="outputs/repro/currentdata_clean_v4_medium_science_gate_report.json")
    parser.add_argument("--claim-table", default="outputs/tables/medium_claim_matrix.csv")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    status_rows, failures = status_audit(args.status_dir)
    metric_paths = {
        "audit": metric_file(args.metrics_dir, "02_clean_v4_failure_modes.json"),
        "rationales": metric_file(args.metrics_dir, "05_generate_rationales_medium.json"),
        "judge": metric_file(args.metrics_dir, "07_independent_judge_medium.json"),
        "flow": metric_file(args.metrics_dir, "12_flow_vs_proxy_medium.json"),
        "alignment": metric_file(args.metrics_dir, "13_alignment_dataset_medium.json"),
        "training": metric_file(args.metrics_dir, "14_alignment_train_medium.json"),
        "backtest": metric_file(args.metrics_dir, "16_backtest_track_breakdown_medium.json"),
        "counterfactual": metric_file(args.metrics_dir, "17_counterfactual_evidence_medium.json"),
    }
    table_paths = {
        "baselines": table_file(args.tables_dir, "medium_baseline_comparison.csv"),
        "ablations": table_file(args.tables_dir, "medium_ablation_results.csv"),
    }
    metrics = {key: read_json(path) for key, path in metric_paths.items()}
    missing_metrics = [path for path in metric_paths.values() if not Path(path).exists()]
    missing_tables = [path for path in table_paths.values() if not Path(path).exists()]
    failures.extend([f"missing metrics file: {path}" for path in missing_metrics])
    failures.extend([f"missing table file: {path}" for path in missing_tables])

    baselines = pd.read_csv(table_paths["baselines"]) if Path(table_paths["baselines"]).exists() else pd.DataFrame()
    ablations = pd.read_csv(table_paths["ablations"]) if Path(table_paths["ablations"]).exists() else pd.DataFrame()
    if len(baselines) < 4:
        failures.append(f"baseline rows {len(baselines)} < 4")
    if len(ablations) < 5:
        failures.append(f"ablation rows {len(ablations)} < 5")
    if len(ablations) and "NOT_RUN" in set(ablations.astype(str).values.ravel()):
        failures.append("ablation table contains NOT_RUN")

    pipeline_pass = not failures
    rationales = metrics["rationales"]
    judge = metrics["judge"]
    flow = metrics["flow"]
    alignment = metrics["alignment"]
    training = metrics["training"]
    backtest = metrics["backtest"]
    counterfactual = metrics["counterfactual"]
    ablation_metrics = read_json(metric_file(args.metrics_dir, "19_ablation_suite_medium.json"))

    rationale_allowed = bool(
        rationales.get("rows", 0) >= 1500
        and rationales.get("parse_ok_rate", 0.0) >= 0.95
        and rationales.get("schema_ok_rate", 0.0) >= 0.95
    )
    judge_allowed = bool(judge.get("mean_true_label_probability", judge.get("true_label_probability_mean", 0.0)) > 0.20)
    flow_allowed = bool(flow.get("flow_reward_improvement") or flow.get("flow_claim_allowed"))
    alignment_allowed = bool(alignment.get("rwsft_examples", 0) >= 1000 and alignment.get("dpo_pairs", 0) >= 300)
    training_allowed = bool(training.get("rwsft_smoke_pass") and training.get("dpo_smoke_pass") or training.get("pipeline_pass"))
    trading_alpha_allowed = bool(backtest.get("alpha_claim_allowed") and backtest.get("num_trading_days", 0) >= 60)
    cf_allowed = bool(counterfactual.get("claim_allowed"))
    ablation_allowed = bool(ablation_metrics.get("all_required_ablations_present") and len(ablations) >= 5)
    aaaai_main_ready_allowed = False

    claims = [
        claim_row("medium_pipeline_reproducible", pipeline_pass, "required status contracts and files", "missing or failed status/artifact"),
        claim_row("rationale_medium_quality", rationale_allowed, metric_paths["rationales"], "rationale rows/schema/parse gate not met"),
        claim_row("independent_judge_signal", judge_allowed, metric_paths["judge"], "true-label probability gate not met"),
        claim_row("flow_reward_improvement", flow_allowed, metric_paths["flow"], "flow did not beat proxy on enough metrics"),
        claim_row("alignment_data_and_smoke_training", alignment_allowed and training_allowed, f"{metric_paths['alignment']} + {metric_paths['training']}", "alignment counts or adapter smoke gate not met"),
        claim_row("trading_alpha", trading_alpha_allowed, metric_paths["backtest"], "requires >=60 trading days and positive alpha gate"),
        claim_row("counterfactual_faithfulness", cf_allowed, metric_paths["counterfactual"], "counterfactual pass/no-change gate not met"),
        claim_row("medium_ablations_present", ablation_allowed, table_paths["ablations"], "required ablation rows missing or NOT_RUN"),
        claim_row("aaai_main_ready", aaaai_main_ready_allowed, "full-scale statistical validation", "medium-scale validation is not enough for main paper readiness"),
    ]
    claim_df = pd.DataFrame(claims)
    Path(args.claim_table).parent.mkdir(parents=True, exist_ok=True)
    claim_df.to_csv(args.claim_table, index=False)

    report = {
        "pipeline_decision": "GO_MEDIUM" if pipeline_pass else "BLOCKED",
        "claim_decision": "CLAIM_RESTRICTED" if not aaaai_main_ready_allowed else "CLAIM_READY",
        "aaaai_main_ready_allowed": aaaai_main_ready_allowed,
        "claims": claims,
        "status_audit": status_rows,
        "missing_metrics": missing_metrics,
        "missing_tables": missing_tables,
        "warnings": [
            "Medium-scale PASS validates the flow mechanics, not AAAI-ready paper claims.",
            "Full-scale statistical validation is still required before claiming main performance.",
        ],
        "metrics_snapshot": metrics,
    }
    write_json(args.output, report)

    outputs = [args.output, args.claim_table, args.manifest, args.status]
    inputs = [row["path"] for row in status_rows] + list(metric_paths.values()) + list(table_paths.values())
    write_manifest(args.manifest, [args.output, args.claim_table, *list(metric_paths.values()), *list(table_paths.values())], STEP)
    status = "PASS" if pipeline_pass else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs,
        outputs,
        {
            "pipeline_decision": report["pipeline_decision"],
            "claim_decision": report["claim_decision"],
            "claim_count": len(claims),
            "allowed_claim_count": int(claim_df["claim_allowed"].sum()),
            "aaaai_main_ready_allowed": aaaai_main_ready_allowed,
            "pipeline_pass": pipeline_pass,
            "claim_allowed": False,
        },
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "pipeline_decision": report["pipeline_decision"], "claim_decision": report["claim_decision"]}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
