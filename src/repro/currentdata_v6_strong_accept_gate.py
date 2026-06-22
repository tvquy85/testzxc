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

STEP = "19_STRICT_AAAI_STRONG_ACCEPT_GATE"

REQUIRED_STATUSES = [
    "01_FREEZE_MEDIUM_BASELINE_AND_TESTS.status.json",
    "02_HARD_EVENT_DATA_AUDIT.status.json",
    "03_HARD_EVENT_FILTER_V6.status.json",
    "04_EVIDENCE_PACK_REPAIR_V6.status.json",
    "05_RATIONALE_PROMPT_NEWS_USAGE_V6.status.json",
    "06_GENERATE_RATIONALES_1000X3_V6.status.json",
    "06_5_POST_PROCESSING_AND_REPAIR_V6.status.json",
    "07_RATIONALE_DIVERSITY_AND_TEMPLATE_GATE.status.json",
    "07_5_RATIONALE_TEMPLATE_DECOMPOSITION_V6.status.json",
    "08_INDEPENDENT_JUDGE_ENSEMBLE_V6.status.json",
    "09_JUDGE_CALIBRATION_AND_DEBIAS_TESTS.status.json",
    "10_FLOW_REWARD_V6_DECISION_TARGETS.status.json",
    "11_FLOW_EVAL_RAW_UTILITY_AND_PROXY.status.json",
    "11_5_FLOW_UTILITY_SURFACE_DIAGNOSTIC_V6.status.json",
    "11_6_FLOW_UTILITY_RERANKER_PROBE_V6.status.json",
    "11_7_FLOW_RERANKER_ABLATION_ATTRIBUTION_V6.status.json",
    "12_ALIGNMENT_PAIRS_MIN_GAP_AND_DIVERSITY.status.json",
    "13_TRAIN_RWSFT_DPO_V6.status.json",
    "14_PREDICT_WITH_V6_ADAPTERS.status.json",
    "14_6_DPO_FORECAST_DISTRIBUTION_REPAIR_V6.status.json",
    "14_6_RWSFT_FORECAST_DISTRIBUTION_REPAIR_V6.status.json",
    "14_6_DPO_VAL_FORECAST_DISTRIBUTION_REPAIR_V6.status.json",
    "14_5_BUILD_V6_VALIDATION_CONTEXTS.status.json",
    "14_5_PREDICT_DPO_VAL_V6.status.json",
    "15_BACKTEST_TRACK_BASELINE_V6.status.json",
    "15_5_TRADING_POLICY_VARIANT_PROBE_V6.status.json",
    "15_6_VALIDATION_SELECTED_TRADING_POLICY_V6.status.json",
    "16A_BUILD_COUNTERFACTUAL_V6.status.json",
    "16_COUNTERFACTUAL_NEWS_EVIDENCE_V6.status.json",
    "16_5_COUNTERFACTUAL_ELIGIBILITY_AUDIT_V6.status.json",
    "16_6_COUNTERFACTUAL_QUALITY_FILTERED_TASKS_V6.status.json",
    "16_6_COUNTERFACTUAL_QUALITY_FILTERED_EVAL_V6.status.json",
    "16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_TASKS_V6.status.json",
    "16_7_COUNTERFACTUAL_SEMANTIC_NEUTRALIZED_EVAL_V6.status.json",
    "17_BASELINES_SEP_POLICY_TECH_RULE.status.json",
    "17_5_VALIDATION_CALIBRATED_FORECAST_HYBRID_V6.status.json",
    "17_6_VALIDATION_STACKED_FORECAST_PROBE_V6.status.json",
    "17_7_SUPERVISED_SIGNAL_CEILING_PROBE_V6.status.json",
    "17_8_REPAIRED_FORECAST_BASELINE_PROBE_V6.status.json",
    "18_ABLATIONS_V6.status.json",
    "18_5_STATISTICAL_TESTS_AND_CI_V6.status.json",
]


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def allow_flow_claim(wins: dict[str, Any]) -> bool:
    return sum(bool(wins.get(key)) for key in ["rank", "pair", "top_decile"]) >= 2


def claim_row(claim: str, allowed: bool, evidence: str, reason: str) -> dict[str, Any]:
    return {
        "claim": claim,
        "claim_allowed": bool(allowed),
        "claim_block_reason": "" if allowed else reason,
        "evidence": evidence,
    }


def audit_statuses(status_dir: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for name in REQUIRED_STATUSES:
        path = Path(status_dir) / name
        valid, contract_failures = validate_status(path)
        payload = read_json(path)
        row = {
            "status_file": str(path).replace("\\", "/"),
            "valid_contract": bool(valid),
            "status": payload.get("status"),
            "next_step_allowed": payload.get("next_step_allowed"),
            "failures": contract_failures or payload.get("failures", []),
        }
        rows.append(row)
        if not valid:
            failures.append(f"invalid status contract: {path}: {contract_failures}")
        elif payload.get("status") != "PASS" or payload.get("next_step_allowed") is not True:
            failures.append(f"required status not PASS/allowed: {path}")
    return rows, failures


def table(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--status-dir", default="outputs/status")
    parser.add_argument("--output", default="outputs/repro/currentdata_v6_strong_accept_gate.json")
    parser.add_argument("--claim-table", default="outputs/tables/19_v6_claim_matrix.csv")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    tables_dir = Path(args.tables_dir)
    status_rows, failures = audit_statuses(args.status_dir)

    paths = {
        "rationale": metrics_dir / "07_v6_rationale_diversity.json",
        "rationale_decomposition": metrics_dir / "07_5_v6_rationale_template_decomposition.json",
        "judge": metrics_dir / "09_v6_judge_calibration.json",
        "flow": metrics_dir / "11_v6_flow_vs_proxy_raw_utility.json",
        "flow_reranker": metrics_dir / "11_6_v6_flow_utility_reranker_probe.json",
        "flow_attribution": metrics_dir / "11_7_v6_flow_reranker_ablation_attribution.json",
        "alignment": metrics_dir / "12_v6_alignment_dataset.json",
        "training": metrics_dir / "13_v6_alignment_training.json",
        "prediction": metrics_dir / "14_v6_prediction_metrics.json",
        "dpo_repair": metrics_dir / "14_6_v6_dpo_forecast_distribution_repair.json",
        "rwsft_repair": metrics_dir / "14_6_v6_rwsft_forecast_distribution_repair.json",
        "dpo_val_repair": metrics_dir / "14_6_v6_dpo_val_forecast_distribution_repair.json",
        "backtest": metrics_dir / "15_v6_backtest_track_baseline.json",
        "trading_policy_variants": metrics_dir / "15_5_v6_trading_policy_variant_probe.json",
        "validation_selected_trading": metrics_dir / "15_6_v6_validation_selected_trading_policy.json",
        "counterfactual": metrics_dir / "16_v6_counterfactual_news_evidence.json",
        "counterfactual_quality_tasks": metrics_dir / "16_6_v6_counterfactual_quality_filtered_tasks.json",
        "counterfactual_quality_eval": metrics_dir / "16_6_v6_counterfactual_quality_filtered_eval.json",
        "counterfactual_semantic_tasks": metrics_dir / "16_7_v6_counterfactual_semantic_neutralized_tasks.json",
        "counterfactual_semantic_eval": metrics_dir / "16_7_v6_counterfactual_semantic_eval.json",
        "baselines": metrics_dir / "17_v6_baseline_comparison.json",
        "stacked_forecast": metrics_dir / "17_6_v6_validation_stacked_forecast_probe.json",
        "supervised_ceiling": metrics_dir / "17_7_v6_supervised_signal_ceiling_probe.json",
        "repaired_forecast": metrics_dir / "17_8_v6_repaired_forecast_baseline_probe.json",
        "ablations": metrics_dir / "18_v6_ablation_summary.json",
        "statistics": metrics_dir / "19_v6_statistical_tests.json",
    }
    tables = {
        "baseline_table": tables_dir / "17_v6_comparable_baselines.csv",
        "ablation_table": tables_dir / "18_v6_ablation_results.csv",
        "statistical_tests": tables_dir / "19_v6_statistical_tests.csv",
    }
    missing = [str(path) for path in [*paths.values(), *tables.values()] if not Path(path).exists()]
    failures.extend(f"missing evidence file: {path}" for path in missing)

    metrics = {key: read_json(path) for key, path in paths.items()}
    baseline_table = table(str(tables["baseline_table"]))
    ablation_table = table(str(tables["ablation_table"]))

    pipeline_pass = not failures
    rationale = metrics["rationale"]
    rationale_decomposition = metrics["rationale_decomposition"]
    judge = metrics["judge"]
    flow = metrics["flow"]
    flow_reranker = metrics["flow_reranker"]
    flow_attribution = metrics["flow_attribution"]
    alignment = metrics["alignment"]
    training = metrics["training"]
    dpo_repair = metrics["dpo_repair"]
    repaired_forecast = metrics["repaired_forecast"]
    prediction = metrics["prediction"]
    backtest = metrics["backtest"]
    trading_policy_variants = metrics["trading_policy_variants"]
    validation_selected_trading = metrics["validation_selected_trading"]
    counterfactual = metrics["counterfactual"]
    counterfactual_quality_tasks = metrics["counterfactual_quality_tasks"]
    counterfactual_quality_eval = metrics["counterfactual_quality_eval"]
    counterfactual_semantic_tasks = metrics["counterfactual_semantic_tasks"]
    counterfactual_semantic_eval = metrics["counterfactual_semantic_eval"]
    baselines = metrics["baselines"]
    stacked_forecast = metrics["stacked_forecast"]
    supervised_ceiling = metrics["supervised_ceiling"]
    ablations = metrics["ablations"]
    statistics = metrics["statistics"]

    data_evidence_allowed = pipeline_pass
    rationale_original_allowed = bool(
        rationale.get("news_rationale_empty_when_N_rate", 1.0) <= 0.05
        and rationale.get("evidence_citation_rate", 0.0) >= 0.95
        and rationale.get("technical_only_phrase_rate", 1.0) <= 0.15
        and rationale.get("mean_within_sample_jaccard", 1.0) <= 0.68
        and rationale.get("repeated_template_cluster_rate", 1.0) <= 0.25
    )
    rationale_decomposed_allowed = bool(
        rationale_decomposition.get("claim_allowed", False)
        and rationale_decomposition.get("news_rationale_empty_when_N_rate", 1.0) <= 0.05
        and rationale_decomposition.get("evidence_citation_rate", 0.0) >= 0.95
        and rationale_decomposition.get("technical_only_phrase_rate", 1.0) <= 0.15
        and rationale_decomposition.get("news_plus_meta_mean_jaccard", 1.0) <= 0.68
        and rationale_decomposition.get("news_plus_meta_template_cluster_rate", 1.0) <= 0.25
        and rationale_decomposition.get("news_repeated_phrase_rate", 1.0) <= 0.05
    )
    rationale_allowed = bool(rationale_original_allowed or rationale_decomposed_allowed)
    judge_allowed = bool(
        judge.get("mean_true_label_probability", 0.0) > 0.22
        and judge.get("mean_argmax_consistency_ensemble", 0.0) >= 0.75
    )
    flow_allowed = bool(allow_flow_claim(flow.get("metric_wins", {})) and flow.get("flow_claim_allowed", False))
    alignment_artifact_allowed = bool(
        alignment.get("rwsft_examples", 0) >= 1000
        and alignment.get("dpo_pairs", 0) >= 300
        and training.get("adapters_exist", False)
    )
    dpo_beats_rwsft = False
    if not baseline_table.empty:
        lookup = {str(row["method"]): row for _, row in baseline_table.iterrows()}
        dpo = lookup.get("Qwen_DPO_V6")
        rwsft = lookup.get("Qwen_RWSFT_V6")
        if dpo is not None and rwsft is not None:
            dpo_beats_rwsft = bool(
                float(dpo.get("macro_f1", 0.0)) > float(rwsft.get("macro_f1", 0.0))
                and float(dpo.get("mcc", 0.0)) > float(rwsft.get("mcc", 0.0))
            )
    alignment_claim_allowed = bool(alignment_artifact_allowed and dpo_beats_rwsft)
    prediction_allowed = bool(
        baselines.get("dpo_beats_technical_rule_macro_f1", False)
        and baselines.get("dpo_beats_technical_rule_mcc", False)
    )
    stacked_forecast_diagnostic_pass = bool(
        stacked_forecast.get("pipeline_pass", False)
        and not stacked_forecast.get("claim_allowed", True)
        and not stacked_forecast.get("paired_ci_support", True)
    )
    supervised_ceiling_diagnostic_pass = bool(
        supervised_ceiling.get("pipeline_pass", False)
        and not supervised_ceiling.get("claim_allowed", True)
        and not supervised_ceiling.get("paired_ci_support", True)
    )
    alpha_diagnostic_allowed = bool(backtest.get("alpha_claim_allowed", False) and backtest.get("num_trading_days", 0) >= 60)
    statistical_tests_allowed = bool(
        statistics.get("pipeline_pass", False)
        and statistics.get("required_tests_present", False)
        and statistics.get("confidence_interval_available", False)
        and Path(tables["statistical_tests"]).exists()
    )
    alpha_paper_allowed = bool(
        (
            backtest.get("sharpe_daily_annualized", 0.0) > 0
            and backtest.get("num_trading_days", 0) >= 120
            and statistical_tests_allowed
            and statistics.get("alpha_paper_level_supported", False)
        )
        or (
            validation_selected_trading.get("pipeline_pass", False)
            and validation_selected_trading.get("alpha_paper_level_supported", False)
        )
    )
    counterfactual_allowed = bool(
        counterfactual.get("claim_allowed", False)
        or (
            counterfactual_quality_tasks.get("pipeline_pass", False)
            and counterfactual_quality_eval.get("pipeline_pass", False)
            and counterfactual_quality_eval.get("claim_allowed", False)
        )
        or (
            counterfactual_semantic_tasks.get("pipeline_pass", False)
            and counterfactual_semantic_eval.get("pipeline_pass", False)
            and counterfactual_semantic_eval.get("claim_allowed", False)
            and counterfactual_semantic_eval.get("news_faithfulness_claim_allowed", False)
        )
    )
    comparable_baselines_allowed = bool(
        baselines.get("required_methods_present", False)
        and baselines.get("comparable_baseline_count", 0) >= 6
        and "NOT_RUN" not in set(baseline_table.astype(str).values.ravel()) if not baseline_table.empty else False
    )
    ablations_allowed = bool(
        ablations.get("required_ablations_present", False)
        and not ablations.get("not_run_present", True)
        and "NOT_RUN" not in set(ablation_table.astype(str).values.ravel()) if not ablation_table.empty else False
    )
    alpha_block_reason = (
        "requires Sharpe > 0, >=120 days, and confidence intervals that support positive alpha; "
        "Step 15.5 is post-hoc diagnostic only and Step 15.6 validation-selected DPO still has absolute "
        "Sharpe/mean-return CIs crossing zero"
        if validation_selected_trading.get("pipeline_pass", False)
        else (
            "requires Sharpe > 0, >=120 days, and confidence intervals that support positive alpha; "
            "policy-variant probe is diagnostic only because variants share the same observed test period"
            if trading_policy_variants.get("pipeline_pass", False)
            else "requires Sharpe > 0, >=120 days, and confidence intervals that support positive alpha"
        )
    )
    if flow_attribution.get("pipeline_pass", False) and not flow_attribution.get("flow_attribution_supported", True):
        flow_block_reason = (
            "Official Flow did not beat proxy on at least 2/3 core metrics; Step 11.6 reranker wins rank/pair, "
            "but Step 11.7 no-flow ablation matches or exceeds the full feature set while only-flow underperforms, "
            "so the diagnostic win is not attributable to Flow"
        )
    elif flow_reranker.get("pipeline_pass", False) and flow_reranker.get("eval_core_utility_win_vs_proxy", False):
        flow_block_reason = (
            "Official Flow did not beat proxy on at least 2/3 core metrics; utility-reranker diagnostic wins rank/pair "
            "on validation but is not the official Flow checkpoint and still misses top-decile utility"
        )
    else:
        flow_block_reason = "Flow did not beat proxy on at least 2/3 core metrics"
    if counterfactual_quality_eval.get("pipeline_pass", False):
        counterfactual_block_reason = (
            "Step 16 official pass/no-change gates failed; Step 16.6 quality-filtered rerun improves no-change "
            f"to {counterfactual_quality_eval.get('no_change_rate', 1.0):.4f} and passes news removal gates, "
            f"but overall pass_rate {counterfactual_quality_eval.get('pass_rate', 0.0):.4f} remains below 0.50 "
            f"and neutralize_negative_evidence is {counterfactual_quality_eval.get('pass_rate_neutralize_negative_evidence', 0.0):.4f}"
        )
    else:
        counterfactual_block_reason = "counterfactual pass/no-change/news gates failed"

    claims = [
        claim_row("v6_pipeline_runnable", pipeline_pass, "outputs/status/*.status.json", "one or more required V6 status files failed"),
        claim_row("data_evidence_quality", data_evidence_allowed, str(paths["prediction"]), "data/evidence status gates failed"),
        claim_row("rationale_quality", rationale_allowed, f"{paths['rationale']} + {paths['rationale_decomposition']}", "rationale diversity/template preferred gates failed"),
        claim_row("judge_quality", judge_allowed, str(paths["judge"]), "judge probability or label-order consistency gate failed"),
        claim_row(
            "flow_reward_improvement",
            flow_allowed,
            f"{paths['flow']} + {paths['flow_reranker']} + {paths['flow_attribution']}",
            flow_block_reason,
        ),
        claim_row("alignment_artifacts", alignment_artifact_allowed, f"{paths['alignment']} + {paths['training']}", "alignment data or adapter artifacts insufficient"),
        claim_row(
            "alignment_improves_over_sft",
            alignment_claim_allowed,
            f"{tables['baseline_table']} + {paths['dpo_repair']} + {paths['repaired_forecast']}",
            "DPO did not beat RWSFT/SFT baseline on both Macro-F1 and MCC",
        ),
        claim_row(
            "forecast_beats_technical_rule",
            prediction_allowed,
            f"{paths['baselines']} + {paths['stacked_forecast']} + {paths['supervised_ceiling']}",
            (
                "Repaired DPO still did not beat Technical_Rule on both Macro-F1 and MCC; "
                "validation-stacked and supervised-ceiling diagnostics have no CI-supported test improvement"
                if stacked_forecast_diagnostic_pass and supervised_ceiling_diagnostic_pass
                else "Repaired DPO still did not beat Technical_Rule on both Macro-F1 and MCC; validation-stacked diagnostic has no CI-supported test improvement"
                if stacked_forecast_diagnostic_pass
                else "Repaired DPO still did not beat Technical_Rule on both Macro-F1 and MCC; supervised-ceiling diagnostic has no CI-supported test improvement"
                if supervised_ceiling_diagnostic_pass
                else "Repaired DPO still did not beat Technical_Rule on both Macro-F1 and MCC"
            ),
        ),
        claim_row("trading_alpha_diagnostic", alpha_diagnostic_allowed, str(paths["backtest"]), "diagnostic alpha gate not met"),
        claim_row(
            "trading_alpha_paper_level",
            alpha_paper_allowed,
            f"{paths['backtest']} + {paths['statistics']} + {paths['trading_policy_variants']} + {paths['validation_selected_trading']}",
            alpha_block_reason,
        ),
        claim_row(
            "counterfactual_faithfulness",
            counterfactual_allowed,
            f"{paths['counterfactual']} + {paths['counterfactual_quality_tasks']} + {paths['counterfactual_quality_eval']} + {paths['counterfactual_semantic_tasks']} + {paths['counterfactual_semantic_eval']}",
            counterfactual_block_reason,
        ),
        claim_row("comparable_baselines", comparable_baselines_allowed, str(tables["baseline_table"]), "required comparable baselines missing or NOT_RUN present"),
        claim_row("ablations_present", ablations_allowed, str(tables["ablation_table"]), "required ablations missing or NOT_RUN present"),
        claim_row("statistical_tests", statistical_tests_allowed, str(tables["statistical_tests"]), "statistical tests/confidence intervals are missing or incomplete"),
    ]
    major_claims = [
        flow_allowed,
        alignment_claim_allowed,
        prediction_allowed,
        alpha_paper_allowed,
        counterfactual_allowed,
        comparable_baselines_allowed,
        ablations_allowed,
        statistical_tests_allowed,
    ]
    strong_accept_ready = bool(pipeline_pass and all(major_claims))
    claim_decision = "STRONG_ACCEPT_READY" if strong_accept_ready else "CLAIM_RESTRICTED"
    pipeline_decision = "GO_V6_PIPELINE" if pipeline_pass else "BLOCKED"

    claim_df = pd.DataFrame(claims)
    Path(args.claim_table).parent.mkdir(parents=True, exist_ok=True)
    claim_df.to_csv(args.claim_table, index=False)
    report = {
        "pipeline_decision": pipeline_decision,
        "claim_decision": claim_decision,
        "strong_accept_ready": strong_accept_ready,
        "pipeline_pass": pipeline_pass,
        "allowed_claims": claim_df[claim_df["claim_allowed"]]["claim"].tolist(),
        "blocked_claims": claim_df[~claim_df["claim_allowed"]][["claim", "claim_block_reason", "evidence"]].to_dict(orient="records"),
        "status_audit": status_rows,
        "missing_evidence_files": missing,
        "metrics_snapshot": metrics,
        "warnings": [
            "Pipeline PASS is not a scientific claim.",
            "Strong-accept readiness requires every major claim gate plus statistical tests.",
            "Step 17.6 is diagnostic only and cannot override the official Technical_Rule forecast gate.",
            "Step 17.7 is diagnostic only and cannot open the aligned-model forecast claim.",
            "Step 15.5 is diagnostic only and cannot open paper-level alpha after multiple variant comparisons on the same test period.",
            "Step 15.6 validation-selected trading is a stricter diagnostic; it cannot open paper-level alpha while absolute Sharpe/mean-return CIs cross zero.",
            "Step 11.6 is diagnostic only and cannot open the Flow claim until an official checkpoint and ablations pass.",
            "Step 11.7 shows the Step 11.6 reranker win is not attributable to Flow-specific signal.",
            "Step 14.6 repaired DPO forecast distributions are now the canonical Step 14 evaluation input; repair is label-free probability normalization.",
            "Step 17.8 shows repaired DPO beats RWSFT point metrics but still trails Technical_Rule.",
            "Step 16.7 semantic-neutralized counterfactual evaluation opens the counterfactual gate, but this does not affect Flow, forecast, or paper-level alpha blockers.",
        ],
    }
    write_json(args.output, report)
    manifest_paths = [args.output, args.claim_table, *[str(path) for path in paths.values()], *[str(path) for path in tables.values()]]
    write_manifest(args.manifest, manifest_paths, STEP)
    status = "PASS" if pipeline_pass else "FAIL"
    status_metrics = {
        "pipeline_decision": pipeline_decision,
        "claim_decision": claim_decision,
        "strong_accept_ready": strong_accept_ready,
        "allowed_claim_count": int(claim_df["claim_allowed"].sum()),
        "blocked_claim_count": int((~claim_df["claim_allowed"]).sum()),
        "pipeline_pass": pipeline_pass,
        "claim_allowed": strong_accept_ready,
    }
    write_status(
        args.status,
        STEP,
        status,
        [row["status_file"] for row in status_rows],
        [args.output, args.claim_table, args.manifest, args.status],
        status_metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, **status_metrics}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
