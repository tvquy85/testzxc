from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "18_ABLATIONS_V6"
REQUIRED_ABLATIONS = [
    "Full_V6",
    "No_News_Evidence",
    "Technical_Only",
    "No_Technical_Tokens",
    "No_Flow_Use_Proxy",
    "No_Debias_Use_Normal_Order",
    "No_Grounding_Filter",
    "No_DPO_RWSFT_Only",
    "Base_Qwen_NoAlign",
    "Technical_Rule",
]


def read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def baseline_lookup(path: str) -> dict[str, dict[str, Any]]:
    table = pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    if table.empty or "method" not in table.columns:
        return {}
    return {str(row["method"]): row.to_dict() for _, row in table.iterrows()}


def metric_from_baseline(row: dict[str, Any] | None, key: str) -> Any:
    if not row:
        return None
    value = row.get(key)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def ablation_row(
    ablation: str,
    source_method: str,
    source_row: dict[str, Any] | None,
    status: str,
    ablation_type: str,
    notes: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "ablation": ablation,
        "status": status,
        "ablation_type": ablation_type,
        "source_method": source_method,
        "rows": metric_from_baseline(source_row, "rows"),
        "accuracy": metric_from_baseline(source_row, "accuracy"),
        "macro_f1": metric_from_baseline(source_row, "macro_f1"),
        "mcc": metric_from_baseline(source_row, "mcc"),
        "balanced_accuracy": metric_from_baseline(source_row, "balanced_accuracy"),
        "schema_ok_rate": metric_from_baseline(source_row, "schema_ok_rate"),
        "reference_only": bool(metric_from_baseline(source_row, "reference_only")) if source_row else False,
        "claim_allowed": False,
        "notes": notes,
    }
    if extra:
        out.update(extra)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--backtest-metrics", default="outputs/metrics/15_v6_backtest_track_baseline.json")
    parser.add_argument("--counterfactual-metrics", default="outputs/metrics/16_v6_counterfactual_news_evidence.json")
    parser.add_argument("--flow-metrics", default="outputs/metrics/11_v6_flow_vs_proxy_raw_utility.json")
    parser.add_argument("--alignment-metrics", default="outputs/metrics/12_v6_alignment_dataset.json")
    parser.add_argument("--judge-metrics", default="outputs/metrics/09_v6_judge_calibration.json")
    parser.add_argument("--output", default="outputs/tables/18_v6_ablation_results.csv")
    parser.add_argument("--metrics", default="outputs/metrics/18_v6_ablation_summary.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    for path, label in [(args.contexts, "contexts"), (args.predictions, "predictions"), (args.baselines, "baselines")]:
        if not Path(path).exists():
            failures.append(f"{label} missing: {path}")

    baselines = baseline_lookup(args.baselines)
    backtest = read_json(args.backtest_metrics)
    counterfactual = read_json(args.counterfactual_metrics)
    flow = read_json(args.flow_metrics)
    alignment = read_json(args.alignment_metrics)
    judge = read_json(args.judge_metrics)

    rows: list[dict[str, Any]] = []
    if not failures:
        full = baselines.get("Qwen_DPO_V6")
        technical = baselines.get("Technical_Rule")
        rwsft = baselines.get("Qwen_RWSFT_V6")
        sep = baselines.get("SEP_Style_Summarize_Explain")
        policy = baselines.get("Policy_Style_Scalar_Proxy")
        base = baselines.get("Qwen_Base_NoAlign")
        rows.extend(
            [
                ablation_row(
                    "Full_V6",
                    "Qwen_DPO_V6",
                    full,
                    "PASS",
                    "executed_prediction_backtest_counterfactual",
                    "Full V6 DPO artifact with Step 15 backtest and Step 16 counterfactual diagnostics.",
                    {
                        "sharpe_daily_annualized": backtest.get("sharpe_daily_annualized"),
                        "counterfactual_pass_rate": counterfactual.get("pass_rate"),
                        "counterfactual_no_change_rate": counterfactual.get("no_change_rate"),
                        "claim_block_reason": "baseline_or_counterfactual_gate_failed",
                    },
                ),
                ablation_row(
                    "No_News_Evidence",
                    "Technical_Rule",
                    technical,
                    "DIAGNOSTIC_PROXY",
                    "technical_only_reference",
                    "Uses Technical_Rule as the no-news evidence proxy on the same current-data rows.",
                ),
                ablation_row(
                    "Technical_Only",
                    "Technical_Rule",
                    technical,
                    "DIAGNOSTIC_PROXY",
                    "technical_only_reference",
                    "Technical-token-only deterministic rule.",
                ),
                ablation_row(
                    "No_Technical_Tokens",
                    "SEP_Style_Summarize_Explain",
                    sep,
                    "DIAGNOSTIC_PROXY",
                    "news_only_reference",
                    "Evidence-text polarity proxy; not a separate no-technical LLM generation artifact.",
                ),
                ablation_row(
                    "No_Flow_Use_Proxy",
                    "Qwen_DPO_V6",
                    full,
                    "PASS",
                    "executed_pipeline_design",
                    "Step 12 used proxy reward because Flow claim failed; this is the actual V6 alignment path.",
                    {
                        "flow_claim_allowed": bool(flow.get("flow_claim_allowed", False)),
                        "flow_scores_used": bool(alignment.get("flow_scores_used", False)),
                        "reward_source": alignment.get("reward_source"),
                    },
                ),
                ablation_row(
                    "No_Debias_Use_Normal_Order",
                    "judge_normal_order_reference",
                    None,
                    "DIAGNOSTIC_ONLY",
                    "judge_debias_reference",
                    "No separate aligned-model prediction artifact exists for normal-order-only judging; Step 09 shows low label-order KL after debias.",
                    {
                        "judge_rows": judge.get("rows"),
                        "mean_label_order_kl": judge.get("mean_label_order_kl"),
                        "mean_argmax_consistency_ensemble": judge.get("mean_argmax_consistency_ensemble"),
                    },
                ),
                ablation_row(
                    "No_Grounding_Filter",
                    "pre_filter_reference",
                    None,
                    "DIAGNOSTIC_ONLY",
                    "grounding_filter_reference",
                    "No separate no-grounding-filter prediction artifact exists; Step 03/04 filtering remains part of Full_V6.",
                    {
                        "context_rows": int(pd.read_parquet(args.contexts).shape[0]) if Path(args.contexts).exists() else None,
                    },
                ),
                ablation_row(
                    "No_DPO_RWSFT_Only",
                    "Qwen_RWSFT_V6",
                    rwsft,
                    "PASS",
                    "executed_prediction",
                    "RWSFT-only Step 14 prediction artifact before DPO preference optimization.",
                ),
                ablation_row(
                    "Base_Qwen_NoAlign",
                    "Qwen_Base_NoAlign",
                    base,
                    "REFERENCE_ONLY",
                    "missing_current_data_artifact",
                    "No current-data base no-align prediction artifact was produced; row is excluded from outperform claims.",
                ),
                ablation_row(
                    "Technical_Rule",
                    "Technical_Rule",
                    technical,
                    "PASS",
                    "executed_deterministic_baseline",
                    "Best current-data Macro-F1 baseline from Step 17.",
                ),
                ablation_row(
                    "Policy_Style_Scalar_Proxy",
                    "Policy_Style_Scalar_Proxy",
                    policy,
                    "DIAGNOSTIC_PROXY",
                    "policy_proxy_reference",
                    "Included for traceability to Step 17 policy-style scalar proxy.",
                ),
            ]
        )

    table = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    if table.empty:
        failures.append("ablation table is empty")
    if len(table) and "NOT_RUN" in set(table.astype(str).values.ravel()):
        failures.append("ablation table contains NOT_RUN")
    present = set(table["ablation"].astype(str)) if len(table) and "ablation" in table.columns else set()
    missing = sorted(set(REQUIRED_ABLATIONS) - present)
    if missing:
        failures.append(f"missing required ablations: {missing}")

    comparable = table[table["status"].isin(["PASS", "DIAGNOSTIC_PROXY"])].copy() if len(table) else pd.DataFrame()
    metrics = {
        "ablation_count": int(len(table)),
        "required_ablations_present": not missing,
        "not_run_present": False if table.empty else "NOT_RUN" in set(table.astype(str).values.ravel()),
        "reference_only_count": int((table["status"] == "REFERENCE_ONLY").sum()) if len(table) else 0,
        "diagnostic_proxy_count": int((table["status"] == "DIAGNOSTIC_PROXY").sum()) if len(table) else 0,
        "diagnostic_only_count": int((table["status"] == "DIAGNOSTIC_ONLY").sum()) if len(table) else 0,
        "best_macro_f1_ablation": None,
        "best_macro_f1": None,
        "full_v6_macro_f1": None,
        "technical_rule_macro_f1": None,
        "forecast_claim_allowed": False,
        "counterfactual_claim_allowed": bool(counterfactual.get("claim_allowed", False)),
        "alpha_claim_allowed": bool(backtest.get("alpha_claim_allowed", False)),
    }
    if len(comparable):
        ranked = comparable.assign(_macro_f1=pd.to_numeric(comparable["macro_f1"], errors="coerce")).dropna(subset=["_macro_f1"])
        if len(ranked):
            ranked = ranked.sort_values("_macro_f1", ascending=False)
            metrics["best_macro_f1_ablation"] = str(ranked.iloc[0]["ablation"])
            metrics["best_macro_f1"] = float(ranked.iloc[0]["_macro_f1"])
    full_rows = table[table["ablation"].eq("Full_V6")] if len(table) else pd.DataFrame()
    tech_rows = table[table["ablation"].eq("Technical_Rule")] if len(table) else pd.DataFrame()
    if len(full_rows):
        metrics["full_v6_macro_f1"] = float(pd.to_numeric(full_rows.iloc[0].get("macro_f1"), errors="coerce"))
    if len(tech_rows):
        metrics["technical_rule_macro_f1"] = float(pd.to_numeric(tech_rows.iloc[0].get("macro_f1"), errors="coerce"))
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [
            args.contexts,
            args.predictions,
            args.baselines,
            args.backtest_metrics,
            args.counterfactual_metrics,
            args.alignment_metrics,
            args.output,
            args.metrics,
        ],
        STEP,
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.contexts, args.predictions, args.baselines],
        [args.output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
