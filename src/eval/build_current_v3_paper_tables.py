from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import artifact_entry, write_json, write_manifest, write_status

STEP = "17_PAPER_TABLES_NEGATIVE_RESULTS_GATES"


def read_json(path: str) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def require_metric(metrics: dict[str, Any], key: str, source: str, failures: list[str]) -> Any:
    value = metrics.get(key)
    if value is None:
        failures.append(f"missing metric {key} in {source}")
    return value


def write_table(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-metrics", required=True)
    parser.add_argument("--grounding-metrics", required=True)
    parser.add_argument("--debias-metrics", required=True)
    parser.add_argument("--flow-eval", required=True)
    parser.add_argument("--backtest-metrics", required=True)
    parser.add_argument("--counterfactual-metrics", required=True)
    parser.add_argument("--ablation-table", required=True)
    parser.add_argument("--prediction-table", default="outputs/tables/current_v3_table_prediction.csv")
    parser.add_argument("--explanation-table", default="outputs/tables/current_v3_table_explanation.csv")
    parser.add_argument("--flow-table", default="outputs/tables/current_v3_table_flow_reward.csv")
    parser.add_argument("--backtest-table", default="outputs/tables/current_v3_table_backtest.csv")
    parser.add_argument("--ablation-output-table", default="outputs/tables/current_v3_table_ablation.csv")
    parser.add_argument("--claim-matrix", default="outputs/metrics/current_v3_claim_matrix.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    pred = read_json(args.prediction_metrics)
    grounding = read_json(args.grounding_metrics)
    debias = read_json(args.debias_metrics)
    flow = read_json(args.flow_eval)
    backtest = read_json(args.backtest_metrics)
    cf = read_json(args.counterfactual_metrics)
    for source, metrics in [
        (args.prediction_metrics, pred),
        (args.grounding_metrics, grounding),
        (args.debias_metrics, debias),
        (args.flow_eval, flow),
        (args.backtest_metrics, backtest),
        (args.counterfactual_metrics, cf),
    ]:
        if not metrics:
            failures.append(f"missing or empty metrics source: {source}")
    if not Path(args.ablation_table).exists():
        failures.append(f"missing ablation table: {args.ablation_table}")
        ablation_df = pd.DataFrame()
    else:
        ablation_df = pd.read_csv(args.ablation_table)

    prediction_rows = [
        {
            "model": "Current_V3_1_DPO",
            "split": pred.get("split", "test"),
            "rows": require_metric(pred, "rows", args.prediction_metrics, failures),
            "schema_ok_rate": require_metric(pred, "schema_ok_rate", args.prediction_metrics, failures),
            "parse_ok_rate": require_metric(pred, "parse_ok_rate", args.prediction_metrics, failures),
            "trading_days": pred.get("selected_trading_days"),
            "action_distribution": json.dumps(pred.get("action_distribution", {}), sort_keys=True),
            "claim_allowed": False,
            "claim_block_reason": "small-scale pipeline validation only",
            "source_file": args.prediction_metrics,
        }
    ]
    explanation_rows = [
        {
            "component": "grounding_v3_1",
            "total_evaluated": require_metric(grounding, "total_evaluated", args.grounding_metrics, failures),
            "supported_rate": require_metric(grounding, "supported_rate", args.grounding_metrics, failures),
            "unverified_rate": require_metric(grounding, "unverified_rate", args.grounding_metrics, failures),
            "not_applicable_rate": require_metric(grounding, "not_applicable_rate", args.grounding_metrics, failures),
            "debias_argmax_consistency": debias.get("argmax_consistency"),
            "debias_reward_source_allowed": debias.get("debias_reward_source_allowed"),
            "claim_allowed": False,
            "claim_block_reason": "grounding/debias are quality gates, not final paper claims at small scale",
            "source_file": f"{args.grounding_metrics};{args.debias_metrics}",
        }
    ]
    flow_methods = flow.get("methods", {})
    flow_row = flow_methods.get("flow_reward_v3_1") or flow_methods.get("flow_reward_v3") or {}
    proxy_row = flow_methods.get("proxy_average_reward") or {}
    flow_rows = [
        {
            "method": "flow_reward_v3_1",
            "rank_correlation_with_realized_utility": flow_row.get("rank_correlation_with_realized_utility"),
            "preference_pair_accuracy": flow_row.get("preference_pair_accuracy"),
            "top_decile_realized_utility": flow_row.get("top_decile_realized_utility"),
            "claim_allowed": bool(flow.get("flow_claim_allowed", False)),
            "claim_block_reason": "" if flow.get("flow_claim_allowed", False) else "did not beat proxy average on at least 2 pre-specified metrics",
            "source_file": args.flow_eval,
        },
        {
            "method": "proxy_average_reward",
            "rank_correlation_with_realized_utility": proxy_row.get("rank_correlation_with_realized_utility"),
            "preference_pair_accuracy": proxy_row.get("preference_pair_accuracy"),
            "top_decile_realized_utility": proxy_row.get("top_decile_realized_utility"),
            "claim_allowed": False,
            "claim_block_reason": "baseline comparator",
            "source_file": args.flow_eval,
        },
    ]
    if any(row["rank_correlation_with_realized_utility"] is None for row in flow_rows):
        failures.append("flow table missing rank correlation metrics")

    alpha_allowed = bool(backtest.get("alpha_claim_allowed", False))
    backtest_rows = [
        {
            "strategy": "Current_V3_1_DPO_Daily_Portfolio",
            "num_trading_days": require_metric(backtest, "num_trading_days", args.backtest_metrics, failures),
            "sharpe_daily_annualized": require_metric(backtest, "sharpe_daily_annualized", args.backtest_metrics, failures),
            "sortino_daily_annualized": backtest.get("sortino_daily_annualized"),
            "max_drawdown": backtest.get("max_drawdown"),
            "mean_daily_return": backtest.get("mean_daily_return"),
            "claim_allowed": alpha_allowed,
            "claim_block_reason": "" if alpha_allowed else "Sharpe/coverage/sample-size gate not sufficient for alpha claim",
            "source_file": args.backtest_metrics,
        },
        {
            "strategy": "Current_V3_1_Counterfactual",
            "num_trading_days": None,
            "sharpe_daily_annualized": None,
            "sortino_daily_annualized": None,
            "max_drawdown": None,
            "mean_daily_return": None,
            "counterfactual_pass_rate": require_metric(cf, "pass_rate", args.counterfactual_metrics, failures),
            "counterfactual_no_change_rate": require_metric(cf, "no_change_rate", args.counterfactual_metrics, failures),
            "claim_allowed": bool((cf.get("pass_rate") or 0) > 0.16 or (cf.get("no_change_rate") or 1.0) < 0.696),
            "claim_block_reason": "" if bool((cf.get("pass_rate") or 0) > 0.16 or (cf.get("no_change_rate") or 1.0) < 0.696) else "counterfactual gate not reached",
            "source_file": args.counterfactual_metrics,
        },
    ]
    if not ablation_df.empty:
        ablation_df = ablation_df.copy()
        ablation_df["claim_allowed"] = False
        ablation_df["claim_block_reason"] = "small-scale ablation, not final paper evidence"

    write_table(args.prediction_table, prediction_rows)
    write_table(args.explanation_table, explanation_rows)
    write_table(args.flow_table, flow_rows)
    write_table(args.backtest_table, backtest_rows)
    Path(args.ablation_output_table).parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(args.ablation_output_table, index=False)

    claim_matrix = {
        "prediction_pipeline_claim_allowed": False,
        "flow_reward_claim_allowed": bool(flow.get("flow_claim_allowed", False)),
        "trading_alpha_claim_allowed": alpha_allowed,
        "counterfactual_faithfulness_claim_allowed": bool((cf.get("pass_rate") or 0) > 0.16 or (cf.get("no_change_rate") or 1.0) < 0.696),
        "reproducibility_claim_allowed": not failures,
        "blocked_claims": [],
        "source_files": {
            "prediction": artifact_entry(args.prediction_metrics, STEP),
            "grounding": artifact_entry(args.grounding_metrics, STEP),
            "debias": artifact_entry(args.debias_metrics, STEP),
            "flow": artifact_entry(args.flow_eval, STEP),
            "backtest": artifact_entry(args.backtest_metrics, STEP),
            "counterfactual": artifact_entry(args.counterfactual_metrics, STEP),
            "ablation": artifact_entry(args.ablation_table, STEP),
        },
    }
    if not claim_matrix["flow_reward_claim_allowed"]:
        claim_matrix["blocked_claims"].append({"claim": "flow_reward_improvement", "reason": "flow did not beat proxy on >=2 metrics"})
    if not claim_matrix["trading_alpha_claim_allowed"]:
        claim_matrix["blocked_claims"].append({"claim": "trading_alpha", "reason": "backtest alpha gate not reached"})
    if not claim_matrix["counterfactual_faithfulness_claim_allowed"]:
        claim_matrix["blocked_claims"].append({"claim": "counterfactual_faithfulness", "reason": "counterfactual directional gate not reached"})
    write_json(args.claim_matrix, claim_matrix)

    outputs = [
        args.prediction_table,
        args.explanation_table,
        args.flow_table,
        args.backtest_table,
        args.ablation_output_table,
        args.claim_matrix,
    ]
    write_manifest(args.manifest, outputs, STEP)
    metrics = {
        "tables_created": outputs[:-1],
        "claim_matrix": args.claim_matrix,
        "claim_allowed_count": int(sum(1 for key, value in claim_matrix.items() if key.endswith("_claim_allowed") and value)),
    }
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.prediction_metrics, args.grounding_metrics, args.debias_metrics, args.flow_eval, args.backtest_metrics, args.counterfactual_metrics, args.ablation_table], [*outputs, args.manifest, args.status], metrics, failures, status == "PASS")
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
