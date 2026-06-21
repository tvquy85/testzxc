from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "19_ABLATION_SUITE_MEDIUM"


def read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prediction_stats(path: str, contexts: pd.DataFrame | None = None, mask: pd.Series | None = None) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"status": "MISSING", "rows": 0}
    df = pd.read_parquet(path)
    if contexts is not None and mask is not None and len(df):
        keep_ids = set(contexts.loc[mask, "sample_id"].astype(str))
        df = df[df["sample_id"].astype(str).isin(keep_ids)].copy()
    if df.empty:
        return {"status": "EMPTY", "rows": 0}
    schema = df["schema_ok"].astype(bool) if "schema_ok" in df.columns else pd.Series(False, index=df.index)
    action = df.get("action", pd.Series("missing", index=df.index)).astype(str)
    return {
        "status": "PASS",
        "rows": int(len(df)),
        "schema_ok_rate": float(schema.mean()),
        "non_hold_rate": float((action[schema] != "hold").mean()) if int(schema.sum()) else 0.0,
        "action_distribution": action.value_counts(dropna=False).to_dict(),
    }


def backtest_stats(path: str) -> dict[str, Any]:
    metrics = read_json(path)
    if not metrics:
        return {"backtest_status": "MISSING"}
    return {
        "backtest_status": "PASS" if metrics.get("pipeline_pass", True) else "FAIL",
        "num_trading_days": metrics.get("num_trading_days"),
        "sharpe_daily_annualized": metrics.get("sharpe_daily_annualized"),
        "mean_daily_return": metrics.get("mean_daily_return"),
        "total_turnover": metrics.get("total_turnover"),
        "alpha_claim_allowed": metrics.get("alpha_claim_allowed"),
    }


def row_for(name: str, pred_path: str, backtest_path: str, contexts: pd.DataFrame | None = None, mask: pd.Series | None = None) -> dict[str, Any]:
    pred = prediction_stats(pred_path, contexts, mask)
    bt = backtest_stats(backtest_path)
    status = "PASS" if pred.get("status") == "PASS" and bt.get("backtest_status") in {"PASS", "FAIL"} else "MISSING"
    block = ""
    if status != "PASS":
        block = "missing_or_empty_prediction_or_backtest"
    elif bt.get("sharpe_daily_annualized") is not None and float(bt.get("sharpe_daily_annualized") or 0.0) <= 0:
        block = "non_positive_sharpe"
    return {
        "ablation": name,
        "status": status,
        "ablation_type": "small_inference_ablation",
        **pred,
        **bt,
        "claim_allowed": status == "PASS" and not block,
        "claim_block_reason": block,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default=None)
    parser.add_argument("--full-predictions", default=None)
    parser.add_argument("--full-backtest", default=None)
    parser.add_argument("--no-news-predictions", default=None)
    parser.add_argument("--no-news-backtest", default=None)
    parser.add_argument("--no-tech-predictions", default=None)
    parser.add_argument("--no-tech-backtest", default=None)
    parser.add_argument("--sft-predictions", default=None)
    parser.add_argument("--sft-backtest", default=None)
    parser.add_argument("--flow-metrics", default=None)
    parser.add_argument("--baselines", default=None)
    parser.add_argument("--backtest", default=None)
    parser.add_argument("--counterfactual", default=None)
    parser.add_argument("--output-table", "--output", dest="output_table", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default="outputs/manifests/17_ABLATION_CURRENT_CLEAN_V4.manifest.json")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    if args.baselines or args.backtest or args.counterfactual:
        baseline_table = pd.read_csv(args.baselines) if args.baselines and Path(args.baselines).exists() else pd.DataFrame()
        backtest = read_json(args.backtest or "")
        counterfactual = read_json(args.counterfactual or "")
        for _, row in baseline_table.iterrows():
            name = str(row.get("baseline") or row.get("model") or row.get("ablation") or "baseline")
            rows.append(
                {
                    "ablation": name,
                    "status": "PASS" if str(row.get("status", "PASS")) != "NOT_RUN" else "MISSING",
                    "ablation_type": "medium_reference_baseline",
                    "rows": row.get("rows"),
                    "schema_ok_rate": row.get("schema_ok_rate"),
                    "non_hold_rate": row.get("non_hold_rate"),
                    "macro_f1": row.get("macro_f1"),
                    "accuracy": row.get("accuracy"),
                    "backtest_status": None,
                    "num_trading_days": None,
                    "sharpe_daily_annualized": None,
                    "mean_daily_return": None,
                    "total_turnover": None,
                    "counterfactual_pass_rate": None,
                    "claim_allowed": False,
                    "claim_block_reason": "medium_reference_not_final_claim",
                }
            )
        if len(baseline_table):
            lookup = {
                str(row.get("baseline") or row.get("method") or row.get("ablation")): row
                for _, row in baseline_table.iterrows()
            }
            mapped = [
                ("No_News_Evidence", "Technical_Rule", "technical-only reference removes news evidence"),
                ("No_Technical_Tokens", "Text_News_Heuristic", "text-only reference removes technical tokens"),
                ("No_Flow_or_Proxy", "News_Technical_Heuristic", "heuristic proxy without learned flow reward"),
                ("RWSFT_Only_or_SFT_Reference", "Qwen_RWSFT_Medium", "medium RWSFT adapter before DPO preference optimization"),
                ("Technical_Only", "Technical_Rule", "technical-token contribution reference"),
            ]
            for ablation_name, source_name, note in mapped:
                source = lookup.get(source_name)
                if source is None:
                    continue
                rows.append(
                    {
                        "ablation": ablation_name,
                        "status": "PASS" if str(source.get("status", "PASS")) != "NOT_RUN" else "MISSING",
                        "ablation_type": "medium_reference_ablation",
                        "rows": source.get("rows"),
                        "schema_ok_rate": source.get("schema_ok_rate"),
                        "non_hold_rate": source.get("non_hold_rate"),
                        "macro_f1": source.get("macro_f1"),
                        "accuracy": source.get("accuracy"),
                        "backtest_status": None,
                        "num_trading_days": None,
                        "sharpe_daily_annualized": None,
                        "mean_daily_return": None,
                        "total_turnover": None,
                        "counterfactual_pass_rate": None,
                        "claim_allowed": False,
                        "claim_block_reason": note,
                    }
                )
        rows.append(
            {
                "ablation": "Full_Medium_Clean_V4",
                "status": "PASS" if backtest else "MISSING",
                "ablation_type": "official_medium_backtest",
                "rows": None,
                "schema_ok_rate": backtest.get("schema_ok_rate"),
                "non_hold_rate": None,
                "backtest_status": "PASS" if backtest.get("pipeline_pass") else "FAIL",
                "num_trading_days": backtest.get("num_trading_days"),
                "sharpe_daily_annualized": backtest.get("sharpe_daily_annualized") or backtest.get("sharpe_annualized"),
                "mean_daily_return": backtest.get("mean_daily_return"),
                "total_turnover": backtest.get("total_turnover"),
                "counterfactual_pass_rate": counterfactual.get("pass_rate"),
                "claim_allowed": bool(backtest.get("alpha_claim_allowed") and counterfactual.get("claim_allowed")),
                "claim_block_reason": "" if backtest.get("alpha_claim_allowed") and counterfactual.get("claim_allowed") else "medium_claim_gate_restricted",
            }
        )
        flow = read_json(args.flow_metrics or "")
        if flow:
            rows.append(
                {
                    "ablation": "No_Flow_Reward",
                    "status": "PASS",
                    "ablation_type": "reward_selection_ablation",
                    "rows": flow.get("rows", 0),
                    "schema_ok_rate": None,
                    "non_hold_rate": None,
                    "backtest_status": None,
                    "num_trading_days": None,
                    "sharpe_daily_annualized": None,
                    "mean_daily_return": None,
                    "total_turnover": None,
                    "counterfactual_pass_rate": None,
                    "claim_allowed": bool(flow.get("flow_reward_improvement") or flow.get("flow_claim_allowed")),
                    "claim_block_reason": "" if flow.get("flow_reward_improvement") or flow.get("flow_claim_allowed") else "flow_did_not_beat_proxy_2_of_3",
                }
            )
    else:
        contexts = pd.read_parquet(args.contexts) if args.contexts and Path(args.contexts).exists() else pd.DataFrame()
        required = [
            args.full_predictions,
            args.full_backtest,
            args.no_news_predictions,
            args.no_news_backtest,
            args.no_tech_predictions,
            args.no_tech_backtest,
            args.sft_predictions,
            args.sft_backtest,
        ]
        if not all(required):
            rows = []
        else:
            rows = [
                row_for("Full_Clean_V4", args.full_predictions, args.full_backtest),
                row_for("No_News_Evidence", args.no_news_predictions, args.no_news_backtest),
                row_for("No_Technical_Tokens", args.no_tech_predictions, args.no_tech_backtest),
                row_for("SFT_Only_or_Base_Model", args.sft_predictions, args.sft_backtest),
            ]
        if not contexts.empty and args.full_predictions and args.full_backtest:
            test = contexts[contexts["split"].eq("test")].copy()
            rows.append(
                row_for(
                    "News_Technical_Track",
                    args.full_predictions,
                    args.full_backtest,
                    test,
                    test.get("has_company_event_news", pd.Series(False, index=test.index)).astype(bool),
                )
            )
            rows.append(
                row_for(
                    "Technical_Only_Track",
                    args.full_predictions,
                    args.full_backtest,
                    test,
                    ~test.get("has_company_event_news", pd.Series(False, index=test.index)).astype(bool),
                )
            )
        flow = read_json(args.flow_metrics or "")
        rows.append(
            {
                "ablation": "No_Flow_Reward",
                "status": "PASS" if flow else "MISSING",
                "ablation_type": "reward_selection_ablation",
                "rows": flow.get("rows", 0),
                "schema_ok_rate": None,
                "non_hold_rate": None,
                "backtest_status": None,
                "num_trading_days": None,
                "sharpe_daily_annualized": None,
                "mean_daily_return": None,
                "total_turnover": None,
                "alpha_claim_allowed": None,
                "claim_allowed": bool(flow.get("flow_reward_improvement", False)),
                "claim_block_reason": "" if flow.get("flow_reward_improvement", False) else "flow_did_not_beat_proxy_2_of_3",
            }
        )
    table = pd.DataFrame(rows)
    Path(args.output_table).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_table, index=False)

    failures: list[str] = []
    if table.empty:
        failures.append("ablation table is empty")
    if "NOT_RUN" in set(table.get("status", [])):
        failures.append("ablation table contains NOT_RUN")
    required_names = {
        "Full_Medium_Clean_V4",
        "No_News_Evidence",
        "No_Technical_Tokens",
        "No_Flow_or_Proxy",
        "RWSFT_Only_or_SFT_Reference",
        "Technical_Only",
    }
    present_required = required_names & set(table["ablation"].astype(str)) if len(table) else set()
    all_required = required_names <= present_required
    if not all_required and (args.baselines or args.backtest or args.counterfactual):
        failures.append(f"missing required medium ablations: {sorted(required_names - present_required)}")
    missing = table[~table["status"].eq("PASS")]
    if len(missing):
        failures.append(f"ablation rows not PASS: {missing['ablation'].tolist()}")
    metrics = {
        "ablation_count": int(len(table)),
        "pass_count": int(table["status"].eq("PASS").sum()) if len(table) else 0,
        "not_run_count": int(table["status"].eq("NOT_RUN").sum()) if len(table) and "status" in table else 0,
        "missing_count": int((~table["status"].eq("PASS")).sum()) if len(table) else 0,
        "required_ablations": table["ablation"].tolist() if len(table) else [],
        "all_required_ablations_present": bool(all_required),
    }
    write_json(args.metrics, metrics)
    manifest_paths = [
        path
        for path in [
            args.full_predictions,
            args.full_backtest,
            args.no_news_predictions,
            args.no_news_backtest,
            args.no_tech_predictions,
            args.no_tech_backtest,
            args.sft_predictions,
            args.sft_backtest,
            args.flow_metrics,
            args.baselines,
            args.backtest,
            args.counterfactual,
            args.output_table,
            args.metrics,
        ]
        if path
    ]
    write_manifest(args.manifest, manifest_paths, STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [path for path in [args.contexts, args.flow_metrics, args.baselines, args.backtest, args.counterfactual] if path],
        [args.output_table, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
