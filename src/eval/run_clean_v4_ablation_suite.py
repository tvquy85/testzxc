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

STEP = "17_ABLATION_CURRENT_CLEAN_V4"


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
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--full-predictions", required=True)
    parser.add_argument("--full-backtest", required=True)
    parser.add_argument("--no-news-predictions", required=True)
    parser.add_argument("--no-news-backtest", required=True)
    parser.add_argument("--no-tech-predictions", required=True)
    parser.add_argument("--no-tech-backtest", required=True)
    parser.add_argument("--sft-predictions", required=True)
    parser.add_argument("--sft-backtest", required=True)
    parser.add_argument("--flow-metrics", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default="outputs/manifests/17_ABLATION_CURRENT_CLEAN_V4.manifest.json")
    args = parser.parse_args()

    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    rows = [
        row_for("Full_Clean_V4", args.full_predictions, args.full_backtest),
        row_for("No_News_Evidence", args.no_news_predictions, args.no_news_backtest),
        row_for("No_Technical_Tokens", args.no_tech_predictions, args.no_tech_backtest),
        row_for("SFT_Only_or_Base_Model", args.sft_predictions, args.sft_backtest),
    ]
    if not contexts.empty:
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
    flow = read_json(args.flow_metrics)
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
    missing = table[~table["status"].eq("PASS")]
    if len(missing):
        failures.append(f"ablation rows not PASS: {missing['ablation'].tolist()}")
    metrics = {
        "ablation_count": int(len(table)),
        "pass_count": int(table["status"].eq("PASS").sum()) if len(table) else 0,
        "not_run_count": int(table["status"].eq("NOT_RUN").sum()) if len(table) and "status" in table else 0,
        "missing_count": int((~table["status"].eq("PASS")).sum()) if len(table) else 0,
        "required_ablations": table["ablation"].tolist() if len(table) else [],
    }
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [
            args.full_predictions,
            args.full_backtest,
            args.no_news_predictions,
            args.no_news_backtest,
            args.no_tech_predictions,
            args.no_tech_backtest,
            args.sft_predictions,
            args.sft_backtest,
            args.flow_metrics,
            args.output_table,
            args.metrics,
        ],
        STEP,
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.contexts, args.flow_metrics],
        [args.output_table, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
