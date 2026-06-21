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

STEP = "17_BACKTEST_COUNTERFACTUAL_ABLATION_V4"


def read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--backtest-metrics", required=True)
    parser.add_argument("--daily-returns", required=True)
    parser.add_argument("--counterfactual-tasks", required=True)
    parser.add_argument("--counterfactual-metrics", required=True)
    parser.add_argument("--ablation-table", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default="outputs/manifests/17_BACKTEST_COUNTERFACTUAL_ABLATION_V4.manifest.json")
    parser.add_argument("--min-trading-days", type=int, default=10)
    args = parser.parse_args()

    failures: list[str] = []
    for path in [
        args.predictions,
        args.backtest_metrics,
        args.daily_returns,
        args.counterfactual_tasks,
        args.counterfactual_metrics,
        args.ablation_table,
    ]:
        if not os.path.exists(path):
            failures.append(f"missing artifact: {path}")

    pred_rows = 0
    schema_ok_rate = 0.0
    if os.path.exists(args.predictions):
        preds = pd.read_parquet(args.predictions)
        pred_rows = int(len(preds))
        if len(preds):
            if set(preds["split"].dropna()) != {"test"}:
                failures.append("prediction artifact contains non-test rows")
            schema_ok_rate = float(preds["schema_ok"].astype(bool).mean()) if "schema_ok" in preds.columns else 0.0
            if schema_ok_rate < 0.80:
                failures.append(f"prediction schema_ok_rate {schema_ok_rate:.4f} < 0.8000")
        else:
            failures.append("prediction artifact is empty")

    backtest = read_json(args.backtest_metrics)
    trading_days = int(backtest.get("num_trading_days", 0) or 0)
    if not backtest:
        failures.append("backtest metrics missing or empty")
    else:
        if trading_days < args.min_trading_days:
            failures.append(f"trading days {trading_days} < {args.min_trading_days}")
        if float(backtest.get("total_turnover", 0.0) or 0.0) <= 0:
            failures.append("backtest total turnover is zero")

    cf = read_json(args.counterfactual_metrics)
    if not cf:
        failures.append("counterfactual metrics missing or empty")
    else:
        if cf.get("no_change_rate") is None:
            failures.append("counterfactual no_change_rate missing")
        if cf.get("pass_rate") is None:
            failures.append("counterfactual pass_rate missing")
        news_keys = [key for key in cf if key.startswith("pass_rate_remove_")]
        if not news_keys:
            failures.append("news counterfactual pass rates missing")

    ablation_rows = 0
    if os.path.exists(args.ablation_table):
        ab = pd.read_csv(args.ablation_table)
        ablation_rows = int(len(ab))
        if ab.empty:
            failures.append("ablation table is empty")
        if "NOT_RUN" in set(ab.get("status", [])):
            failures.append("ablation table contains NOT_RUN")
        if len(ab) and not ab["status"].eq("PASS").all():
            failures.append("one or more ablation rows did not PASS")

    metrics = {
        "prediction_rows": pred_rows,
        "prediction_schema_ok_rate": schema_ok_rate,
        "num_trading_days": trading_days,
        "sharpe_daily_annualized": backtest.get("sharpe_daily_annualized"),
        "trading_alpha_claim_allowed": bool(backtest.get("alpha_claim_allowed", False)),
        "counterfactual_pass_rate": cf.get("pass_rate"),
        "counterfactual_no_change_rate": cf.get("no_change_rate"),
        "ablation_rows": ablation_rows,
        "pipeline_decision": "GO_SMALL" if not failures else "BLOCKED",
        "claim_notes": {
            "trading_alpha": "allowed" if backtest.get("alpha_claim_allowed", False) else "blocked",
            "counterfactual": "reported_small_scale",
            "ablation": "reported_no_not_run" if ablation_rows else "blocked",
        },
    }
    write_json(args.output_metrics, metrics)
    write_manifest(
        args.manifest,
        [
            args.predictions,
            args.backtest_metrics,
            args.daily_returns,
            args.counterfactual_tasks,
            args.counterfactual_metrics,
            args.ablation_table,
            args.output_metrics,
        ],
        STEP,
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [
            args.predictions,
            args.backtest_metrics,
            args.daily_returns,
            args.counterfactual_tasks,
            args.counterfactual_metrics,
            args.ablation_table,
        ],
        [args.output_metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
