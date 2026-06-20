from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.backtest_daily_portfolio_v3 import action_to_position
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "16_MINIMUM_ABLATIONS_CURRENT_DATA"


def read_json(path: str) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prediction_metrics(name: str, path: str, contexts: pd.DataFrame, cost_bps: float) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    if not Path(path).exists():
        return {"ablation": name, "status": "MISSING", "artifact": path}, [f"{name} prediction missing: {path}"]
    preds = pd.read_parquet(path)
    if len(preds) == 0:
        return {"ablation": name, "status": "EMPTY", "artifact": path}, [f"{name} prediction empty"]
    if "split" in preds.columns and set(preds["split"].dropna()) != {"test"}:
        failures.append(f"{name} contains non-test rows")
    schema_ok_rate = float(preds["schema_ok"].astype(bool).mean()) if "schema_ok" in preds.columns else 0.0
    parse_ok_rate = float(preds["parse_ok"].astype(bool).mean()) if "parse_ok" in preds.columns else 0.0
    valid = preds[preds.get("schema_ok", False).astype(bool)].copy() if "schema_ok" in preds.columns else preds.copy()
    valid = valid.drop(columns=["event_date", "ticker"], errors="ignore")
    merged = valid.merge(contexts[["sample_id", "target_return", "target_label_5", "target_direction", "event_date"]], on="sample_id", how="inner")
    merged["position"] = merged["action"].apply(action_to_position) if "action" in merged.columns else merged["pred_label"].apply(action_to_position)
    merged["realized_utility"] = merged["position"] * merged["target_return"] - merged["position"].abs() * (cost_bps / 10000.0)
    target_sign = merged["target_direction"].map({"up": 1, "down": -1, "neutral": 0}).fillna(0)
    directional_accuracy = float((np.sign(merged["position"]) == target_sign).mean()) if len(merged) else 0.0
    non_hold_rate = float((merged["position"] != 0).mean()) if len(merged) else 0.0
    return {
        "ablation": name,
        "status": "PASS",
        "artifact": path,
        "rows": int(len(preds)),
        "merged_rows": int(len(merged)),
        "schema_ok_rate": schema_ok_rate,
        "parse_ok_rate": parse_ok_rate,
        "non_hold_rate": non_hold_rate,
        "directional_accuracy": directional_accuracy,
        "mean_realized_utility": float(merged["realized_utility"].mean()) if len(merged) else 0.0,
        "trading_days": int(pd.to_datetime(merged["event_date"]).dt.date.nunique()) if len(merged) else 0,
        "action_distribution": preds["action"].value_counts(dropna=False).to_dict() if "action" in preds.columns else {},
    }, failures


def no_flow_row(flow_eval: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    methods = flow_eval.get("methods", {})
    flow = methods.get("flow_reward_v3_1") or methods.get("flow_reward_v3") or {}
    proxy = methods.get("proxy_average_reward") or {}
    if not flow or not proxy:
        return {"ablation": "No_Flow_Reward", "status": "MISSING"}, ["No_Flow_Reward missing flow/proxy metrics"]
    return {
        "ablation": "No_Flow_Reward",
        "status": "PASS",
        "artifact": "outputs/metrics/flow_vs_proxy_v3_1_eval.json",
        "rows": int(flow_eval.get("eval_rows", 0)),
        "schema_ok_rate": None,
        "parse_ok_rate": None,
        "non_hold_rate": None,
        "directional_accuracy": None,
        "mean_realized_utility": None,
        "trading_days": None,
        "rank_corr_without_flow": proxy.get("rank_correlation_with_realized_utility"),
        "rank_corr_with_flow": flow.get("rank_correlation_with_realized_utility"),
        "preference_accuracy_without_flow": proxy.get("preference_pair_accuracy"),
        "preference_accuracy_with_flow": flow.get("preference_pair_accuracy"),
        "flow_claim_allowed": bool(flow_eval.get("flow_claim_allowed", False)),
    }, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="data/processed/ticker_date_contexts_h1_v2_targets.parquet")
    parser.add_argument("--full-predictions", required=True)
    parser.add_argument("--no-tech-predictions", required=True)
    parser.add_argument("--no-news-predictions", required=True)
    parser.add_argument("--sft-predictions", required=True)
    parser.add_argument("--flow-eval", default="outputs/metrics/flow_vs_proxy_v3_1_eval.json")
    parser.add_argument("--output-table", default="outputs/tables/ablation_current_v3.csv")
    parser.add_argument("--metrics", default="outputs/metrics/ablation_current_v3.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--cost-bps", type=float, default=7.0)
    parser.add_argument("--min-required", type=int, default=5)
    args = parser.parse_args()

    contexts = pd.read_parquet(args.contexts)
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    required = [
        ("Full_Current_V3", args.full_predictions),
        ("No_Technical_Tokens", args.no_tech_predictions),
        ("No_News_Body", args.no_news_predictions),
        ("SFT_Only", args.sft_predictions),
    ]
    for name, path in required:
        row, row_failures = prediction_metrics(name, path, contexts, args.cost_bps)
        rows.append(row)
        failures.extend(row_failures)
    row, row_failures = no_flow_row(read_json(args.flow_eval))
    rows.append(row)
    failures.extend(row_failures)

    table = pd.DataFrame(rows)
    if len(table) < args.min_required:
        failures.append(f"ablation rows {len(table)} < {args.min_required}")
    if (table["status"] != "PASS").any():
        failures.append("one or more required ablations did not PASS")
    if "NOT_RUN" in set(table["status"].astype(str)):
        failures.append("NOT_RUN rows are not allowed")

    Path(args.output_table).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_table, index=False)
    metrics = {
        "ablation_count": int(len(table)),
        "required_ablation_count": args.min_required,
        "status_counts": table["status"].value_counts(dropna=False).to_dict(),
        "required_ablations": [name for name, _ in required] + ["No_Flow_Reward"],
    }
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [
            args.full_predictions,
            args.no_tech_predictions,
            args.no_news_predictions,
            args.sft_predictions,
            args.flow_eval,
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
        [args.contexts, args.full_predictions, args.no_tech_predictions, args.no_news_predictions, args.sft_predictions, args.flow_eval],
        [args.output_table, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
