from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.counterfactual_direction_rules_v6 import normalized_expected_direction
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "16_5_COUNTERFACTUAL_ELIGIBILITY_AUDIT_V6"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prediction_side_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    required = {"p_strong_down", "p_mild_down", "p_neutral", "p_mild_up", "p_strong_up"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"predictions missing probability columns: {missing}")
    out["up_side"] = out["p_mild_up"].astype(float) + out["p_strong_up"].astype(float)
    out["down_side"] = out["p_mild_down"].astype(float) + out["p_strong_down"].astype(float)
    out["signed_forecast_score"] = (
        -2.0 * out["p_strong_down"].astype(float)
        - out["p_mild_down"].astype(float)
        + out["p_mild_up"].astype(float)
        + 2.0 * out["p_strong_up"].astype(float)
    )
    out["schema_ok_bool"] = out["schema_ok"].astype(bool) if "schema_ok" in out.columns else True
    return out


def expected_side(expected_direction: str) -> str:
    normalized = normalized_expected_direction(expected_direction)
    if normalized == "up_decrease":
        return "up"
    if normalized == "down_decrease":
        return "down"
    return "unknown"


def side_signal_eligible(row: pd.Series, up_threshold: float, down_threshold: float) -> bool:
    if not bool(row.get("schema_ok_bool", False)):
        return False
    side = str(row.get("expected_side", "unknown"))
    if side == "up":
        return float(row.get("up_side", 0.0)) >= up_threshold
    if side == "down":
        return float(row.get("down_side", 0.0)) >= down_threshold
    return False


def aggregate_by_type(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "counterfactual_type",
                "tasks",
                "schema_ok_rate",
                "eligible_side_signal_rate",
                "mean_up_side",
                "mean_down_side",
                "zero_up_side_rate",
                "zero_down_side_rate",
            ]
        )
    grouped = df.groupby("counterfactual_type", dropna=False)
    out = grouped.agg(
        tasks=("sample_id", "count"),
        schema_ok_rate=("schema_ok_bool", "mean"),
        eligible_side_signal_rate=("eligible_side_signal", "mean"),
        mean_up_side=("up_side", "mean"),
        mean_down_side=("down_side", "mean"),
        median_up_side=("up_side", "median"),
        median_down_side=("down_side", "median"),
        zero_up_side_rate=("up_side_is_zero", "mean"),
        zero_down_side_rate=("down_side_is_zero", "mean"),
    ).reset_index()
    return out.sort_values("counterfactual_type").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/eval/current_v6_counterfactual_tasks.jsonl")
    parser.add_argument("--predictions", default="outputs/predictions/current_v6_dpo_predictions.parquet")
    parser.add_argument("--breakdown", default="outputs/tables/16_v6_counterfactual_breakdown.csv")
    parser.add_argument("--output", default="outputs/tables/16_5_v6_counterfactual_eligibility_by_type.csv")
    parser.add_argument("--task-output", default="outputs/tables/16_5_v6_counterfactual_task_eligibility.csv")
    parser.add_argument("--metrics", default="outputs/metrics/16_5_v6_counterfactual_eligibility_audit.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--up-threshold", type=float, default=0.25)
    parser.add_argument("--down-threshold", type=float, default=0.20)
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.tasks).exists():
        failures.append(f"tasks missing: {args.tasks}")
    if not Path(args.predictions).exists():
        failures.append(f"predictions missing: {args.predictions}")

    task_df = pd.DataFrame()
    by_type = pd.DataFrame()
    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    if not failures:
        tasks = read_jsonl(args.tasks)
        preds = prediction_side_metrics(pd.read_parquet(args.predictions))
        if not tasks:
            failures.append("counterfactual tasks are empty")
        if preds.empty:
            failures.append("predictions are empty")

    if not failures:
        pred_cols = [
            "sample_id",
            "schema_ok_bool",
            "pred_label",
            "action",
            "up_side",
            "down_side",
            "signed_forecast_score",
        ]
        task_df = pd.DataFrame(tasks)
        task_df["sample_id"] = task_df["sample_id"].astype(str)
        pred_subset = preds[pred_cols].copy()
        pred_subset["sample_id"] = pred_subset["sample_id"].astype(str)
        task_df = task_df.merge(pred_subset, on="sample_id", how="left")
        task_df["schema_ok_bool"] = task_df["schema_ok_bool"].fillna(False).astype(bool)
        task_df["up_side"] = task_df["up_side"].fillna(0.0).astype(float)
        task_df["down_side"] = task_df["down_side"].fillna(0.0).astype(float)
        task_df["signed_forecast_score"] = task_df["signed_forecast_score"].fillna(0.0).astype(float)
        task_df["expected_side"] = task_df["expected_direction"].apply(expected_side)
        task_df["eligible_side_signal"] = task_df.apply(
            lambda row: side_signal_eligible(row, args.up_threshold, args.down_threshold),
            axis=1,
        )
        task_df["up_side_is_zero"] = task_df["up_side"].abs() < 1e-12
        task_df["down_side_is_zero"] = task_df["down_side"].abs() < 1e-12
        by_type = aggregate_by_type(task_df)

        down_expected = task_df[task_df["expected_side"].eq("down")]
        up_expected = task_df[task_df["expected_side"].eq("up")]
        breakdown = pd.read_csv(args.breakdown) if Path(args.breakdown).exists() else pd.DataFrame()
        if not breakdown.empty and "counterfactual_type" in breakdown.columns:
            by_type = by_type.merge(
                breakdown[["counterfactual_type", "pass_rate", "no_change_rate", "wrong_direction_rate"]],
                on="counterfactual_type",
                how="left",
            )
        metrics = {
            "tasks": int(len(task_df)),
            "prediction_rows": int(len(preds)),
            "schema_ok_rate_on_tasks": float(task_df["schema_ok_bool"].mean()) if len(task_df) else 0.0,
            "eligible_side_signal_rate": float(task_df["eligible_side_signal"].mean()) if len(task_df) else 0.0,
            "up_expected_tasks": int(len(up_expected)),
            "down_expected_tasks": int(len(down_expected)),
            "up_expected_eligible_rate": float(up_expected["eligible_side_signal"].mean()) if len(up_expected) else 0.0,
            "down_expected_eligible_rate": float(down_expected["eligible_side_signal"].mean()) if len(down_expected) else 0.0,
            "down_expected_mean_down_side": float(down_expected["down_side"].mean()) if len(down_expected) else 0.0,
            "down_expected_zero_down_side_rate": float(down_expected["down_side_is_zero"].mean()) if len(down_expected) else 0.0,
            "up_expected_mean_up_side": float(up_expected["up_side"].mean()) if len(up_expected) else 0.0,
            "up_expected_zero_up_side_rate": float(up_expected["up_side_is_zero"].mean()) if len(up_expected) else 0.0,
            "up_threshold": float(args.up_threshold),
            "down_threshold": float(args.down_threshold),
            "claim_allowed": False,
            "pipeline_pass": True,
            "root_cause_summary": (
                "Counterfactual directional failures are partly confounded by original-side signal absence; "
                "down-decrease tasks have low DPO down-side probability mass."
            ),
        }
        if metrics["eligible_side_signal_rate"] >= 0.80:
            metrics["root_cause_summary"] = "Counterfactual tasks mostly have eligible original-side signal; failure likely reflects model insensitivity."

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    by_type.to_csv(args.output, index=False)
    Path(args.task_output).parent.mkdir(parents=True, exist_ok=True)
    task_df.to_csv(args.task_output, index=False)
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [args.tasks, args.predictions, args.breakdown, args.output, args.task_output, args.metrics],
        STEP,
        extra={
            "references": [
                "Ribeiro et al. 2020 CheckList directional expectation tests",
                "Gardner et al. 2020 contrast sets for local decision boundaries",
                "Kaushik et al. 2020 counterfactually augmented data",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.tasks, args.predictions, args.breakdown],
        [args.output, args.task_output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
