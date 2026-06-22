from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baselines.run_v6_comparable_baselines import (
    LABELS,
    label_from_score,
    metric_row,
    normalize_label,
    prediction_metric_row,
    technical_score,
)
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "17_8_REPAIRED_FORECAST_BASELINE_PROBE_V6"


def row_by_method(table: pd.DataFrame, method: str) -> dict[str, Any]:
    rows = table[table["method"].astype(str).eq(method)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def metric_value(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="data/processed/current_v6_prediction_contexts.parquet")
    parser.add_argument("--dpo-original", default="outputs/predictions/current_v6_dpo_predictions.parquet")
    parser.add_argument("--dpo-repaired", default="outputs/predictions/current_v6_dpo_predictions_repaired.parquet")
    parser.add_argument("--rwsft-repaired", default="outputs/predictions/current_v6_rwsft_predictions_repaired.parquet")
    parser.add_argument("--output", default="outputs/tables/17_8_v6_repaired_forecast_baseline_probe.csv")
    parser.add_argument("--metrics", default="outputs/metrics/17_8_v6_repaired_forecast_baseline_probe.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--label-col", default="target_label_5")
    args = parser.parse_args()

    failures: list[str] = []
    required_paths = [args.contexts, args.dpo_original, args.dpo_repaired, args.rwsft_repaired]
    for path in required_paths:
        if not Path(path).exists():
            failures.append(f"missing input: {path}")

    table = pd.DataFrame()
    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    if not failures:
        contexts = pd.read_parquet(args.contexts)
        dpo_original = pd.read_parquet(args.dpo_original)
        dpo_repaired = pd.read_parquet(args.dpo_repaired)
        rwsft_repaired = pd.read_parquet(args.rwsft_repaired)
        if args.label_col not in contexts.columns:
            failures.append(f"contexts missing label column: {args.label_col}")
        else:
            test_contexts = contexts[contexts["split"].eq("test")].copy() if "split" in contexts.columns else contexts.copy()
            shared_ids = (
                set(dpo_original["sample_id"].astype(str))
                & set(dpo_repaired["sample_id"].astype(str))
                & set(rwsft_repaired["sample_id"].astype(str))
            )
            eval_contexts = test_contexts[test_contexts["sample_id"].astype(str).isin(shared_ids)].drop_duplicates("sample_id").copy()
            if eval_contexts.empty:
                failures.append("no shared rows for repaired baseline probe")
            else:
                y_true = eval_contexts[args.label_col].map(normalize_label).tolist()
                tech_scores = eval_contexts.apply(technical_score, axis=1)
                rows = [
                    metric_row(
                        "Technical_Rule",
                        y_true,
                        [label_from_score(score) for score in tech_scores],
                        "same deterministic technical-token rule used in Step 17",
                    ),
                    prediction_metric_row(
                        "Qwen_DPO_V6_Original",
                        dpo_original,
                        eval_contexts,
                        args.label_col,
                        "original Step 14 DPO prediction artifact before probability-sum repair",
                    ),
                    prediction_metric_row(
                        "Qwen_DPO_V6_Repaired",
                        dpo_repaired,
                        eval_contexts,
                        args.label_col,
                        "Step 14.6 schema-repaired DPO probabilities; no target labels used in repair",
                    ),
                    prediction_metric_row(
                        "Qwen_RWSFT_V6_Repaired",
                        rwsft_repaired,
                        eval_contexts,
                        args.label_col,
                        "Step 14.6 RWSFT pass-through repair artifact",
                    ),
                ]
                table = pd.DataFrame(rows)
                dpo_orig = row_by_method(table, "Qwen_DPO_V6_Original")
                dpo_rep = row_by_method(table, "Qwen_DPO_V6_Repaired")
                rwsft = row_by_method(table, "Qwen_RWSFT_V6_Repaired")
                tech = row_by_method(table, "Technical_Rule")
                metrics = {
                    "pipeline_pass": True,
                    "claim_allowed": False,
                    "diagnostic_only": True,
                    "evaluation_rows": int(len(eval_contexts)),
                    "dpo_original_macro_f1": metric_value(dpo_orig, "macro_f1"),
                    "dpo_original_mcc": metric_value(dpo_orig, "mcc"),
                    "dpo_repaired_macro_f1": metric_value(dpo_rep, "macro_f1"),
                    "dpo_repaired_mcc": metric_value(dpo_rep, "mcc"),
                    "rwsft_repaired_macro_f1": metric_value(rwsft, "macro_f1"),
                    "rwsft_repaired_mcc": metric_value(rwsft, "mcc"),
                    "technical_rule_macro_f1": metric_value(tech, "macro_f1"),
                    "technical_rule_mcc": metric_value(tech, "mcc"),
                    "dpo_repair_delta_macro_f1": metric_value(dpo_rep, "macro_f1") - metric_value(dpo_orig, "macro_f1"),
                    "dpo_repair_delta_mcc": metric_value(dpo_rep, "mcc") - metric_value(dpo_orig, "mcc"),
                    "dpo_repaired_beats_rwsft_macro_f1": metric_value(dpo_rep, "macro_f1") > metric_value(rwsft, "macro_f1"),
                    "dpo_repaired_beats_rwsft_mcc": metric_value(dpo_rep, "mcc") > metric_value(rwsft, "mcc"),
                    "dpo_repaired_beats_technical_macro_f1": metric_value(dpo_rep, "macro_f1") > metric_value(tech, "macro_f1"),
                    "dpo_repaired_beats_technical_mcc": metric_value(dpo_rep, "mcc") > metric_value(tech, "mcc"),
                    "promotion_boundary": (
                        "diagnostic only; promoting repaired predictions requires rerunning downstream baseline, "
                        "backtest, counterfactual, ablation, statistical, and strict-gate artifacts consistently"
                    ),
                }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    metrics["pipeline_pass"] = not failures and not table.empty
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [*required_paths, args.output, args.metrics], STEP)
    status = "PASS" if metrics["pipeline_pass"] else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        required_paths,
        [args.output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
