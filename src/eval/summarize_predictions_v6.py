from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "14_PREDICT_WITH_V6_ADAPTERS"
LABELS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def metric_row(name: str, path: str, contexts: pd.DataFrame, min_rows: int, min_schema_ok_rate: float) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    p = Path(path)
    if not p.exists():
        return {"model": name, "path": path, "rows": 0, "status": "MISSING"}, [f"{name} predictions missing: {path}"]
    try:
        preds = pd.read_parquet(p)
    except Exception as exc:
        return {"model": name, "path": path, "rows": 0, "status": "INVALID"}, [f"{name} predictions unreadable: {type(exc).__name__}: {str(exc)[:200]}"]
    row: dict[str, Any] = {"model": name, "path": path, "rows": int(len(preds)), "status": "RUN"}
    if len(preds) < min_rows:
        failures.append(f"{name} prediction rows {len(preds)} < {min_rows}")
    if len(preds) == 0:
        row.update({"schema_ok_rate": 0.0, "macro_f1": None, "mcc": None, "accuracy": None})
        return row, failures
    if "split" not in preds.columns or set(preds["split"].dropna()) != {"test"}:
        failures.append(f"{name} predictions contain non-test rows")
    if "schema_ok" not in preds.columns:
        failures.append(f"{name} predictions missing schema_ok")
        schema_ok_rate = 0.0
    else:
        schema_ok_rate = float(preds["schema_ok"].mean())
        if schema_ok_rate < min_schema_ok_rate:
            failures.append(f"{name} schema_ok_rate {schema_ok_rate:.4f} < {min_schema_ok_rate:.4f}")
    row["schema_ok_rate"] = schema_ok_rate
    valid = preds[preds.get("schema_ok", False).astype(bool)].copy() if "schema_ok" in preds.columns else preds.iloc[0:0].copy()
    if valid.empty:
        failures.append(f"{name} has no schema-valid predictions")
        row.update({"macro_f1": None, "mcc": None, "accuracy": None})
        return row, failures
    merged = valid.merge(contexts[["sample_id", "target_label_5"]], on="sample_id", how="inner")
    row["merged_rows"] = int(len(merged))
    if merged.empty:
        failures.append(f"{name} has no merged target labels")
        row.update({"macro_f1": None, "mcc": None, "accuracy": None})
        return row, failures
    try:
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

        row["macro_f1"] = finite_float(f1_score(merged["target_label_5"], merged["pred_label"], labels=LABELS, average="macro", zero_division=0))
        row["mcc"] = finite_float(matthews_corrcoef(merged["target_label_5"], merged["pred_label"]))
        row["accuracy"] = finite_float(accuracy_score(merged["target_label_5"], merged["pred_label"]))
    except Exception as exc:
        failures.append(f"{name} metric computation failed: {type(exc).__name__}: {str(exc)[:200]}")
        row.update({"macro_f1": None, "mcc": None, "accuracy": None})
    row["pred_label_distribution"] = preds["pred_label"].value_counts(dropna=False).to_dict() if "pred_label" in preds.columns else {}
    row["action_distribution"] = preds["action"].value_counts(dropna=False).to_dict() if "action" in preds.columns else {}
    row["checkpoint_paths"] = sorted(set(str(x) for x in preds.get("model_checkpoint", pd.Series(dtype=str)).dropna()))
    return row, failures


def child_status_failures(name: str, path: str | None) -> tuple[dict[str, Any] | None, list[str]]:
    if not path:
        return None, []
    p = Path(path)
    if not p.exists():
        return None, [f"{name} child status missing: {path}"]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"{name} child status unreadable: {type(exc).__name__}: {str(exc)[:200]}"]
    failures = [f"{name} child status is {data.get('status')}"] if data.get("status") != "PASS" else []
    failures.extend(f"{name} child failure: {failure}" for failure in data.get("failures", []))
    return data, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--rwsft", required=True)
    parser.add_argument("--contexts", default="data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--dpo-status", default=None)
    parser.add_argument("--rwsft-status", default=None)
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.90)
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.contexts).exists():
        failures.append(f"contexts missing: {args.contexts}")
        contexts = pd.DataFrame(columns=["sample_id", "target_label_5"])
    else:
        contexts = pd.read_parquet(args.contexts)
    rows: list[dict[str, Any]] = []
    for name, path in [("DPO", args.dpo), ("RWSFT", args.rwsft)]:
        row, row_failures = metric_row(name, path, contexts, args.min_rows, args.min_schema_ok_rate)
        rows.append(row)
        failures.extend(row_failures)
    dpo_child, dpo_child_failures = child_status_failures("DPO", args.dpo_status)
    rwsft_child, rwsft_child_failures = child_status_failures("RWSFT", args.rwsft_status)
    failures.extend(dpo_child_failures)
    failures.extend(rwsft_child_failures)
    metrics = {
        "pipeline_pass": not failures,
        "claim_allowed": False,
        "min_rows": args.min_rows,
        "min_schema_ok_rate": args.min_schema_ok_rate,
        "models": rows,
        "dpo_rows": next((row.get("rows", 0) for row in rows if row["model"] == "DPO"), 0),
        "rwsft_rows": next((row.get("rows", 0) for row in rows if row["model"] == "RWSFT"), 0),
        "child_statuses": {
            "dpo": {"path": args.dpo_status, "status": dpo_child.get("status") if dpo_child else None, "metrics": dpo_child.get("metrics") if dpo_child else None},
            "rwsft": {"path": args.rwsft_status, "status": rwsft_child.get("status") if rwsft_child else None, "metrics": rwsft_child.get("metrics") if rwsft_child else None},
        },
    }
    write_json(args.metrics, metrics)
    outputs = [args.metrics]
    for path in [args.dpo, args.rwsft]:
        if Path(path).exists():
            outputs.append(path)
    write_manifest(args.manifest, outputs, STEP)
    status = "PASS" if not failures else "FAIL"
    inputs = [args.dpo, args.rwsft, args.contexts]
    if args.dpo_status:
        inputs.append(args.dpo_status)
    if args.rwsft_status:
        inputs.append(args.rwsft_status)
    write_status(args.status, STEP, status, inputs, [*outputs, args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
