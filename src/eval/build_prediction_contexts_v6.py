from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.audit_hard_event_v6 import classify_context_hardness
from src.data.filter_hard_event_v6 import assign_v6_track, training_weight
from src.data.select_medium_clean_v4_samples import ensure_v4_columns, stratified_sample
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "14_BUILD_V6_PREDICTION_CONTEXTS"


def split_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "split" not in df.columns:
        return {}
    out: dict[str, Any] = {"counts": df["split"].value_counts(dropna=False).to_dict()}
    if "event_date" in df.columns:
        out["days"] = {
            str(split): int(pd.to_datetime(group["event_date"], errors="coerce").dt.date.nunique())
            for split, group in df.groupby("split")
        }
    return out


def split_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if df.empty or "split" not in df.columns:
        return df.iloc[0:0].copy()
    out = df[df["split"].astype(str).eq(split)].copy()
    if "event_date" in out.columns:
        out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    return out


def add_v6_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "evidence_pack_json" in out.columns:
        res = out["evidence_pack_json"].apply(classify_context_hardness)
        res_df = pd.DataFrame(res.tolist(), index=out.index)
        for col in res_df.columns:
            out[col] = res_df[col]
    else:
        out["num_hard_event_evidence"] = 0
    out["v6_track"] = out.apply(assign_v6_track, axis=1)
    out["v6_training_weight"] = out["v6_track"].apply(training_weight)
    return out


def source_is_sufficient(df: pd.DataFrame, split: str, min_rows: int, min_trading_days: int) -> bool:
    part = split_rows(df, split)
    if len(part) < min_rows:
        return False
    if min_trading_days > 0:
        days = int(part["event_date"].dt.date.nunique()) if "event_date" in part.columns and len(part) else 0
        if days < min_trading_days:
            return False
    return True


def select_source_contexts(args: argparse.Namespace) -> tuple[pd.DataFrame, str, bool, list[str], dict[str, Any]]:
    warnings: list[str] = []
    source = pd.read_parquet(args.source) if Path(args.source).exists() else pd.DataFrame()
    source_summary = split_summary(source)
    if source_is_sufficient(source, args.split, args.min_rows, args.min_trading_days):
        return source.copy(), args.source, False, warnings, {"source": source_summary}
    if not Path(args.fallback_contexts).exists():
        return source.copy(), args.source, False, warnings, {"source": source_summary, "fallback": {}}
    fallback = pd.read_parquet(args.fallback_contexts)
    fallback_summary = split_summary(fallback)
    warnings.append(
        f"source contexts insufficient for split={args.split} min_rows={args.min_rows} "
        f"min_trading_days={args.min_trading_days}; used fallback current-data contexts"
    )
    return fallback.copy(), args.fallback_contexts, True, warnings, {"source": source_summary, "fallback": fallback_summary}


def build_prediction_contexts(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    failures: list[str] = []
    base, source_used, used_fallback, warnings, input_summaries = select_source_contexts(args)
    if base.empty:
        failures.append("no source contexts available")
        selected = pd.DataFrame()
    else:
        contexts = ensure_v4_columns(base).drop_duplicates("sample_id").copy()
        selected = stratified_sample(contexts, args.split, args.min_rows, args.seed).drop_duplicates("sample_id").copy()
        selected = selected[selected["split"].astype(str).eq(args.split)].copy()
        selected = add_v6_columns(selected)
        selected = selected.sort_values(["event_date", "ticker", "sample_id"]).copy()
    days = int(pd.to_datetime(selected["event_date"], errors="coerce").dt.date.nunique()) if len(selected) and "event_date" in selected.columns else 0
    if len(selected) < args.min_rows:
        failures.append(f"selected rows {len(selected)} < {args.min_rows}")
    if days < args.min_trading_days:
        failures.append(f"selected trading days {days} < {args.min_trading_days}")
    if len(selected) and set(selected["split"].dropna()) != {args.split}:
        failures.append("prediction contexts contain non-target split rows")
    if len(selected) and selected["sample_id"].duplicated().any():
        failures.append("duplicate sample_id in prediction contexts")
    metrics = {
        "pipeline_pass": not failures,
        "claim_allowed": False,
        "source_used": source_used,
        "used_fallback_contexts": used_fallback,
        "warnings": warnings,
        "input_summaries": input_summaries,
        "split": args.split,
        "rows": int(len(selected)),
        "unique_sample_ids": int(selected["sample_id"].nunique()) if len(selected) else 0,
        "min_rows": args.min_rows,
        "selected_trading_days": days,
        "min_trading_days": args.min_trading_days,
        "selected_start_date": str(pd.to_datetime(selected["event_date"]).min().date()) if len(selected) and "event_date" in selected.columns else None,
        "selected_end_date": str(pd.to_datetime(selected["event_date"]).max().date()) if len(selected) and "event_date" in selected.columns else None,
        "label_distribution": selected["target_label_5"].value_counts(dropna=False).to_dict() if len(selected) and "target_label_5" in selected.columns else {},
        "v6_track_distribution": selected["v6_track"].value_counts(dropna=False).to_dict() if len(selected) and "v6_track" in selected.columns else {},
        "hard_event_news": int((selected.get("v6_track") == "hard_event_news").sum()) if len(selected) and "v6_track" in selected.columns else 0,
    }
    return selected, metrics, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet")
    parser.add_argument("--fallback-contexts", default="data/processed/ticker_date_contexts_h1_v2_targets.parquet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--min-trading-days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output", default="data/processed/current_v6_prediction_contexts.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/14_v6_prediction_contexts.json")
    parser.add_argument("--samples", default="review_samples/currentdata_v6/14_prediction_context_samples.jsonl")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    selected, metrics, failures = build_prediction_contexts(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    Path(args.samples).parent.mkdir(parents=True, exist_ok=True)
    selected.head(50).to_json(args.samples, orient="records", lines=True)
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics, args.samples], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.source, args.fallback_contexts],
        [args.output, args.metrics, args.samples, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
