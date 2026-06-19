from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.generate_rationales import parsed_records_from_raw, read_jsonl
from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "09_RATIONALE_GENERATION_SCALEUP_TRAIN_ONLY"
LEAK_RE = re.compile(r"\b(realized|ground[-\s]?truth|true)\s+(label|return|price)\b", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", action="append", required=True)
    parser.add_argument("--parsed-output", default="data/rationales/parsed/train_candidates_stage1_strict.parquet")
    parser.add_argument("--summary", default="outputs/metrics/stage1_rationale_generation_summary.json")
    parser.add_argument("--stage", default="stage_1_small_scale")
    parser.add_argument("--parse-ok-min", type=float, default=0.95)
    parser.add_argument("--schema-ok-min", type=float, default=0.85)
    parser.add_argument("--avg-output-tokens-max", type=float, default=280.0)
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    raw_rows = []
    for raw_path in args.raw_input:
        for record in read_jsonl(raw_path):
            record = dict(record)
            record["source_raw_path"] = raw_path
            raw_rows.append(record)
    parsed_records = parsed_records_from_raw(raw_rows)
    df = pd.DataFrame(parsed_records)
    failures: list[str] = []
    if df.empty:
        failures.append("combined stage1 rationale dataset is empty")
    key_cols = ["sample_id", "candidate_id"]
    duplicate_count = int(df.duplicated(key_cols).sum()) if set(key_cols).issubset(df.columns) else len(df)
    if duplicate_count:
        failures.append(f"duplicate sample_id/candidate_id rows: {duplicate_count}")
    non_train = sorted(set(df["split"].dropna()) - {"train"}) if "split" in df else ["missing_split_column"]
    if non_train:
        failures.append(f"combined rationale output contains non-train split: {non_train}")
    parse_ok_rate = float(df["parse_ok"].mean()) if len(df) else 0.0
    schema_ok_rate = float(df["schema_ok"].mean()) if len(df) else 0.0
    avg_output_tokens = sum(int(r.get("output_tokens_est", 0) or 0) for r in raw_rows) / max(1, len(raw_rows))
    leak_count = sum(1 for r in raw_rows if LEAK_RE.search(str(r.get("raw_output", r.get("raw_text", "")))))
    if parse_ok_rate < args.parse_ok_min:
        failures.append(f"parse_ok_rate {parse_ok_rate:.6f} < {args.parse_ok_min:.6f}")
    if schema_ok_rate < args.schema_ok_min:
        failures.append(f"schema_ok_rate {schema_ok_rate:.6f} < {args.schema_ok_min:.6f}")
    if avg_output_tokens > args.avg_output_tokens_max:
        failures.append(f"avg_output_tokens_est {avg_output_tokens:.3f} > {args.avg_output_tokens_max:.3f}")
    if leak_count:
        failures.append(f"explicit label/return leak patterns found: {leak_count}")
    Path(args.parsed_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.parsed_output, index=False)
    metrics = {
        "row_count": int(len(df)),
        "unique_sample_count": int(df["sample_id"].nunique()) if "sample_id" in df else 0,
        "raw_input_count": len(args.raw_input),
        "parse_ok_rate": parse_ok_rate,
        "schema_ok_rate": schema_ok_rate,
        "avg_output_tokens_est": avg_output_tokens,
        "duplicate_key_count": duplicate_count,
        "explicit_leak_pattern_count": leak_count,
        "stage": args.stage,
    }
    write_json(args.summary, metrics)
    write_manifest(args.manifest, [*args.raw_input, args.parsed_output, args.summary], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=args.raw_input,
        outputs_created=[args.parsed_output, args.summary, args.manifest, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
