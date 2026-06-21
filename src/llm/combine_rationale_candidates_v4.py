from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import write_manifest, write_status

STEP = "13_COMBINE_RATIONALE_CANDIDATES_V4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-candidates-per-sample", type=int, default=1)
    parser.add_argument("--require-train-only", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    frames = []
    for path in args.inputs:
        if not Path(path).exists():
            failures.append(f"missing input: {path}")
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            failures.append(f"empty input: {path}")
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out["candidate_id"] = out["candidate_id"].astype(int)
        out = out.drop_duplicates(["sample_id", "candidate_id"], keep="last")
        out = out.sort_values(["sample_id", "candidate_id"]).reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    candidate_counts = out.groupby("sample_id")["candidate_id"].nunique() if len(out) else pd.Series(dtype=int)
    metrics = {
        "rows": int(len(out)),
        "unique_samples": int(out["sample_id"].nunique()) if len(out) and "sample_id" in out else 0,
        "min_candidates_per_sample_observed": int(candidate_counts.min()) if len(candidate_counts) else 0,
        "mean_candidates_per_sample": float(candidate_counts.mean()) if len(candidate_counts) else 0.0,
        "candidate_distribution": out["candidate_id"].value_counts().sort_index().to_dict() if len(out) else {},
        "split_distribution": out["split"].value_counts(dropna=False).to_dict() if len(out) and "split" in out else {},
    }
    if len(out) < args.min_rows:
        failures.append(f"combined rows {len(out)} < {args.min_rows}")
    if metrics["min_candidates_per_sample_observed"] < args.min_candidates_per_sample:
        failures.append(
            f"min candidates per sample {metrics['min_candidates_per_sample_observed']} < {args.min_candidates_per_sample}"
        )
    if args.require_train_only and len(out) and set(out["split"].dropna()) != {"train"}:
        failures.append("combined rationale candidates contain non-train split")

    write_manifest(args.manifest, [*args.inputs, args.output], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        args.inputs,
        [args.output, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
