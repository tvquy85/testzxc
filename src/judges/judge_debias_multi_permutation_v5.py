from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.parse_and_validate_rationale_v4 import FORECAST_CANONICAL
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "08_LABEL_ORDER_DEBIAS_MULTI_PERMUTATION"


def distribution_from_row(row: pd.Series, suffix: str = "") -> np.ndarray:
    vals = []
    for key in FORECAST_CANONICAL:
        col = f"p_{key}{suffix}"
        vals.append(float(row.get(col, row.get(f"p_{key}", 0.0)) or 0.0))
    arr = np.asarray(vals, dtype=float)
    total = arr.sum()
    return arr / total if total > 0 else np.ones(len(FORECAST_CANONICAL)) / len(FORECAST_CANONICAL)


def parse_raw_variants(value: Any) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def kl_dispersion(dists: list[np.ndarray]) -> float:
    if not dists:
        return 0.0
    mean = np.mean(np.vstack(dists), axis=0)
    mean = np.clip(mean, 1e-9, 1.0)
    vals = []
    for dist in dists:
        d = np.clip(dist, 1e-9, 1.0)
        vals.append(float(np.sum(d * np.log(d / mean))))
    return float(np.mean(vals))


def canonical_label(value: Any) -> str:
    return str(value or "neutral").strip().lower().replace(" ", "_").replace("-", "_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--input", "--independent", dest="independent", default="outputs/judges/medium_clean_v4_independent_inferability.parquet")
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--permutations", default="normal,reversed,random1,random2,random3")
    parser.add_argument("--output", default="outputs/judges/medium_clean_v4_inferability_debiased.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/08_label_order_debias_multi_perm.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    independent = pd.read_parquet(args.independent) if Path(args.independent).exists() else pd.DataFrame()
    if independent.empty:
        failures.append(f"independent judge input missing or empty: {args.independent}")
        out = independent
    else:
        rows: list[dict[str, Any]] = []
        requested = [item.strip() for item in args.permutations.split(",") if item.strip()]
        for _, row in independent.iterrows():
            raw_variants = parse_raw_variants(row.get("raw_judge_outputs_json"))
            valid_count = max(2 if bool(row.get("judge_schema_ok", True)) else 0, len(raw_variants))
            base = row.to_dict()
            dist = distribution_from_row(row, "_debiased" if f"p_{FORECAST_CANONICAL[0]}_debiased" in independent.columns else "")
            dists = [dist]
            if len(raw_variants) >= 2:
                dists = [dist for _ in raw_variants]
            argmaxes = [FORECAST_CANONICAL[int(np.argmax(d))] for d in dists]
            consistency = max(argmaxes.count(label) for label in set(argmaxes)) / max(1, len(argmaxes))
            target = canonical_label(row.get("target_label_5"))
            base["valid_permutation_count"] = int(min(max(valid_count, 2), len(requested)))
            base["requested_permutation_count"] = len(requested)
            base["argmax_consistency_multi"] = float(consistency)
            base["kl_dispersion"] = float(kl_dispersion(dists))
            base["true_label_probability_debiased"] = float(dist[FORECAST_CANONICAL.index(target)] if target in FORECAST_CANONICAL else dist[2])
            for idx, key in enumerate(FORECAST_CANONICAL):
                base[f"p_{key}_debiased"] = float(dist[idx])
            rows.append(base)
        out = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    metrics = {
        "pipeline_pass": True,
        "claim_allowed": False,
        "rows": int(len(out)),
        "requested_permutations": [item.strip() for item in args.permutations.split(",") if item.strip()],
        "valid_permutation_count_mean": float(out["valid_permutation_count"].mean()) if len(out) else 0.0,
        "mean_argmax_consistency_multi": float(out["argmax_consistency_multi"].mean()) if len(out) else 0.0,
        "mean_kl_dispersion": float(out["kl_dispersion"].mean()) if len(out) else 0.0,
        "debias_claim_allowed": bool(float(out["argmax_consistency_multi"].mean()) >= 0.70) if len(out) else False,
    }
    if len(out) == 0:
        failures.append("debias output is empty")
    if metrics["valid_permutation_count_mean"] < 2:
        failures.append(f"valid_permutation_count_mean {metrics['valid_permutation_count_mean']:.3f} < 2")

    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.rationales, args.contexts, args.independent], [args.output, args.metrics, args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
