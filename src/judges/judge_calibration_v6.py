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

from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "09_JUDGE_CALIBRATION_AND_DEBIAS_TESTS"
FORECAST_KEYS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
PROB_COLS = [f"p_{key}" for key in FORECAST_KEYS]
MIN_TRUE_LABEL_PROBABILITY = 0.20
MIN_FULLY_PERMUTED_ROW_RATE = 0.95
MIN_ARGMAX_CONSISTENCY = 0.75
SUM_TOL = 1e-6


def canonical_label(value: Any) -> str:
    text = str(value or "neutral").strip().lower().replace(" ", "_").replace("-", "_")
    return text if text in FORECAST_KEYS else "neutral"


def expected_calibration_error(confidences: list[float] | np.ndarray, outcomes: list[int] | np.ndarray, n_bins: int = 10) -> float:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if conf.shape[0] != y.shape[0]:
        raise ValueError("confidences and outcomes must have the same length")
    if conf.size == 0:
        return float("nan")

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lower = boundaries[idx]
        upper = boundaries[idx + 1]
        if idx == n_bins - 1:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf >= lower) & (conf < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_confidence = float(conf[mask].mean())
        accuracy = float(y[mask].mean())
        ece += abs(accuracy - avg_confidence) * count / conf.size
    return float(ece)


def reliability_bins(confidences: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    total = max(1, len(confidences))
    for idx in range(n_bins):
        lower = float(boundaries[idx])
        upper = float(boundaries[idx + 1])
        if idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        avg_confidence = float(confidences[mask].mean()) if count else float("nan")
        accuracy = float(outcomes[mask].mean()) if count else float("nan")
        gap = abs(accuracy - avg_confidence) if count else float("nan")
        rows.append(
            {
                "bin": idx,
                "lower": lower,
                "upper": upper,
                "count": count,
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
                "abs_gap": gap,
                "weighted_abs_gap": float(gap * count / total) if count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def brier_score_multiclass(probs: np.ndarray, y_true: np.ndarray) -> float:
    one_hot = np.zeros_like(probs, dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    true_probs = probs[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(true_probs, 1e-12, 1.0))))


def macro_ovr_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    vals = []
    for class_idx in range(probs.shape[1]):
        outcomes = (y_true == class_idx).astype(int)
        vals.append(expected_calibration_error(probs[:, class_idx], outcomes, n_bins=n_bins))
    return float(np.mean(vals))


def validate_probabilities(df: pd.DataFrame) -> dict[str, Any]:
    probs = df[PROB_COLS].to_numpy(dtype=float)
    sums = probs.sum(axis=1)
    finite_mask = np.isfinite(probs).all(axis=1)
    sum_mask = np.abs(sums - 1.0) <= SUM_TOL
    nonnegative_mask = (probs >= -SUM_TOL).all(axis=1)
    upper_mask = (probs <= 1.0 + SUM_TOL).all(axis=1)
    valid_mask = finite_mask & sum_mask & nonnegative_mask & upper_mask
    return {
        "probs": probs,
        "sums": sums,
        "valid_mask": valid_mask,
        "all_probabilities_finite": bool(finite_mask.all()),
        "all_probability_sums_ok": bool(sum_mask.all()),
        "all_probabilities_nonnegative": bool(nonnegative_mask.all()),
        "all_probabilities_le_one": bool(upper_mask.all()),
        "valid_probability_row_rate": float(valid_mask.mean()) if len(valid_mask) else 0.0,
        "probability_sum_min": float(np.min(sums)) if len(sums) else float("nan"),
        "probability_sum_max": float(np.max(sums)) if len(sums) else float("nan"),
    }


def compute_calibration(df: pd.DataFrame, n_bins: int = 10) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    failures: list[str] = []
    required = set(PROB_COLS + ["target_label_5", "argmax_consistency_ensemble", "label_order_kl_mean", "judge_schema_ok"])
    missing = sorted(required - set(df.columns))
    if missing:
        return {}, pd.DataFrame(), [f"missing required columns: {missing}"]
    if df.empty:
        return {}, pd.DataFrame(), ["input judge ensemble is empty"]

    validation = validate_probabilities(df)
    probs = validation["probs"]
    targets = np.asarray([FORECAST_KEYS.index(canonical_label(v)) for v in df["target_label_5"].tolist()], dtype=int)
    target_labels = [FORECAST_KEYS[idx] for idx in targets]
    known_target_mask = np.asarray([canonical_label(v) in FORECAST_KEYS for v in df["target_label_5"].tolist()], dtype=bool)
    valid_mask = validation["valid_mask"] & known_target_mask

    if not validation["all_probabilities_finite"]:
        failures.append("not all aggregate probabilities are finite")
    if not validation["all_probability_sums_ok"]:
        failures.append("not all aggregate probabilities sum to one")
    if not validation["all_probabilities_nonnegative"] or not validation["all_probabilities_le_one"]:
        failures.append("aggregate probabilities outside [0, 1]")
    if int(valid_mask.sum()) == 0:
        failures.append("no valid probability rows available for calibration")
        return {}, pd.DataFrame(), failures

    cal_probs = probs[valid_mask]
    cal_targets = targets[valid_mask]
    cal_df = df.loc[valid_mask].copy()
    top_idx = np.argmax(cal_probs, axis=1)
    top_conf = cal_probs[np.arange(len(cal_probs)), top_idx]
    correct = (top_idx == cal_targets).astype(int)
    true_probs = cal_probs[np.arange(len(cal_probs)), cal_targets]
    bins = reliability_bins(top_conf, correct, n_bins=n_bins)

    by_target: dict[str, dict[str, Any]] = {}
    for label in FORECAST_KEYS:
        mask = np.asarray(target_labels)[valid_mask] == label
        if not mask.any():
            continue
        sub_probs = cal_probs[mask]
        sub_targets = cal_targets[mask]
        sub_top_idx = np.argmax(sub_probs, axis=1)
        sub_correct = (sub_top_idx == sub_targets).astype(int)
        sub_df = cal_df.loc[mask]
        by_target[label] = {
            "rows": int(mask.sum()),
            "true_label_probability_mean": float(sub_probs[np.arange(mask.sum()), sub_targets].mean()),
            "top_label_accuracy": float(sub_correct.mean()),
            "argmax_consistency_mean": float(sub_df["argmax_consistency_ensemble"].mean()),
            "label_order_kl_mean": float(sub_df["label_order_kl_mean"].mean()),
        }

    fully_permuted = df["judge_schema_ok"].astype(bool).to_numpy()
    fully_permuted_row_rate = float(fully_permuted.mean())
    mean_true_prob = float(true_probs.mean())
    mean_argmax_consistency = float(cal_df["argmax_consistency_ensemble"].mean())

    metrics: dict[str, Any] = {
        "pipeline_pass": True,
        "claim_allowed": False,
        "rows": int(len(df)),
        "calibration_rows": int(valid_mask.sum()),
        "fully_permuted_rows": int(fully_permuted.sum()),
        "fully_permuted_row_rate": fully_permuted_row_rate,
        **{k: v for k, v in validation.items() if k not in {"probs", "sums", "valid_mask"}},
        "mean_true_label_probability": mean_true_prob,
        "top_label_accuracy": float(correct.mean()),
        "brier_score_multiclass": brier_score_multiclass(cal_probs, cal_targets),
        "nll": negative_log_likelihood(cal_probs, cal_targets),
        "ece_top_label": expected_calibration_error(top_conf, correct, n_bins=n_bins),
        "ece_macro_ovr": macro_ovr_ece(cal_probs, cal_targets, n_bins=n_bins),
        "mean_argmax_consistency_ensemble": mean_argmax_consistency,
        "mean_label_order_kl": float(cal_df["label_order_kl_mean"].mean()),
        "mean_judge_disagreement_entropy": float(cal_df["judge_disagreement_entropy"].mean())
        if "judge_disagreement_entropy" in cal_df.columns
        else float("nan"),
        "true_label_probability_by_target_label": by_target,
        "reliability_bin_count": int(n_bins),
    }

    if mean_true_prob <= MIN_TRUE_LABEL_PROBABILITY:
        failures.append(f"mean_true_label_probability {mean_true_prob:.3f} <= {MIN_TRUE_LABEL_PROBABILITY:.2f}")
    if fully_permuted_row_rate < MIN_FULLY_PERMUTED_ROW_RATE:
        failures.append(f"fully_permuted_row_rate {fully_permuted_row_rate:.3f} < {MIN_FULLY_PERMUTED_ROW_RATE:.2f}")
    if mean_argmax_consistency < MIN_ARGMAX_CONSISTENCY:
        failures.append(f"argmax_consistency {mean_argmax_consistency:.3f} < {MIN_ARGMAX_CONSISTENCY:.2f}")
    if not math.isfinite(metrics["brier_score_multiclass"]) or not math.isfinite(metrics["nll"]):
        failures.append("Brier score or NLL is not finite")

    metrics["pipeline_pass"] = not failures
    return metrics, bins, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--bins", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input)
    failures: list[str] = []
    if not input_path.exists():
        failures.append(f"input missing: {args.input}")
        df = pd.DataFrame()
    else:
        df = pd.read_parquet(input_path)

    metrics, bins, compute_failures = compute_calibration(df, n_bins=args.n_bins) if not failures else ({}, pd.DataFrame(), [])
    failures.extend(compute_failures)

    Path(args.bins).parent.mkdir(parents=True, exist_ok=True)
    bins.to_csv(args.bins, index=False)
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.metrics, args.bins], STEP)
    status = "PASS" if not failures else "FAIL"
    outputs = [args.metrics, args.bins, args.manifest, args.status]
    write_status(args.status, STEP, status, [args.input], outputs, metrics, failures, status == "PASS")
    print(json.dumps({"status": status, "failures": failures, "metrics": metrics}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
