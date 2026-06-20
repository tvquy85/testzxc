from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "10_FLOW_DATASET_V3_TARGETS_AND_EMBEDDINGS"
TARGET_NAMES = [
    "independent_true_label_prob",
    "inferability_certainty",
    "debias_stability_score",
    "news_grounding_score",
    "technical_grounding_score",
    "utility_proxy",
    "calibration_proxy",
]
FORECAST_KEYS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]


def hash_embedding(text: str, dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_entropy(probs: list[float]) -> float:
    p = np.asarray(probs, dtype=np.float32)
    p = p[np.isfinite(p) & (p > 0)]
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def parse_action(parsed_json: Any) -> str:
    try:
        data = json.loads(parsed_json) if isinstance(parsed_json, str) else parsed_json
        return str(data.get("action", "hold")).lower() if isinstance(data, dict) else "hold"
    except Exception:
        return "hold"


def action_value(action: str) -> float:
    return {"long": 1.0, "short": -1.0, "hold": 0.0}.get(str(action).lower(), 0.0)


def target_direction_value(label: Any) -> float:
    text = str(label).lower()
    if "up" in text:
        return 1.0
    if "down" in text:
        return -1.0
    return 0.0


def select_inferability_columns(inferability: pd.DataFrame, min_debias_consistency: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    if inferability.empty:
        return pd.DataFrame(), {"debias_used": False, "debias_reason": "inferability_empty"}
    keys = [f"p_{key}" for key in FORECAST_KEYS]
    debiased_keys = [f"p_{key}_debiased" for key in FORECAST_KEYS]
    consistency = None
    if "argmax_consistency" in inferability.columns:
        consistency = float(pd.to_numeric(inferability["argmax_consistency"], errors="coerce").mean())
    can_use_debiased = (
        "true_label_probability_debiased" in inferability.columns
        and all(col in inferability.columns for col in debiased_keys)
        and consistency is not None
        and consistency >= min_debias_consistency
    )
    if can_use_debiased:
        cols = ["sample_id", "candidate_id", "true_label_probability_debiased", *debiased_keys, "argmax_consistency", "l1_probability_delta"]
        out = inferability[cols].copy()
        out.rename(
            columns={
                "true_label_probability_debiased": "independent_true_label_prob",
                **{f"p_{key}_debiased": f"judge_{key}" for key in FORECAST_KEYS},
            },
            inplace=True,
        )
        return out, {"debias_used": True, "argmax_consistency_mean": consistency}
    if "true_label_probability" not in inferability.columns or not all(col in inferability.columns for col in keys):
        return pd.DataFrame(), {"debias_used": False, "debias_reason": "missing_base_columns", "argmax_consistency_mean": consistency}
    out = inferability[["sample_id", "candidate_id", "true_label_probability", *keys]].copy()
    out.rename(
        columns={
            "true_label_probability": "independent_true_label_prob",
            **{f"p_{key}": f"judge_{key}" for key in FORECAST_KEYS},
        },
        inplace=True,
    )
    out["argmax_consistency"] = np.nan
    out["l1_probability_delta"] = np.nan
    reason = "debias_consistency_below_gate" if consistency is not None else "no_debias_metrics"
    return out, {"debias_used": False, "debias_reason": reason, "argmax_consistency_mean": consistency}


def build_dataset(
    rationales: pd.DataFrame,
    inferability: pd.DataFrame,
    grounding: pd.DataFrame,
    contexts: pd.DataFrame,
    min_debias_consistency: float = 0.55,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    df = rationales.copy()
    if len(df) == 0:
        return df, {"target": np.zeros((0, 7), dtype=np.float32), "cond": np.zeros((0, 64), dtype=np.float32), "mask": np.zeros((0, 7), dtype=np.float32), "target_names": TARGET_NAMES, "cond_dim": 64}, {"debias_used": False}

    inf, debias_info = select_inferability_columns(inferability, min_debias_consistency)
    if not inf.empty:
        def certainty(row: pd.Series) -> float:
            probs = [float(row.get(f"judge_{key}", 0.0) or 0.0) for key in FORECAST_KEYS]
            return max(0.0, 1.0 - compute_entropy(probs) / np.log(5))

        inf["inferability_certainty"] = inf.apply(certainty, axis=1)
        inf["debias_stability_score"] = 1.0 - pd.to_numeric(inf.get("l1_probability_delta", np.nan), errors="coerce").fillna(0.0).clip(0.0, 2.0) / 2.0
        df = df.merge(
            inf[["sample_id", "candidate_id", "independent_true_label_prob", "inferability_certainty", "debias_stability_score"]],
            on=["sample_id", "candidate_id"],
            how="left",
        )

    if not grounding.empty:
        g = grounding.copy()
        if "supported_claim_rate" not in g.columns:
            total = pd.to_numeric(g.get("total_claims", 0), errors="coerce").fillna(0.0)
            supported = pd.to_numeric(g.get("supported_claims", 0), errors="coerce").fillna(0.0)
            g["supported_claim_rate"] = np.where(total > 0, supported / total, np.nan)
        for col in ["news_grounding_score", "technical_grounding_score"]:
            if col not in g.columns:
                g[col] = g["supported_claim_rate"]
        df = df.merge(
            g[["sample_id", "candidate_id", "news_grounding_score", "technical_grounding_score"]],
            on=["sample_id", "candidate_id"],
            how="left",
        )

    if not contexts.empty:
        context_cols = [col for col in ["sample_id", "target_return", "target_label_5", "target_direction", "split"] if col in contexts.columns]
        df = df.merge(contexts[context_cols], on="sample_id", how="left", suffixes=("", "_context"))

    df["action"] = df.get("parsed_json", "").apply(parse_action)
    df["action_value"] = df["action"].apply(action_value)
    df["target_return"] = pd.to_numeric(df.get("target_return", 0.0), errors="coerce").fillna(0.0)
    df["utility_proxy"] = df["action_value"] * df["target_return"] - df["action_value"].abs() * 0.001
    target_direction = df.get("target_label_5", df.get("target_direction", "neutral")).apply(target_direction_value)
    df["calibration_proxy"] = np.where(
        np.sign(df["action_value"]) == np.sign(target_direction),
        1.0,
        np.where((df["action_value"] == 0) & (target_direction == 0), 1.0, -1.0),
    )

    for name in TARGET_NAMES:
        if name not in df.columns:
            df[name] = np.nan

    target = df[TARGET_NAMES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(target).astype(np.float32)
    target = np.nan_to_num(target, nan=0.0)
    text = (df.get("parsed_json", "").fillna("") + "\n" + df.get("prompt", "").fillna("")).tolist()
    cond = np.vstack([hash_embedding(item) for item in text]).astype(np.float32) if len(df) else np.zeros((0, 64), dtype=np.float32)
    return df, {"target": target, "mask": mask, "cond": cond, "target_names": TARGET_NAMES, "cond_dim": int(cond.shape[1])}, debias_info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--inferability", required=True)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-debias-consistency", type=float, default=0.55)
    args = parser.parse_args()

    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if os.path.exists(args.rationales) else pd.DataFrame()
    inferability = pd.read_parquet(args.inferability) if os.path.exists(args.inferability) else pd.DataFrame()
    grounding = pd.read_parquet(args.grounding) if os.path.exists(args.grounding) else pd.DataFrame()
    contexts = pd.read_parquet(args.contexts) if os.path.exists(args.contexts) else pd.DataFrame()
    if args.limit and args.limit > 0:
        rationales = rationales.head(args.limit).copy()

    df, data, debias_info = build_dataset(rationales, inferability, grounding, contexts, args.min_debias_consistency)
    if len(df) == 0:
        failures.append("Dataset has zero rows")
    if data["target"].shape[1] != len(TARGET_NAMES):
        failures.append("Target dimension mismatch")
    if float(data["mask"].mean()) < 0.50:
        failures.append(f"target mask coverage {float(data['mask'].mean()):.4f} < 0.5000")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **data,
            "sample_id": df["sample_id"].tolist() if len(df) else [],
            "candidate_id": df["candidate_id"].tolist() if len(df) and "candidate_id" in df else [],
            "split": df["split"].tolist() if len(df) and "split" in df else [],
            "metadata": {"debias": debias_info, "embedding_backend": "hash_blake2b_64"},
        },
        args.output,
    )

    metrics = {
        "rows": int(len(df)),
        "target_dim": len(TARGET_NAMES),
        "cond_dim": int(data["cond_dim"]),
        "mask_coverage": float(data["mask"].mean()) if data["mask"].size else 0.0,
        "target_names": TARGET_NAMES,
        "debias": debias_info,
        "embedding_backend": "hash_blake2b_64",
        "split_distribution": df["split"].value_counts(dropna=False).to_dict() if len(df) and "split" in df else {},
    }
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.rationales, args.inferability, args.grounding, args.contexts],
        outputs_created=[args.output, args.metrics, args.manifest, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
