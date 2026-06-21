from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.parse_and_validate_rationale_v4 import FORECAST_CANONICAL
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "15_FLOW_REWARD_REBUILD_CLEAN_V4_DATASET"
TARGET_NAMES = [
    "true_label_probability_independent",
    "inferability_confidence",
    "news_grounding_score",
    "technical_grounding_score",
    "evidence_quality_weight",
    "counterfactual_proxy_if_available",
    "utility_proxy",
]


def hash_embedding(text: str, dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def entropy_confidence(row: pd.Series) -> float:
    vals = np.asarray([float(row.get(f"p_{key}_debiased", row.get(f"p_{key}", 0.0)) or 0.0) for key in FORECAST_CANONICAL], dtype=np.float32)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return 0.0
    entropy = float(-np.sum(vals * np.log(vals)))
    return max(0.0, min(1.0, 1.0 - entropy / math.log(5)))


def parse_action(value: Any) -> str:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return str(parsed.get("action", "hold")).lower() if isinstance(parsed, dict) else "hold"
    except Exception:
        return "hold"


def action_value(action: str) -> float:
    return {"long": 1.0, "short": -1.0, "hold": 0.0}.get(str(action).lower(), 0.0)


def build_dataset(rationales: pd.DataFrame, inferability: pd.DataFrame, grounding: pd.DataFrame, tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = rationales.copy()
    if df.empty:
        data = {
            "target": np.zeros((0, len(TARGET_NAMES)), dtype=np.float32),
            "mask": np.zeros((0, len(TARGET_NAMES)), dtype=np.float32),
            "cond": np.zeros((0, 64), dtype=np.float32),
            "target_names": TARGET_NAMES,
            "cond_dim": 64,
        }
        return df, data

    inf_cols = [
        col
        for col in [
            "sample_id",
            "candidate_id",
            "true_label_probability_debiased",
            "true_label_probability",
            *[f"p_{key}" for key in FORECAST_CANONICAL],
            *[f"p_{key}_debiased" for key in FORECAST_CANONICAL],
        ]
        if col in inferability.columns
    ]
    if inf_cols:
        df = df.merge(inferability[inf_cols], on=["sample_id", "candidate_id"], how="left")
    df["true_label_probability_independent"] = pd.to_numeric(
        df.get("true_label_probability_debiased", df.get("true_label_probability")), errors="coerce"
    )
    df["inferability_confidence"] = df.apply(entropy_confidence, axis=1)

    g_cols = [col for col in ["sample_id", "candidate_id", "news_grounding_score", "technical_grounding_score", "status"] if col in grounding.columns]
    if g_cols:
        df = df.merge(grounding[g_cols], on=["sample_id", "candidate_id"], how="left")

    t_cols = [
        col
        for col in [
            "sample_id",
            "split",
            "track",
            "target_return",
            "abnormal_return_h1",
            "mean_evidence_quality_score",
            "training_weight",
            "clean_context_text",
        ]
        if col in tracks.columns
    ]
    if t_cols:
        df = df.merge(tracks[t_cols].drop_duplicates("sample_id"), on="sample_id", how="left", suffixes=("", "_track"))
    if "split_track" in df.columns:
        df["split"] = df["split_track"].combine_first(df.get("split"))
    if "track_track" in df.columns:
        df["track"] = df["track_track"].combine_first(df.get("track"))

    df["evidence_quality_weight"] = pd.to_numeric(df.get("mean_evidence_quality_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    returns = pd.to_numeric(df.get("target_return", df.get("abnormal_return_h1", 0.0)), errors="coerce").fillna(0.0)
    df["action"] = df.get("parsed_json", "").apply(parse_action)
    df["action_value"] = df["action"].apply(action_value)
    df["utility_proxy"] = df["action_value"] * returns - df["action_value"].abs() * 0.001
    df["counterfactual_proxy_if_available"] = np.nan

    for name in TARGET_NAMES:
        if name not in df.columns:
            df[name] = np.nan
    target = df[TARGET_NAMES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(target).astype(np.float32)
    target = np.nan_to_num(target, nan=0.0)
    text = (df.get("clean_context_text", "").fillna("") + "\n" + df.get("parsed_json", "").fillna("")).tolist()
    cond = np.vstack([hash_embedding(item) for item in text]).astype(np.float32)
    data = {
        "target": target,
        "mask": mask,
        "cond": cond,
        "target_names": TARGET_NAMES,
        "cond_dim": int(cond.shape[1]),
        "sample_id": df["sample_id"].tolist(),
        "candidate_id": df["candidate_id"].astype(int).tolist() if "candidate_id" in df.columns else [0] * len(df),
        "split": df["split"].tolist() if "split" in df.columns else [],
        "track": df["track"].tolist() if "track" in df.columns else [],
        "metadata": {"semantic_backend": False, "embedding_backend": "hash_blake2b_64"},
    }
    return df, data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--inferability", required=True)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", default="outputs/status/15_FLOW_REWARD_REBUILD_CLEAN_V4_DATASET.status.json")
    parser.add_argument("--manifest", default="outputs/manifests/15_FLOW_REWARD_REBUILD_CLEAN_V4_DATASET.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if os.path.exists(args.rationales) else pd.DataFrame()
    inferability = pd.read_parquet(args.inferability) if os.path.exists(args.inferability) else pd.DataFrame()
    grounding = pd.read_parquet(args.grounding) if os.path.exists(args.grounding) else pd.DataFrame()
    tracks = pd.read_parquet(args.tracks) if os.path.exists(args.tracks) else pd.DataFrame()
    for label, df, path in [
        ("rationales", rationales, args.rationales),
        ("inferability", inferability, args.inferability),
        ("grounding", grounding, args.grounding),
        ("tracks", tracks, args.tracks),
    ]:
        if df.empty:
            failures.append(f"{label} missing or empty: {path}")

    df, data = build_dataset(rationales, inferability, grounding, tracks)
    if len(df) == 0:
        failures.append("flow v4 dataset has zero rows")
    mask_coverage = float(data["mask"].mean()) if data["mask"].size else 0.0
    if mask_coverage < 0.60:
        failures.append(f"target mask coverage {mask_coverage:.4f} < 0.6000")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    metrics = {
        "rows": int(len(df)),
        "target_dim": len(TARGET_NAMES),
        "cond_dim": int(data["cond_dim"]),
        "mask_coverage": mask_coverage,
        "target_names": TARGET_NAMES,
        "semantic_backend": False,
        "embedding_backend": "hash_blake2b_64",
        "split_distribution": df["split"].value_counts(dropna=False).to_dict() if len(df) and "split" in df else {},
        "track_distribution": df["track"].value_counts(dropna=False).to_dict() if len(df) and "track" in df else {},
    }
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.rationales, args.inferability, args.grounding, args.tracks],
        [args.output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
