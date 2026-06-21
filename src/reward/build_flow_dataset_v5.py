from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.parse_and_validate_rationale_v4 import FORECAST_CANONICAL
from src.reward.build_flow_dataset_v4 import action_value, entropy_confidence, parse_action
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "11_FLOW_TARGET_NORMALIZATION_AND_SPLIT"
TARGET_NAMES = [
    "true_label_probability_independent",
    "inferability_confidence",
    "news_grounding_score",
    "technical_grounding_score",
    "evidence_quality_weight",
    "debias_stability",
    "utility_proxy",
]


def stable_val_mask(sample_ids: list[Any], frac: float = 0.2) -> np.ndarray:
    import hashlib

    mask = []
    for value in sample_ids:
        digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
        mask.append(int.from_bytes(digest[:4], "little") % 10_000 < int(frac * 10_000))
    out = np.asarray(mask, dtype=bool)
    if out.sum() < max(20, len(out) * 0.1):
        out[:] = False
        out[-max(20, int(len(out) * frac)) :] = True
    return out


def finite_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def normalize_train(values: pd.Series, train_mask: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    train_vals = vals[train_mask & vals.notna()]
    if train_vals.empty or train_vals.nunique() <= 1:
        return pd.Series(0.5, index=vals.index, dtype=float)
    lo = float(train_vals.quantile(0.05))
    hi = float(train_vals.quantile(0.95))
    if abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=vals.index, dtype=float)
    return ((vals.clip(lo, hi) - lo) / (hi - lo)).clip(0.0, 1.0)


def proxy_average(target: np.ndarray, mask: np.ndarray, names: list[str]) -> np.ndarray:
    indices = [idx for idx, name in enumerate(names) if name != "utility_proxy"]
    vals = np.where(mask[:, indices] > 0, target[:, indices], np.nan)
    return np.nanmean(vals, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--inferability", required=True)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--embeddings-npy", required=True)
    parser.add_argument("--embeddings-index", required=True)
    parser.add_argument("--output", default="data/reward/medium_clean_v4_flow_dataset_v5.pt")
    parser.add_argument("--metrics", default="outputs/metrics/11_flow_dataset_v5.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if Path(args.rationales).exists() else pd.DataFrame()
    inferability = pd.read_parquet(args.inferability) if Path(args.inferability).exists() else pd.DataFrame()
    grounding = pd.read_parquet(args.grounding) if Path(args.grounding).exists() else pd.DataFrame()
    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    X = np.load(args.embeddings_npy) if Path(args.embeddings_npy).exists() else np.zeros((0, 0), dtype=np.float32)
    emb_index = pd.read_parquet(args.embeddings_index) if Path(args.embeddings_index).exists() else pd.DataFrame()
    for label, df, path in [
        ("rationales", rationales, args.rationales),
        ("inferability", inferability, args.inferability),
        ("grounding", grounding, args.grounding),
        ("contexts", contexts, args.contexts),
        ("embedding_index", emb_index, args.embeddings_index),
    ]:
        if df.empty:
            failures.append(f"{label} missing or empty: {path}")
    if X.size == 0:
        failures.append(f"embeddings missing or empty: {args.embeddings_npy}")

    df = rationales.copy()
    if not failures:
        keys = ["sample_id", "candidate_id"]
        df["candidate_id"] = df["candidate_id"].astype(int)
        inferability["candidate_id"] = inferability["candidate_id"].astype(int)
        grounding["candidate_id"] = grounding["candidate_id"].astype(int)
        df = df.merge(inferability[[c for c in ["sample_id", "candidate_id", "true_label_probability_debiased", "true_label_probability", "argmax_consistency_multi", "argmax_consistency", *[f"p_{k}_debiased" for k in FORECAST_CANONICAL]] if c in inferability.columns]], on=keys, how="left")
        df = df.merge(grounding[[c for c in ["sample_id", "candidate_id", "news_grounding_score", "technical_grounding_score", "status"] if c in grounding.columns]], on=keys, how="left")
        ctx_cols = [c for c in ["sample_id", "target_return", "abnormal_return_h1", "mean_evidence_quality_score", "news_reasoning_track", "track"] if c in contexts.columns]
        df = df.merge(contexts[ctx_cols].drop_duplicates("sample_id"), on="sample_id", how="left")
        emb_index = emb_index.copy()
        emb_index["candidate_id"] = emb_index["candidate_id"].astype(int)
        df = df.merge(emb_index[["sample_id", "candidate_id", "row_idx"]], on=keys, how="left")
        if df["row_idx"].isna().any():
            failures.append("some candidates missing embedding row_idx")

    if failures:
        target = np.zeros((0, len(TARGET_NAMES)), dtype=np.float32)
        mask = np.zeros_like(target)
        cond = np.zeros((0, max(128, X.shape[1] if X.ndim == 2 else 128)), dtype=np.float32)
        split = []
    else:
        row_idx = df["row_idx"].astype(int).to_numpy()
        cond = X[row_idx].astype(np.float32)
        sample_val = stable_val_mask(df["sample_id"].tolist(), frac=0.2)
        flow_split = np.where(sample_val, "val", "train")
        train_mask = pd.Series(flow_split == "train", index=df.index)
        returns = finite_series(df, "target_return").fillna(finite_series(df, "abnormal_return_h1", 0.0))
        actions = df.get("parsed_json", "").apply(parse_action)
        utility_raw = actions.apply(action_value) * returns - actions.apply(action_value).abs() * 0.001
        df["utility_proxy_raw"] = utility_raw
        df["utility_proxy"] = normalize_train(utility_raw, train_mask)
        df["true_label_probability_independent"] = finite_series(df, "true_label_probability_debiased").fillna(finite_series(df, "true_label_probability", 0.0)).clip(0.0, 1.0)
        df["inferability_confidence"] = df.apply(entropy_confidence, axis=1)
        df["news_grounding_score"] = finite_series(df, "news_grounding_score").clip(0.0, 1.0)
        df["technical_grounding_score"] = finite_series(df, "technical_grounding_score").clip(0.0, 1.0)
        df["evidence_quality_weight"] = finite_series(df, "mean_evidence_quality_score", 0.5).fillna(0.5).clip(0.0, 1.0)
        stability = finite_series(df, "argmax_consistency_multi").fillna(finite_series(df, "argmax_consistency", 0.5)).clip(0.0, 1.0)
        df["debias_stability"] = stability
        target_df = df[TARGET_NAMES].apply(pd.to_numeric, errors="coerce")
        mask = target_df.notna().to_numpy(dtype=np.float32)
        target = target_df.fillna(0.0).to_numpy(dtype=np.float32)
        split = flow_split.tolist()
        df["flow_split"] = split
        df["proxy_reward"] = proxy_average(target, mask, TARGET_NAMES)

    data = {
        "target": target,
        "mask": mask,
        "cond": cond,
        "target_names": TARGET_NAMES,
        "cond_dim": int(cond.shape[1]) if cond.ndim == 2 else 0,
        "sample_id": df["sample_id"].tolist() if len(df) else [],
        "candidate_id": df["candidate_id"].astype(int).tolist() if len(df) else [],
        "split": split,
        "source_split": df.get("split", pd.Series(["train"] * len(df))).tolist() if len(df) else [],
        "track": df.get("track", pd.Series([""] * len(df))).tolist() if len(df) else [],
        "proxy_reward": df.get("proxy_reward", pd.Series([], dtype=float)).tolist() if len(df) else [],
        "utility_proxy_raw": df.get("utility_proxy_raw", pd.Series([], dtype=float)).tolist() if len(df) else [],
        "metadata": {"embedding_backend": "semantic_v5", "normalization": "train_winsor_minmax_for_utility"},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    mask_coverage = float(mask.mean()) if mask.size else 0.0
    metrics = {
        "pipeline_pass": True,
        "claim_allowed": False,
        "rows": int(target.shape[0]),
        "target_dim": len(TARGET_NAMES),
        "cond_dim": int(data["cond_dim"]),
        "target_mask_coverage": mask_coverage,
        "split_distribution": pd.Series(split).value_counts().to_dict() if split else {},
        "target_names": TARGET_NAMES,
        "utility_normalized_train_only": True,
    }
    if target.shape[0] < 1000:
        failures.append(f"flow dataset rows {target.shape[0]} < 1000")
    if mask_coverage < 0.70:
        failures.append(f"target mask coverage {mask_coverage:.4f} < 0.70")
    if not {"train", "val"}.issubset(set(split)):
        failures.append("flow dataset lacks train/val split")
    if cond.ndim != 2 or cond.shape[1] < 128:
        failures.append(f"condition dim {cond.shape[1] if cond.ndim == 2 else 0} < 128")
    if target.size and not np.isfinite(target).all():
        failures.append("target contains NaN/Inf")

    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.rationales, args.inferability, args.grounding, args.contexts, args.embeddings_npy, args.embeddings_index], [args.output, args.metrics, args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
