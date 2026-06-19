from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_manifest, write_status


STEP = "12_FLOW_REWARD_MULTITARGET_V2"
TARGET_NAMES = [
    "inferability_true_label_prob",
    "multi_judge_agreement",
    "news_grounding_score",
    "technical_grounding_score",
    "counterfactual_directional_score",
    "utility_score",
    "calibration_proxy",
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


def build_dataset(rationales, inferability, grounding, features):
    import pandas as pd

    df = rationales.copy()
    if "sample_id" not in df.columns:
        target = np.zeros((0, len(TARGET_NAMES)), dtype=np.float32)
        return df, {
            "target": target,
            "mask": target.copy(),
            "cond": np.zeros((0, 67), dtype=np.float32),
            "sigma": np.zeros((0,), dtype=np.float32),
            "target_names": TARGET_NAMES,
            "cond_dim": 67,
        }
    if not inferability.empty:
        inf = inferability.groupby(["sample_id", "candidate_id"], dropna=False).agg(
            inferability_true_label_prob=("inferability_true_label_prob", "mean"),
            multi_judge_agreement=("inferability_true_label_prob", "std"),
        ).reset_index()
        inf["multi_judge_agreement"] = 1.0 - inf["multi_judge_agreement"].fillna(0.0).clip(0, 1)
        df = df.merge(inf, on=["sample_id", "candidate_id"], how="left")
    if not grounding.empty:
        g = grounding.groupby(["sample_id", "candidate_id"], dropna=False).agg(
            news_grounding_score=("news_grounding_score", "mean"),
            technical_grounding_score=("technical_grounding_score", "mean"),
        ).reset_index()
        df = df.merge(g, on=["sample_id", "candidate_id"], how="left")
    if not features.empty:
        df = df.merge(features, on="sample_id", how="left")

    for name in TARGET_NAMES:
        if name not in df.columns:
            df[name] = np.nan
    target = df[TARGET_NAMES].astype(float).to_numpy(dtype=np.float32) if len(df) else np.zeros((0, len(TARGET_NAMES)), dtype=np.float32)
    mask = np.isfinite(target).astype(np.float32)
    target = np.nan_to_num(target, nan=0.0)
    text = (df.get("raw_text", "").fillna("") if len(df) else [])
    text_emb = np.vstack([hash_embedding(t) for t in text]) if len(df) else np.zeros((0, 64), dtype=np.float32)
    tech_cols = [c for c in features.columns if c not in {"sample_id", "ticker", "event_date", "window_end_date", "regime_label"}] if not features.empty else []
    tech = df[tech_cols].select_dtypes(include=[np.number]).fillna(0.0).to_numpy(dtype=np.float32) if len(df) and tech_cols else np.zeros((len(df), 0), dtype=np.float32)
    regimes = df.get("regime_label", pd.Series(["normal_vol"] * len(df))).fillna("normal_vol") if len(df) else []
    regime_map = {"low_vol": [1, 0, 0], "normal_vol": [0, 1, 0], "high_vol": [0, 0, 1]}
    regime_one_hot = np.asarray([regime_map.get(r, [0, 1, 0]) for r in regimes], dtype=np.float32) if len(df) else np.zeros((0, 3), dtype=np.float32)
    sigma_map = {"low_vol": 0.5, "normal_vol": 1.0, "high_vol": 1.5}
    sigma = np.asarray([sigma_map.get(r, 1.0) for r in regimes], dtype=np.float32) if len(df) else np.zeros((0,), dtype=np.float32)
    cond = np.concatenate([text_emb, tech, regime_one_hot], axis=1) if len(df) else np.zeros((0, 67), dtype=np.float32)
    return df, {"target": target, "mask": mask, "cond": cond, "sigma": sigma, "target_names": TARGET_NAMES, "cond_dim": int(cond.shape[1])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", default="data/rationales/parsed/train_candidates_strict.parquet")
    parser.add_argument("--inferability", default="data/judges/inferability_multi_judge.parquet")
    parser.add_argument("--grounding", default="data/judges/claim_grounding_scores.parquet")
    parser.add_argument("--features", default="data/indicators/technical_features_h1_v2.parquet")
    parser.add_argument("--output", default="data/reward/flow_v2_train_dataset.pt")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd
    import torch

    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if Path(args.rationales).exists() else pd.DataFrame()
    inferability = pd.read_parquet(args.inferability) if Path(args.inferability).exists() else pd.DataFrame()
    grounding = pd.read_parquet(args.grounding) if Path(args.grounding).exists() else pd.DataFrame()
    features = pd.read_parquet(args.features) if Path(args.features).exists() else pd.DataFrame()
    df, data = build_dataset(rationales, inferability, grounding, features)
    if len(df) == 0:
        failures.append("flow v2 dataset has zero rows")
    if len(TARGET_NAMES) < 5:
        failures.append("target dimension must be >= 5")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({**data, "sample_id": df["sample_id"].tolist() if len(df) else [], "candidate_id": df["candidate_id"].tolist() if len(df) and "candidate_id" in df else []}, args.output)
    write_manifest(args.manifest, [args.output], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.rationales, args.inferability, args.grounding, args.features],
        outputs_created=[args.output, args.manifest, args.status],
        metrics={"rows": int(len(df)), "target_dim": len(TARGET_NAMES), "cond_dim": int(data["cond_dim"])},
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps({"rows": len(df), "target_dim": len(TARGET_NAMES), "cond_dim": data["cond_dim"]}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
