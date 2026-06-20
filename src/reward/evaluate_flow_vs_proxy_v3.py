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
from src.reward.flow_model_v2 import FlowRewardV2
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "11_FLOW_TRAIN_EVAL_FIX_VALID_SPLIT"


def spearman(x: Any, y: Any) -> float | None:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 2 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return None
    return float(df["x"].rank().corr(df["y"].rank()))


def preference_pair_accuracy(df: pd.DataFrame, score_col: str, utility_col: str) -> float | None:
    correct = 0
    total = 0
    for _, group in df.dropna(subset=[score_col, utility_col]).groupby("sample_id"):
        if len(group) < 2:
            continue
        records = group[[score_col, utility_col]].to_numpy(dtype=float)
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                utility_delta = records[i, 1] - records[j, 1]
                if abs(utility_delta) < 1e-9:
                    continue
                score_delta = records[i, 0] - records[j, 0]
                if abs(score_delta) < 1e-9:
                    continue
                total += 1
                correct += int((utility_delta > 0) == (score_delta > 0))
    return float(correct / total) if total else None


def top_decile_utility(df: pd.DataFrame, score_col: str, utility_col: str) -> float | None:
    clean = df[[score_col, utility_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 10 or clean[score_col].nunique() < 2:
        return None
    cutoff = clean[score_col].quantile(0.90)
    selected = clean[clean[score_col] >= cutoff]
    return float(selected[utility_col].mean()) if len(selected) else None


def stable_holdout_mask(sample_ids: list[Any], frac: float, seed: int) -> np.ndarray:
    threshold = max(0, min(10_000, int(frac * 10_000)))
    out = []
    for value in sample_ids:
        digest = hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8).digest()
        out.append(int.from_bytes(digest[:4], "little") % 10_000 < threshold)
    return np.asarray(out, dtype=bool)


def integrate_flow(model: FlowRewardV2, cond: np.ndarray, steps: int = 12) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, cond.shape[0], 512):
            c = torch.tensor(cond[start : start + 512], dtype=torch.float32)
            z = torch.zeros((c.shape[0], model.target_dim), dtype=torch.float32)
            for idx in range(steps):
                t = torch.full((c.shape[0], 1), float(idx) / float(steps), dtype=torch.float32)
                z = z + model(t, z, c) / float(steps)
            preds.append(z.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, model.target_dim), dtype=float)


def proxy_average(target: np.ndarray, mask: np.ndarray, names: list[str], utility_idx: int) -> np.ndarray:
    proxy_names = [
        "independent_true_label_prob",
        "inferability_certainty",
        "debias_stability_score",
        "news_grounding_score",
        "technical_grounding_score",
        "calibration_proxy",
    ]
    indices = [names.index(name) for name in proxy_names if name in names and names.index(name) != utility_idx]
    if not indices:
        return np.zeros(target.shape[0], dtype=float)
    vals = target[:, indices].astype(float)
    m = mask[:, indices].astype(bool)
    vals = np.where(m, vals, np.nan)
    return np.nanmean(vals, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.dataset).exists():
        failures.append(f"Missing dataset: {args.dataset}")
    if not Path(args.model).exists():
        failures.append(f"Missing model: {args.model}")
    if failures:
        metrics = {"methods": {}, "flow_claim_allowed": False}
        write_json(args.output, metrics)
        write_manifest(args.manifest, [args.output], STEP)
        write_status(args.status, STEP, "FAIL", [args.dataset, args.model], [args.output, args.manifest, args.status], metrics, failures, False)
        return 1

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    target = np.asarray(data["target"], dtype=np.float32)
    mask = np.asarray(data.get("mask", np.ones_like(target)), dtype=np.float32)
    cond = np.asarray(data["cond"], dtype=np.float32)
    target_names = list(data.get("target_names", []))
    if "utility_proxy" not in target_names:
        failures.append("utility_proxy not found in target_names")
        utility_idx = 0
    else:
        utility_idx = target_names.index("utility_proxy")

    state = torch.load(args.model, map_location="cpu", weights_only=False)
    model = FlowRewardV2(cond_dim=cond.shape[1], target_dim=target.shape[1])
    model.load_state_dict(state["model_state_dict"])
    pred = integrate_flow(model, cond)

    sample_ids = data.get("sample_id", list(range(len(cond))))
    holdout = stable_holdout_mask(sample_ids, args.holdout_frac, args.seed)
    if int(holdout.sum()) < 20:
        holdout[:] = False
        holdout[-min(len(holdout), max(1, int(len(holdout) * args.holdout_frac))):] = True

    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "candidate_id": data.get("candidate_id", np.zeros(len(cond))),
            "realized_utility": target[:, utility_idx],
            "utility_mask": mask[:, utility_idx],
            "flow_reward": pred[:, utility_idx],
            "proxy_reward": proxy_average(target, mask, target_names, utility_idx),
            "holdout": holdout,
        }
    )
    eval_df = df[df["holdout"] & (df["utility_mask"] > 0)].copy()
    if len(eval_df) < 20:
        failures.append(f"holdout eval rows {len(eval_df)} < 20")

    flow_rank = spearman(eval_df["flow_reward"], eval_df["realized_utility"]) or 0.0
    flow_pref = preference_pair_accuracy(eval_df, "flow_reward", "realized_utility") or 0.0
    flow_top = top_decile_utility(eval_df, "flow_reward", "realized_utility") or 0.0
    proxy_rank = spearman(eval_df["proxy_reward"], eval_df["realized_utility"]) or 0.0
    proxy_pref = preference_pair_accuracy(eval_df, "proxy_reward", "realized_utility") or 0.0
    proxy_top = top_decile_utility(eval_df, "proxy_reward", "realized_utility") or 0.0

    wins = {
        "rank_correlation_with_realized_utility": flow_rank >= proxy_rank + 0.03,
        "preference_pair_accuracy": flow_pref >= proxy_pref + 0.02,
        "top_decile_realized_utility": flow_top >= proxy_top,
    }
    flow_claim_allowed = sum(bool(v) for v in wins.values()) >= 2
    metrics = {
        "rows": int(len(df)),
        "eval_rows": int(len(eval_df)),
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "methods": {
            "flow_reward_v3_1": {
                "rank_correlation_with_realized_utility": float(flow_rank),
                "preference_pair_accuracy": float(flow_pref),
                "top_decile_realized_utility": float(flow_top),
            },
            "proxy_average_reward": {
                "rank_correlation_with_realized_utility": float(proxy_rank),
                "preference_pair_accuracy": float(proxy_pref),
                "top_decile_realized_utility": float(proxy_top),
            },
        },
        "metric_wins": wins,
        "flow_claim_allowed": bool(flow_claim_allowed),
    }

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
    write_json(args.output, metrics)
    outputs = [args.output] + ([args.output_csv] if args.output_csv else [])
    write_manifest(args.manifest, [args.dataset, args.model, *outputs], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.dataset, args.model], [*outputs, args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
