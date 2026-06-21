from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reward.evaluate_flow_vs_proxy_v4 import preference_pair_accuracy, spearman, top_decile_utility
from src.reward.flow_model_v2 import FlowRewardV2
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "12_FLOW_TRAIN_EVAL_MEDIUM"


def integrate_flow(model: FlowRewardV2, cond: np.ndarray, device: torch.device, steps: int = 12) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, cond.shape[0], 512):
            c = torch.tensor(cond[start : start + 512], dtype=torch.float32, device=device)
            z = torch.zeros((c.shape[0], model.target_dim), dtype=torch.float32, device=device)
            for idx in range(steps):
                t = torch.full((c.shape[0], 1), float(idx) / float(steps), dtype=torch.float32, device=device)
                z = z + model(t, z, c) / float(steps)
            preds.append(z.detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, model.target_dim), dtype=float)


def proxy_average(target: np.ndarray, mask: np.ndarray, target_names: list[str], utility_idx: int) -> np.ndarray:
    indices = [idx for idx, _name in enumerate(target_names) if idx != utility_idx]
    vals = np.where(mask[:, indices] > 0, target[:, indices], np.nan)
    return np.nanmean(vals, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="outputs/metrics/12_flow_vs_proxy_medium.json")
    parser.add_argument("--output-csv", default="outputs/tables/medium_flow_vs_proxy.csv")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    model_path = Path(args.checkpoint) / "model.pt" if Path(args.checkpoint).is_dir() else Path(args.checkpoint)
    if not Path(args.dataset).exists():
        failures.append(f"dataset missing: {args.dataset}")
    if not model_path.exists():
        failures.append(f"model checkpoint missing: {model_path}")
    if failures:
        metrics = {"pipeline_pass": False, "flow_claim_allowed": False, "flow_reward_improvement": False, "eval_rows": 0}
        write_json(args.output, metrics)
        write_manifest(args.manifest, [args.output], STEP)
        write_status(args.status, STEP, "FAIL", [args.dataset, str(model_path)], [args.output, args.manifest, args.status], metrics, failures, False)
        return 1

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    target = np.asarray(data["target"], dtype=np.float32)
    mask = np.asarray(data.get("mask", np.ones_like(target)), dtype=np.float32)
    cond = np.asarray(data["cond"], dtype=np.float32)
    names = list(data.get("target_names", []))
    utility_idx = names.index("utility_proxy") if "utility_proxy" in names else target.shape[1] - 1
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowRewardV2(cond_dim=cond.shape[1], target_dim=target.shape[1]).to(device)
    model.load_state_dict(state["model_state_dict"])
    pred = integrate_flow(model, cond, device)
    df = pd.DataFrame(
        {
            "sample_id": data.get("sample_id", list(range(len(cond)))),
            "candidate_id": data.get("candidate_id", [0] * len(cond)),
            "split": data.get("split", [""] * len(cond)),
            "track": data.get("track", [""] * len(cond)),
            "realized_utility": target[:, utility_idx],
            "utility_mask": mask[:, utility_idx],
            "flow_reward": pred[:, utility_idx],
            "proxy_reward": data.get("proxy_reward", proxy_average(target, mask, names, utility_idx)),
        }
    )
    eval_df = df[df["split"].eq(args.split) & (df["utility_mask"] > 0)].copy()
    if len(eval_df) < 100:
        failures.append(f"eval rows {len(eval_df)} < 100")
    flow_rank = spearman(eval_df["flow_reward"], eval_df["realized_utility"])
    proxy_rank = spearman(eval_df["proxy_reward"], eval_df["realized_utility"])
    flow_pref = preference_pair_accuracy(eval_df, "flow_reward", "realized_utility")
    proxy_pref = preference_pair_accuracy(eval_df, "proxy_reward", "realized_utility")
    flow_top = top_decile_utility(eval_df, "flow_reward", "realized_utility")
    proxy_top = top_decile_utility(eval_df, "proxy_reward", "realized_utility")
    wins = {
        "rank_correlation_with_realized_utility": (flow_rank or 0.0) >= (proxy_rank or 0.0) + 0.03,
        "preference_pair_accuracy": flow_pref is not None and proxy_pref is not None and flow_pref >= proxy_pref + 0.02,
        "top_decile_realized_utility": flow_top is not None and proxy_top is not None and flow_top >= proxy_top,
    }
    allowed = sum(bool(v) for v in wins.values()) >= 2
    metrics = {
        "pipeline_pass": not failures,
        "claim_allowed": bool(allowed),
        "rows": int(len(df)),
        "eval_rows": int(len(eval_df)),
        "eval_split": args.split,
        "methods": {
            "flow_reward_v5": {
                "rank_correlation_with_realized_utility": float(flow_rank or 0.0),
                "preference_pair_accuracy": None if flow_pref is None else float(flow_pref),
                "top_decile_realized_utility": None if flow_top is None else float(flow_top),
            },
            "proxy_average_reward": {
                "rank_correlation_with_realized_utility": float(proxy_rank or 0.0),
                "preference_pair_accuracy": None if proxy_pref is None else float(proxy_pref),
                "top_decile_realized_utility": None if proxy_top is None else float(proxy_top),
            },
        },
        "metric_wins": wins,
        "flow_reward_improvement": bool(allowed),
        "flow_claim_allowed": bool(allowed),
    }
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    write_json(args.output, metrics)
    write_manifest(args.manifest, [args.dataset, str(model_path), args.output, args.output_csv], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.dataset, str(model_path)], [args.output, args.output_csv, args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
