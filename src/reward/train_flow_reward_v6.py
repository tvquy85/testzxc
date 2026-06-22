from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reward.flow_model_v2 import FlowRewardV2
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "11_FLOW_TRAIN_V6"


def weighted_masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() == 1:
        weight = weight.unsqueeze(1)
    denom = (mask * weight).sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask * weight).sum() / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.dataset).exists():
        failures.append(f"dataset missing: {args.dataset}")
        data = {
            "target": np.zeros((0, 5), dtype=np.float32),
            "mask": np.zeros((0, 5), dtype=np.float32),
            "cond": np.zeros((0, 128), dtype=np.float32),
            "split": [],
            "target_names": [],
            "auxiliary": {},
        }
    else:
        data = torch.load(args.dataset, map_location="cpu", weights_only=False)

    target = torch.tensor(data["target"], dtype=torch.float32)
    mask = torch.tensor(data.get("mask", np.ones_like(data["target"])), dtype=torch.float32)
    cond = torch.tensor(data["cond"], dtype=torch.float32)
    split = np.asarray(data.get("split", ["train"] * len(target)))
    weights = np.asarray(data.get("auxiliary", {}).get("judge_reliability_weight", np.ones(len(target))), dtype=np.float32)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0, 1.0)
    weight = torch.tensor(weights, dtype=torch.float32)

    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        failures.append("train/val split missing for flow training")
    if target.ndim != 2 or target.shape[1] != 5:
        failures.append(f"target shape invalid: {tuple(target.shape)}")
    if cond.ndim != 2 or cond.shape[0] != target.shape[0]:
        failures.append(f"condition shape invalid: {tuple(cond.shape)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not failures:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        train_ds = TensorDataset(target[train_idx], mask[train_idx], cond[train_idx], weight[train_idx])
        val_ds = TensorDataset(target[val_idx], mask[val_idx], cond[val_idx], weight[val_idx])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        model = FlowRewardV2(cond_dim=cond.shape[1], target_dim=target.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for z1, m, c, w in train_loader:
                z1, m, c, w = z1.to(device), m.to(device), c.to(device), w.to(device)
                z0 = torch.randn_like(z1)
                t = torch.rand(z1.shape[0], 1, device=device)
                zt = t * z1 + (1 - t) * z0
                pred = model(t, zt, c)
                loss = weighted_masked_mse(pred, z1 - z0, m, w)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * z1.shape[0]
            train_losses.append(total / max(1, len(train_ds)))

            model.eval()
            vtotal = 0.0
            with torch.no_grad():
                for z1, m, c, w in val_loader:
                    z1, m, c, w = z1.to(device), m.to(device), c.to(device), w.to(device)
                    z0 = torch.randn_like(z1)
                    t = torch.rand(z1.shape[0], 1, device=device)
                    zt = t * z1 + (1 - t) * z0
                    loss = weighted_masked_mse(model(t, zt, c), z1 - z0, m, w)
                    vtotal += float(loss.detach().cpu()) * z1.shape[0]
            val = vtotal / max(1, len(val_ds))
            val_losses.append(val)
            if val < best_val:
                best_val = val
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "cond_dim": int(cond.shape[1]),
                            "target_dim": int(target.shape[1]),
                            "target_names": data.get("target_names", []),
                            "weighted_by": "judge_reliability_weight",
                        },
                    },
                    args.output,
                )

    if not Path(args.output).exists():
        failures.append(f"missing model checkpoint: {args.output}")

    metrics = {
        "pipeline_pass": not failures,
        "claim_allowed": False,
        "rows": int(target.shape[0]) if target.ndim == 2 else 0,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "epochs_completed": int(args.epochs if Path(args.output).exists() else 0),
        "best_val_loss": None if best_val == float("inf") else float(best_val),
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "loss_decreased": bool(len(val_losses) < 2 or val_losses[-1] <= val_losses[0]),
        "device": str(device),
        "weighted_by": "judge_reliability_weight",
    }
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.dataset, args.metrics, args.output], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.dataset], [args.output, args.metrics, args.manifest, args.status], metrics, failures, status == "PASS")
    print(json.dumps({"status": status, "failures": failures, "best_val_loss": metrics["best_val_loss"]}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
