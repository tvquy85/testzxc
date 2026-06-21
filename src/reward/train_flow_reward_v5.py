from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reward.flow_model_v2 import FlowRewardV2, masked_mse
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "12_FLOW_TRAIN_MEDIUM"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/models/flow_reward_v5_medium")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", default="outputs/metrics/12_flow_train_medium.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.dataset).exists():
        failures.append(f"dataset missing: {args.dataset}")
        data = {"target": np.zeros((0, 1)), "mask": np.zeros((0, 1)), "cond": np.zeros((0, 1)), "split": []}
    else:
        data = torch.load(args.dataset, map_location="cpu", weights_only=False)

    target = torch.tensor(data["target"], dtype=torch.float32)
    mask = torch.tensor(data.get("mask", np.ones_like(data["target"])), dtype=torch.float32)
    cond = torch.tensor(data["cond"], dtype=torch.float32)
    split = np.asarray(data.get("split", ["train"] * len(target)))
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        failures.append("train/val split missing for flow training")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.output_dir) / "model.pt"
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not failures:
        torch.manual_seed(args.seed)
        train_ds = TensorDataset(target[train_idx], mask[train_idx], cond[train_idx])
        val_ds = TensorDataset(target[val_idx], mask[val_idx], cond[val_idx])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        model = FlowRewardV2(cond_dim=cond.shape[1], target_dim=target.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for _epoch in range(args.epochs):
            model.train()
            total = 0.0
            for z1, m, c in train_loader:
                z1, m, c = z1.to(device), m.to(device), c.to(device)
                z0 = torch.randn_like(z1)
                t = torch.rand(z1.shape[0], 1, device=device)
                zt = t * z1 + (1 - t) * z0
                pred = model(t, zt, c)
                loss = masked_mse(pred, z1 - z0, m)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * z1.shape[0]
            train_losses.append(total / max(1, len(train_ds)))
            model.eval()
            vtotal = 0.0
            with torch.no_grad():
                for z1, m, c in val_loader:
                    z1, m, c = z1.to(device), m.to(device), c.to(device)
                    z0 = torch.randn_like(z1)
                    t = torch.rand(z1.shape[0], 1, device=device)
                    zt = t * z1 + (1 - t) * z0
                    loss = masked_mse(model(t, zt, c), z1 - z0, m)
                    vtotal += float(loss.detach().cpu()) * z1.shape[0]
            val = vtotal / max(1, len(val_ds))
            val_losses.append(val)
            if val < best_val:
                best_val = val
                torch.save({"model_state_dict": model.state_dict(), "config": {"cond_dim": cond.shape[1], "target_dim": target.shape[1], "target_names": data.get("target_names", [])}}, model_path)

    if not model_path.exists():
        failures.append(f"missing model checkpoint: {model_path}")
    metrics = {
        "pipeline_pass": not failures,
        "claim_allowed": False,
        "rows": int(target.shape[0]),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "epochs_completed": int(args.epochs if model_path.exists() else 0),
        "best_val_loss": None if best_val == float("inf") else float(best_val),
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "loss_decreased": bool(len(val_losses) < 2 or val_losses[-1] <= val_losses[0]),
        "device": str(device),
    }
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.dataset, args.metrics, str(model_path)], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.dataset], [args.metrics, str(model_path), args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
