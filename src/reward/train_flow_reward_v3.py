import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reward.flow_model_v2 import FlowRewardV2, masked_mse
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "11_FLOW_TRAIN_EVAL_FIX_VALID_SPLIT"

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--status", default=f"outputs/status/{STEP}_TRAIN.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}_TRAIN.manifest.json")
    args = parser.parse_args()

    failures = []
    if not Path(args.dataset).exists():
        failures.append(f"dataset missing: {args.dataset}")
        metrics = {"epochs_completed": 0, "failures": failures}
        write_json(args.metrics, metrics)
        write_manifest(args.manifest, [args.metrics], STEP)
        write_status(args.status, STEP, "FAIL", [args.dataset], [args.metrics, args.manifest, args.status], metrics, failures, False)
        return 1

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    target = torch.tensor(data["target"], dtype=torch.float32)
    mask = torch.tensor(data.get("mask", np.ones_like(data["target"])), dtype=torch.float32)
    cond = torch.tensor(data["cond"], dtype=torch.float32)
    
    if target.shape[0] == 0:
        failures.append("cannot train flow reward on empty dataset")
        metrics = {"epochs_completed": 0, "rows": 0, "failures": failures}
        write_json(args.metrics, metrics)
        write_manifest(args.manifest, [args.metrics], STEP)
        write_status(args.status, STEP, "FAIL", [args.dataset], [args.metrics, args.manifest, args.status], metrics, failures, False)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TensorDataset(target, mask, cond)
    
    val_size = max(1, int(len(ds) * args.holdout_frac))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FlowRewardV2(cond_dim=cond.shape[1], target_dim=target.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
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
            train_total += float(loss.detach().cpu()) * z1.shape[0]
            
        train_loss = train_total / train_size
        train_losses.append(train_loss)
        
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for z1, m, c in val_loader:
                z1, m, c = z1.to(device), m.to(device), c.to(device)
                z0 = torch.randn_like(z1)
                t = torch.rand(z1.shape[0], 1, device=device)
                zt = t * z1 + (1 - t) * z0
                pred = model(t, zt, c)
                loss = masked_mse(pred, z1 - z0, m)
                val_total += float(loss.detach().cpu()) * z1.shape[0]
                
        val_loss = val_total / val_size
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "cond_dim": cond.shape[1],
                    "target_dim": target.shape[1],
                    "target_names": data.get("target_names", [])
                }
            }, Path(args.output_dir) / "model.pt")

    if len(val_losses) >= 2 and val_losses[-1] > val_losses[0]:
        failures.append("Holdout loss did not decrease")

    metrics = {
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_completed": args.epochs,
        "rows": int(target.shape[0]),
        "train_rows": int(train_size),
        "val_rows": int(val_size),
        "holdout_loss_decreased": bool(len(val_losses) < 2 or val_losses[-1] <= val_losses[0]),
    }

    write_json(args.metrics, metrics)
    model_path = str(Path(args.output_dir) / "model.pt")
    write_manifest(args.manifest, [args.dataset, args.metrics, model_path], STEP)
    if not Path(model_path).exists():
        failures.append(f"missing model checkpoint: {model_path}")
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.dataset],
        [args.metrics, model_path, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )

    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
