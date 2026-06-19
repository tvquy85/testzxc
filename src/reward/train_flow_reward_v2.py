from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reward.flow_model_v2 import FlowRewardV2, masked_mse
from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "12_FLOW_REWARD_MULTITARGET_V2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/reward/flow_v2_train_dataset.pt")
    parser.add_argument("--output-dir", default="checkpoints/flow_reward_v2")
    parser.add_argument("--metrics", default="outputs/metrics/flow_reward_v2_train_metrics.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    failures: list[str] = []
    if not Path(args.dataset).exists():
        failures.append(f"dataset missing: {args.dataset}")
        data = {"target": torch.zeros((0, 0)), "mask": torch.zeros((0, 0)), "cond": torch.zeros((0, 0)), "sigma": torch.zeros((0,)), "target_names": []}
    else:
        data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    target = torch.tensor(data["target"], dtype=torch.float32)
    mask = torch.tensor(data["mask"], dtype=torch.float32)
    cond = torch.tensor(data["cond"], dtype=torch.float32)
    sigma = torch.tensor(data["sigma"], dtype=torch.float32)
    if target.shape[0] == 0:
        failures.append("cannot train flow reward on empty dataset")
    losses = []
    if not failures:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds = TensorDataset(target, mask, cond, sigma)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        model = FlowRewardV2(cond_dim=cond.shape[1], target_dim=target.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(args.epochs):
            total = 0.0
            count = 0
            for z1, m, c, s in loader:
                z1, m, c, s = z1.to(device), m.to(device), c.to(device), s.to(device)
                z0 = torch.randn_like(z1) * s.unsqueeze(1)
                t = torch.rand(z1.shape[0], 1, device=device)
                zt = t * z1 + (1 - t) * z0
                pred = model(t, zt, c)
                loss = masked_mse(pred, z1 - z0, m)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * z1.shape[0]
                count += z1.shape[0]
            losses.append(total / max(1, count))
        if len(losses) >= 2 and losses[-1] > losses[0]:
            failures.append("training loss did not decrease versus first epoch")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": {"cond_dim": cond.shape[1], "target_dim": target.shape[1], "target_names": data["target_names"]}}, Path(args.output_dir) / "model.pt")
    metrics = {"train_loss_by_epoch": losses, "rows": int(target.shape[0]), "target_dim": int(target.shape[1]) if target.ndim == 2 else 0}
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.dataset, args.metrics, str(Path(args.output_dir) / "model.pt")], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.dataset],
        outputs_created=[args.metrics, args.manifest, args.status] + ([str(Path(args.output_dir) / "model.pt")] if not failures else []),
        metrics=metrics,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps(metrics, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
