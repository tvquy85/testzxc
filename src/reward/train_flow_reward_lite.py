import argparse
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from flow_dataset import FlowRewardDataset
from flow_model_lite import FlowRewardLite
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-scores", required=True)
    parser.add_argument("--tech-features", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    full_dataset = FlowRewardDataset(args.judge_scores, args.tech_features)
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    # cond dim is tech (26) + regime (3) + grounding (6) = 35
    model = FlowRewardLite(cond_dim=35).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Compute naive baseline: mean of z1 on train set
    z1_sum = torch.zeros(5)
    for batch in train_loader:
        z1_sum += batch['z1'].sum(dim=0)
    z1_mean = (z1_sum / train_size).to(device)
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            z1 = batch['z1'].to(device)
            cond = batch['cond'].to(device)
            sigma = batch['sigma'].to(device)
            B = z1.size(0)
            
            # Flow matching
            z0 = torch.randn_like(z1) * sigma.unsqueeze(1)
            t = torch.rand(B, 1).to(device)
            zt = t * z1 + (1 - t) * z0
            
            v_true = z1 - z0
            v_pred = model(t, zt, cond)
            
            loss = F.mse_loss(v_pred, v_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * B
            
        print(f"Epoch {epoch+1}/{args.epochs} - Train MSE: {epoch_loss / train_size:.4f}")
        
    # Validation
    model.eval()
    val_loss = 0.0
    naive_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            z1 = batch['z1'].to(device)
            cond = batch['cond'].to(device)
            sigma = batch['sigma'].to(device)
            B = z1.size(0)
            
            z0 = torch.randn_like(z1) * sigma.unsqueeze(1)
            t = torch.rand(B, 1).to(device)
            zt = t * z1 + (1 - t) * z0
            
            v_true = z1 - z0
            v_pred = model(t, zt, cond)
            v_naive = z1_mean.unsqueeze(0).expand(B, -1)
            
            val_loss += F.mse_loss(v_pred, v_true).item() * B
            naive_loss += F.mse_loss(v_naive, v_true).item() * B
            
    val_mse = val_loss / val_size
    naive_mse = naive_loss / val_size
    print(f"Final Val MSE: {val_mse:.4f} (Naive: {naive_mse:.4f})")
    
    # Save checkpoint
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    
    # Save metrics
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    metrics = {
        "validation_mse": val_mse,
        "naive_baseline_mse": naive_mse
    }
    with open(args.metrics, 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
