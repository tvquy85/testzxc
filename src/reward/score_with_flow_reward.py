import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flow_dataset import FlowRewardDataset
from flow_model_lite import FlowRewardLite
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--judge-scores", required=True)
    parser.add_argument("--tech-features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = FlowRewardDataset(args.judge_scores, args.tech_features)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Model
    model = FlowRewardLite(cond_dim=35).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # We need to compute flow_prob_true_label.
    # The classes map to 0..4
    label_map = {
        'Strong Down': 0,
        'Mild Down': 1,
        'Neutral': 2,
        'Mild Up': 3,
        'Strong Up': 4
    }
    
    # Pre-extract labels for faster lookup
    label_5_series = dataset.df['label_5'].map(label_map).fillna(2).astype(int).values
    
    results = []
    
    with torch.no_grad():
        start_idx = 0
        for batch in loader:
            cond = batch['cond'].to(device)
            sigma = batch['sigma'].to(device)
            sample_ids = batch['sample_id']
            candidate_ids = batch['candidate_id']
            B = cond.size(0)
            
            # Start at z0
            z0 = torch.randn(B, 5, device=device) * sigma.unsqueeze(1)
            zt = z0.clone()
            
            # Euler integration
            n_steps = 10
            dt = 1.0 / n_steps
            for step in range(n_steps):
                t = torch.full((B, 1), step * dt, device=device)
                v = model(t, zt, cond)
                zt = zt + v * dt
                
            z1_pred = zt
            
            # Compute probabilities
            probs = F.softmax(z1_pred, dim=-1)
            
            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            
            # Compute prob true label
            batch_labels = label_5_series[start_idx : start_idx + B]
            # Advanced indexing to get p_true for each sample
            p_true = probs[torch.arange(B), batch_labels]
            
            # Overall reward
            overall_reward = p_true - 0.1 * entropy
            
            for i in range(B):
                results.append({
                    'sample_id': sample_ids[i],
                    'candidate_id': candidate_ids[i].item() if isinstance(candidate_ids[i], torch.Tensor) else candidate_ids[i],
                    'flow_p_strong_down': probs[i, 0].item(),
                    'flow_p_mild_down': probs[i, 1].item(),
                    'flow_p_neutral': probs[i, 2].item(),
                    'flow_p_mild_up': probs[i, 3].item(),
                    'flow_p_strong_up': probs[i, 4].item(),
                    'flow_prob_true_label': p_true[i].item(),
                    'flow_entropy': entropy[i].item(),
                    'flow_overall_reward': overall_reward[i].item()
                })
            
            start_idx += B

    # Save to parquet
    df_res = pd.DataFrame(results)
    
    # Merge with original judge scores to keep all columns
    df_judge = pd.read_parquet(args.judge_scores)
    df_out = df_judge.merge(df_res, on=['sample_id', 'candidate_id'], how='left')
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    print(f"Scored {len(df_res)} rationales. Saved to {args.output}")

if __name__ == "__main__":
    main()
