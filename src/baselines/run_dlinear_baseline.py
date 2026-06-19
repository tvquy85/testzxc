import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.eval.classification_metrics import compute_metrics

class LinearBaseline(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default="data/labels/aligned_samples_h1.parquet")
    parser.add_argument("--features", type=str, default="data/indicators/technical_features_h1.parquet")
    parser.add_argument("--split", type=str, default="data/processed/split_h1.parquet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="outputs/metrics/dlinear_baseline_h1.json")
    args = parser.parse_args()

    samples = pd.read_parquet(args.samples)
    features = pd.read_parquet(args.features)
    split_df = pd.read_parquet(args.split)
    
    df = pd.merge(samples, features, on="sample_id")
    df = pd.merge(df, split_df, on="sample_id")

    # Drop non-numeric / irrelevant columns
    exclude_cols = ["sample_id", "ticker", "event_date", "regime_label", "window_end_date", "headline", "body", "stock_return_h1", "market_return_h1", "abnormal_return_h1", "label_5", "timestamp_utc", "split"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float32, np.float64, np.int32, np.int64, bool]]
    
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    
    # Simple standardization
    X_train_raw = train[feature_cols].values.astype(np.float32)
    X_test_raw = test[feature_cols].values.astype(np.float32)
    
    # Handle NaNs
    X_train_raw = np.nan_to_num(X_train_raw)
    X_test_raw = np.nan_to_num(X_test_raw)
    
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0) + 1e-8
    
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    
    # Label mapping
    unique_labels = sorted(df["label_5"].unique())
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    
    y_train = np.array([label_to_idx[l] for l in train["label_5"].values])
    y_test = np.array([label_to_idx[l] for l in test["label_5"].values])
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearBaseline(X_train.shape[1], len(unique_labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model.train()
    for epoch in range(args.epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    all_preds = []
    all_probs = []
    X_test_tensor = torch.tensor(X_test).to(device)
    
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
    y_test_str = test["label_5"].values
    y_pred_str = [idx_to_label[p] for p in preds]
    
    metrics = compute_metrics(y_test_str, y_pred_str, probs, unique_labels)
    
    res = {
        "model": "DLinear_Tabular",
        "test_accuracy": metrics.get("accuracy"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_mcc": metrics.get("mcc"),
        "test_brier": metrics.get("brier_score_multiclass")
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
        
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
