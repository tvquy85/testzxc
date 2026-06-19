import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from src.eval.classification_metrics import compute_metrics
from src.utils.config import load_config
from sklearn.linear_model import LogisticRegression

def chunk_texts(texts, tokenizer, max_length=512):
    # simple truncation
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default="data/labels/aligned_samples_h1.parquet")
    parser.add_argument("--split", type=str, default="data/processed/split_h1.parquet")
    parser.add_argument("--config", type=str, default="configs/local_paths.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/metrics/finbert_baseline_h1.json")
    args = parser.parse_args()

    config = load_config(args.config)
    hf_home = config.get("hf_home")
    if hf_home and "$" not in str(hf_home):
        os.environ.setdefault("HF_HOME", str(hf_home))
    model_path = config.get("models", {}).get("finbert")

    df = pd.read_parquet(args.samples)
    split_df = pd.read_parquet(args.split)
    df = pd.merge(df, split_df, on="sample_id")

    if args.limit:
        df = df.sample(n=min(len(df), args.limit), random_state=42)

    if os.path.exists("data/indicators/finbert_features_h1.parquet"):
        print("Loading existing FinBERT features...")
        finbert_df = pd.read_parquet("data/indicators/finbert_features_h1.parquet")
        df = pd.merge(df, finbert_df, on="sample_id", how="left")
        # Ensure fallback for missing
        for col in ["finbert_0", "finbert_1", "finbert_2"]:
            if col not in df.columns:
                df[col] = 0.33
    else:
        # Fallback if not found (should not happen)
        print("FinBERT features not found, training will fail.")
        return

    # Train a classifier on FinBERT features
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    
    # Drop rows with NaNs in finbert columns just in case
    train = train.dropna(subset=["finbert_0", "finbert_1", "finbert_2"])
    test = test.dropna(subset=["finbert_0", "finbert_1", "finbert_2"])
    
    X_train = train[["finbert_0", "finbert_1", "finbert_2"]].values
    y_train = train["label_5"].values
    
    X_test = test[["finbert_0", "finbert_1", "finbert_2"]].values
    y_test = test["label_5"].values
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    classes = clf.classes_
    metrics = compute_metrics(y_test, y_pred, y_prob, classes)
    
    res = {
        "model": "FinBERT_LR",
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
