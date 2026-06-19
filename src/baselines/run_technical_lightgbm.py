import os
import json
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.eval.classification_metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default="data/labels/aligned_samples_h1.parquet")
    parser.add_argument("--features", type=str, default="data/indicators/technical_features_h1.parquet")
    parser.add_argument("--split", type=str, default="data/processed/split_h1.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/metrics/technical_lgbm_h1.json")
    args = parser.parse_args()

    samples = pd.read_parquet(args.samples)
    features = pd.read_parquet(args.features)
    split_df = pd.read_parquet(args.split)
    
    df = pd.merge(samples, features, on="sample_id")
    df = pd.merge(df, split_df, on="sample_id")

    if args.limit:
        df = df.sample(n=min(len(df), args.limit), random_state=42)

    feature_cols = [c for c in features.columns if c not in ["sample_id", "ticker", "event_date", "regime_label", "window_end_date"]]
    
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    
    X_train = train[feature_cols].values
    y_train = train["label_5"].values
    
    X_test = test[feature_cols].values
    y_test = test["label_5"].values
    
    clf = lgb.LGBMClassifier(random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    classes = clf.classes_
    metrics = compute_metrics(y_test, y_pred, y_prob, classes)
    
    res = {
        "model": "Technical_LightGBM",
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
