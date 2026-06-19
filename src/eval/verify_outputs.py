import os
import json
import pandas as pd
import glob

def verify_all():
    print("=== VERIFYING STEP 03 ===")
    news_path = "data/processed/news_mvp.parquet"
    prices_path = "data/processed/prices_mvp.parquet"
    if os.path.exists(news_path) and os.path.exists(prices_path):
        news = pd.read_parquet(news_path)
        prices = pd.read_parquet(prices_path)
        print(f"[PASS] News shape: {news.shape}, Prices shape: {prices.shape}")
        if news['ticker'].nunique() >= 10 and prices['ticker'].nunique() >= 10:
            print(f"[PASS] Tickers count: {news['ticker'].nunique()}")
        else:
            print(f"[FAIL] Not enough tickers")
    else:
        print("[FAIL] Missing MVP parquet files")

    print("\n=== VERIFYING STEP 04 ===")
    aligned_path = "data/labels/aligned_samples_h1.parquet"
    label_dist_path = "data/labels/label_distribution_h1.csv"
    if os.path.exists(aligned_path) and os.path.exists(label_dist_path):
        s = pd.read_parquet(aligned_path)
        print(f"[PASS] Aligned samples shape: {s.shape}")
        if len(s) >= 5000:
            print(f"[PASS] Sample count >= 5000 ({len(s)})")
        else:
            print(f"[FAIL] Sample count too low ({len(s)})")
            
        if s['label_5'].nunique() >= 3:
            print(f"[PASS] Label classes count: {s['label_5'].nunique()}")
        else:
            print("[FAIL] Not enough label classes")
            
        if "leakage_phrase_flag" in s.columns:
            print(f"[PASS] Leakage flags created. Average leakage: {s['leakage_phrase_flag'].mean()*100:.2f}%")
        else:
            print("[FAIL] Leakage flags missing")
    else:
        print("[FAIL] Missing aligned parquet or distribution csv")

    print("\n=== VERIFYING STEP 05 ===")
    features_path = "data/indicators/technical_features_h1.parquet"
    tokens_path = "data/indicators/technical_event_tokens_h1.parquet"
    if os.path.exists(features_path) and os.path.exists(tokens_path):
        f = pd.read_parquet(features_path)
        t = pd.read_parquet(tokens_path)
        print(f"[PASS] Features shape: {f.shape}, Tokens shape: {t.shape}")
        
        rsi_valid = f['RSI_14'].between(0, 100).mean()
        if rsi_valid > 0.95:
            print(f"[PASS] RSI valid for {rsi_valid*100:.2f}% of samples")
        else:
            print(f"[FAIL] RSI validation failed")
            
        tokens_valid = t['technical_event_tokens'].notna().mean()
        if tokens_valid > 0.95:
            print(f"[PASS] Tokens valid for {tokens_valid*100:.2f}% of samples")
        else:
            print(f"[FAIL] Tokens validation failed")
            
        if "regime_label" in t.columns and t["regime_label"].nunique() >= 2:
            print(f"[PASS] Regime labels exist and count: {t['regime_label'].nunique()}")
        else:
            print("[FAIL] Regime labels missing or insufficient classes")
    else:
        print("[FAIL] Missing technical indicators parquets")

    print("\n=== VERIFYING STEP 06 ===")
    baseline_files = glob.glob('outputs/metrics/*baseline*h1.json') + glob.glob('outputs/metrics/*lgbm*h1.json')
    if len(baseline_files) >= 3:
        print(f"[PASS] Baseline metrics files found: {len(baseline_files)}")
    else:
        print(f"[FAIL] Not enough baseline files: {len(baseline_files)}")
        
    split_path = "data/processed/split_h1.parquet"
    if os.path.exists(split_path):
        sp = pd.read_parquet(split_path)
        print(f"[PASS] Split file exists with counts:\n{sp['split'].value_counts(normalize=True).to_dict()}")
    else:
        print("[FAIL] Split file missing")

    combined = "outputs/metrics/combined_lgbm_h1.json"
    if os.path.exists(combined):
        with open(combined) as fl:
            c = json.load(fl)
            if "test_macro_f1" in c and "majority_macro_f1" in c:
                print(f"[PASS] Combined baseline Macro F1: {c['test_macro_f1']:.4f} vs Majority: {c['majority_macro_f1']:.4f}")
            else:
                print("[FAIL] Combined baseline missing metrics")
    else:
        print("[FAIL] Combined baseline missing")

if __name__ == "__main__":
    verify_all()
