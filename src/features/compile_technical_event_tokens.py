import os
import argparse
import pandas as pd
import json

def compile_tokens(row):
    tokens = []
    
    # RSI
    rsi = row.get("RSI_14", None)
    if pd.notna(rsi):
        if rsi >= 70:
            tokens.append(f"[RSI_OVERBOUGHT: RSI14={rsi:.1f}, strength={'high' if rsi >= 80 else 'medium'}]")
        elif rsi <= 30:
            tokens.append(f"[RSI_OVERSOLD: RSI14={rsi:.1f}, strength={'high' if rsi <= 20 else 'medium'}]")
            
    # MACD
    macd_hist = row.get("MACD_hist", None)
    if pd.notna(macd_hist):
        if macd_hist > 0:
            tokens.append(f"[MACD_BULLISH: hist={macd_hist:.2f}]")
        elif macd_hist < 0:
            tokens.append(f"[MACD_BEARISH: hist={macd_hist:.2f}]")
            
    # SMA
    p_sma20 = row.get("price_vs_SMA20", None)
    if pd.notna(p_sma20):
        if p_sma20 > 0:
            tokens.append(f"[PRICE_ABOVE_SMA20: distance={p_sma20*100:.1f}%]")
        else:
            tokens.append(f"[PRICE_BELOW_SMA20: distance={p_sma20*100:.1f}%]")
            
    # Bollinger
    bb_pos = row.get("Bollinger_position_20", None)
    if pd.notna(bb_pos):
        if bb_pos >= 0.9:
            tokens.append(f"[BOLLINGER_UPPER_PRESSURE: position={bb_pos:.2f}]")
        elif bb_pos <= 0.1:
            tokens.append(f"[BOLLINGER_LOWER_PRESSURE: position={bb_pos:.2f}]")
            
    # Volume
    vol_z = row.get("volume_zscore_20", None)
    if pd.notna(vol_z):
        if vol_z >= 1.5:
            tokens.append(f"[VOLUME_SPIKE: zscore={vol_z:.1f}]")
        elif vol_z <= -1.0:
            tokens.append(f"[VOLUME_DRY_UP: zscore={vol_z:.1f}]")
            
    # Relative Strength
    rs_5d = row.get("relative_strength_vs_market_5d", None)
    if pd.notna(rs_5d):
        if rs_5d > 0.02:
            tokens.append(f"[MARKET_OUTPERFORMANCE_5D: value={rs_5d*100:.1f}%]")
        elif rs_5d < -0.02:
            tokens.append(f"[MARKET_UNDERPERFORMANCE_5D: value={rs_5d*100:.1f}%]")
            
    # Volatility / Regime
    vol_rank = row.get("market_vol_20d_rank", None)
    if pd.notna(vol_rank) and vol_rank >= 0.8:
        tokens.append(f"[HIGH_VOLATILITY_REGIME: vol20_pctile={vol_rank:.2f}]")
        
    return json.dumps(tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/indicators/technical_features_h1.parquet")
    parser.add_argument("--output", type=str, default="data/indicators/technical_event_tokens_h1.parquet")
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    
    print("Compiling event tokens...")
    features["technical_event_tokens"] = features.apply(compile_tokens, axis=1)
    features["technical_summary_text"] = features["technical_event_tokens"].apply(lambda x: ", ".join(json.loads(x)))
    
    out_df = features[["sample_id", "ticker", "event_date", "technical_event_tokens", "technical_summary_text", "regime_label"]]
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    
    # Validation info
    missing_rate = float(out_df["technical_event_tokens"].apply(lambda x: len(json.loads(x)) == 0).mean())
    regimes = out_df["regime_label"].value_counts().to_dict()
    
    sample_tokens = []
    if len(out_df) > 0:
        non_empty = out_df[out_df["technical_event_tokens"] != "[]"]
        if len(non_empty) > 0:
            sample_tokens = json.loads(non_empty.iloc[0]["technical_event_tokens"])

    status = {
        "step": "05_TECHNICAL_INDICATORS_AND_EVENT_TOKENS",
        "status": "PASS",
        "feature_rows": len(features),
        "token_rows": len(out_df),
        "missing_rate": missing_rate,
        "regime_distribution": regimes,
        "example_tokens": sample_tokens
    }
    
    with open("outputs/status/05_TECHNICAL_INDICATORS_AND_EVENT_TOKENS.status.json", "w") as f:
        json.dump(status, f, indent=2)
        
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
