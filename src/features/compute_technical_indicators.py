import os
import argparse
import pandas as pd
import numpy as np

def compute_indicators_for_ticker(df):
    df = df.sort_values("date").copy()
    
    # Returns
    df["ret_1d"] = df["adj_close"].pct_change(1)
    df["ret_5d"] = df["adj_close"].pct_change(5)
    df["ret_20d"] = df["adj_close"].pct_change(20)
    
    # Volatility
    df["volatility_10d"] = df["ret_1d"].rolling(10).std() * np.sqrt(252)
    df["volatility_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
    
    # RSI (14)
    delta = df["adj_close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["adj_close"].ewm(span=12, adjust=False).mean()
    ema26 = df["adj_close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    
    # SMA
    df["SMA_5"] = df["adj_close"].rolling(5).mean()
    df["SMA_20"] = df["adj_close"].rolling(20).mean()
    df["SMA_60"] = df["adj_close"].rolling(60).mean()
    
    df["price_vs_SMA20"] = (df["adj_close"] / df["SMA_20"]) - 1
    df["price_vs_SMA60"] = (df["adj_close"] / df["SMA_60"]) - 1
    
    # Bollinger Bands
    df["Bollinger_width_20"] = (df["SMA_20"] + 2 * df["adj_close"].rolling(20).std()) - (df["SMA_20"] - 2 * df["adj_close"].rolling(20).std())
    df["Bollinger_width_20"] = df["Bollinger_width_20"] / df["SMA_20"]
    upper = df["SMA_20"] + 2 * df["adj_close"].rolling(20).std()
    lower = df["SMA_20"] - 2 * df["adj_close"].rolling(20).std()
    df["Bollinger_position_20"] = (df["adj_close"] - lower) / (upper - lower)
        
    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR_14"] = true_range.rolling(14).mean()
    
    # Volume z-score
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["volume_zscore_20"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)
    
    # Gap pct last day
    prev_close = df["close"].shift(1)
    df["gap_pct_last_day"] = (df["open"] / prev_close) - 1
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default="data/labels/aligned_samples_h1.parquet")
    parser.add_argument("--prices", type=str, default="data/processed/prices_mvp.parquet")
    parser.add_argument("--output", type=str, default="data/indicators/technical_features_h1.parquet")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    samples = pd.read_parquet(args.samples)
    
    # 1. Compute market proxy for relative strength and regime
    print("Computing market proxy...")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    samples["window_end_date"] = pd.to_datetime(samples["window_end_date"]).dt.tz_localize(None)
    
    market = prices.groupby("date")["ret_1d" if "ret_1d" in prices.columns else "close"].mean().reset_index() # approximation
    if "ret_1d" not in prices.columns:
        # compute simple returns first
        prices["ret_1d"] = prices.groupby("ticker")["adj_close"].pct_change(1)
        market = prices.groupby("date")["ret_1d"].mean().reset_index()
    
    market.rename(columns={"ret_1d": "market_ret_1d"}, inplace=True)
    market["market_ret_5d"] = (1 + market["market_ret_1d"]).rolling(5).apply(np.prod, raw=True) - 1
    market["market_ret_20d"] = (1 + market["market_ret_1d"]).rolling(20).apply(np.prod, raw=True) - 1
    market["market_vol_20d"] = market["market_ret_1d"].rolling(20).std() * np.sqrt(252)
    
    # Rolling percentile for regime
    # A simple way is rank / len, over a rolling window. We'll use expanding or large rolling window
    market["market_vol_20d_rank"] = market["market_vol_20d"].rolling(252, min_periods=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # 2. Compute stock indicators
    print("Computing stock indicators...")
    prices = prices.groupby("ticker", group_keys=False).apply(compute_indicators_for_ticker)
    
    # Merge market to compute relative strength
    prices = pd.merge(prices, market, on="date", how="left")
    prices["relative_strength_vs_market_5d"] = prices["ret_5d"] - prices["market_ret_5d"]
    prices["relative_strength_vs_market_20d"] = prices["ret_20d"] - prices["market_ret_20d"]
    
    # Create Regime label
    def get_regime(pctile):
        if pd.isna(pctile): return "normal_vol"
        if pctile <= 0.33: return "low_vol"
        elif pctile >= 0.66: return "high_vol"
        else: return "normal_vol"
        
    prices["regime_label"] = prices["market_vol_20d_rank"].apply(get_regime)
    
    # 3. Join with samples at window_end_date to prevent future leakage
    print("Joining features with samples...")
    # The prices df has date as the actual trading date. We join on ticker and date = window_end_date.
    features = pd.merge(samples[["sample_id", "ticker", "event_date", "window_end_date"]], 
                        prices, 
                        left_on=["ticker", "window_end_date"], 
                        right_on=["ticker", "date"], 
                        how="inner")
                        
    # Clean up duplicated or unnecessary columns
    drop_cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "raw_file"]
    features = features.drop(columns=[c for c in drop_cols if c in features.columns])
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"Features saved to {args.output}")

if __name__ == "__main__":
    main()
