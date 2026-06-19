from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "06_TECHNICAL_FEATURES_V2"


FEATURE_SPECS = [
    {"name": "ret_1d", "lookback": 1, "kind": "price_only", "leakage_note": "pct_change uses current and prior close only"},
    {"name": "ret_5d", "lookback": 5, "kind": "price_only", "leakage_note": "pct_change uses current and prior close only"},
    {"name": "ret_20d", "lookback": 20, "kind": "price_only", "leakage_note": "pct_change uses current and prior close only"},
    {"name": "volatility_10d", "lookback": 10, "kind": "price_only", "leakage_note": "rolling window includes rows <= feature date"},
    {"name": "volatility_20d", "lookback": 20, "kind": "price_only", "leakage_note": "rolling window includes rows <= feature date"},
    {"name": "RSI_14", "lookback": 14, "kind": "price_only", "leakage_note": "rolling gains/losses use historical deltas"},
    {"name": "MACD", "lookback": 26, "kind": "price_only", "leakage_note": "EMA uses current and past prices only"},
    {"name": "MACD_signal", "lookback": 35, "kind": "price_only", "leakage_note": "EMA over MACD uses current and past values only"},
    {"name": "MACD_hist", "lookback": 35, "kind": "price_only", "leakage_note": "derived from MACD and signal at feature date"},
    {"name": "SMA_5", "lookback": 5, "kind": "price_only", "leakage_note": "rolling window uses historical prices"},
    {"name": "SMA_20", "lookback": 20, "kind": "price_only", "leakage_note": "rolling window uses historical prices"},
    {"name": "SMA_60", "lookback": 60, "kind": "price_only", "leakage_note": "rolling window uses historical prices"},
    {"name": "price_vs_SMA20", "lookback": 20, "kind": "price_only", "leakage_note": "current close vs trailing SMA"},
    {"name": "price_vs_SMA60", "lookback": 60, "kind": "price_only", "leakage_note": "current close vs trailing SMA"},
    {"name": "Bollinger_width_20", "lookback": 20, "kind": "price_only", "leakage_note": "trailing mean/std only"},
    {"name": "Bollinger_position_20", "lookback": 20, "kind": "price_only", "leakage_note": "trailing bands only"},
    {"name": "ATR_14", "lookback": 14, "kind": "price_only", "leakage_note": "true range uses current high/low and previous close"},
    {"name": "volume_zscore_20", "lookback": 20, "kind": "price_only", "leakage_note": "trailing volume mean/std only"},
    {"name": "gap_pct_last_day", "lookback": 1, "kind": "price_only", "leakage_note": "open vs previous close at feature date"},
    {"name": "relative_strength_vs_market_5d", "lookback": 5, "kind": "market_relative", "leakage_note": "stock trailing return minus market trailing return"},
    {"name": "relative_strength_vs_market_20d", "lookback": 20, "kind": "market_relative", "leakage_note": "stock trailing return minus market trailing return"},
    {"name": "regime_label", "lookback": 252, "kind": "market_relative", "leakage_note": "market volatility percentile from trailing window"},
]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
    rsi = rsi.mask((gain > 0) & (loss == 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    return rsi.clip(0, 100)


def compute_macd(close: pd.Series) -> pd.DataFrame:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "MACD_signal": signal, "MACD_hist": macd - signal})


def compute_bollinger(close: pd.Series, window: int = 20) -> pd.DataFrame:
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mean + 2 * std
    lower = mean - 2 * std
    width = ((upper - lower) / mean.replace(0, np.nan)).clip(lower=0)
    position = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"Bollinger_width_20": width, "Bollinger_position_20": position})


def compute_volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std()
    z = (volume - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_indicators_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    close = out["adj_close"]
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)
    out["volatility_10d"] = out["ret_1d"].rolling(10, min_periods=10).std() * np.sqrt(252)
    out["volatility_20d"] = out["ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    out["RSI_14"] = compute_rsi(close)
    out = pd.concat([out, compute_macd(close)], axis=1)
    out["SMA_5"] = close.rolling(5, min_periods=5).mean()
    out["SMA_20"] = close.rolling(20, min_periods=20).mean()
    out["SMA_60"] = close.rolling(60, min_periods=60).mean()
    out["price_vs_SMA20"] = (close / out["SMA_20"]) - 1
    out["price_vs_SMA60"] = (close / out["SMA_60"]) - 1
    out = pd.concat([out, compute_bollinger(close)], axis=1)
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close = (out["low"] - out["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR_14"] = true_range.rolling(14, min_periods=14).mean()
    out["volume_zscore_20"] = compute_volume_zscore(out["volume"])
    out["gap_pct_last_day"] = (out["open"] / out["close"].shift(1)) - 1
    return out


def compute_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    market = prices.groupby("date")["ret_1d"].mean().reset_index(name="market_ret_1d")
    market["market_ret_5d"] = (1 + market["market_ret_1d"]).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    market["market_ret_20d"] = (1 + market["market_ret_1d"]).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1
    market["market_vol_20d"] = market["market_ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    market["market_vol_20d_rank"] = market["market_vol_20d"].rolling(252, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )
    return market


def regime_from_percentile(pctile) -> str:
    if pd.isna(pctile):
        return "normal_vol"
    if pctile <= 0.33:
        return "low_vol"
    if pctile >= 0.66:
        return "high_vol"
    return "normal_vol"


def build_features(samples: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    samples = samples.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    samples["window_end_date"] = pd.to_datetime(samples["window_end_date"]).dt.tz_localize(None)
    prices = prices.sort_values(["ticker", "date"])
    prices["ret_1d"] = prices.groupby("ticker")["adj_close"].pct_change(1)
    indicators = prices.groupby("ticker", group_keys=False).apply(compute_indicators_for_ticker)
    market = compute_market_features(indicators)
    indicators = indicators.merge(market, on="date", how="left")
    indicators["relative_strength_vs_market_5d"] = indicators["ret_5d"] - indicators["market_ret_5d"]
    indicators["relative_strength_vs_market_20d"] = indicators["ret_20d"] - indicators["market_ret_20d"]
    indicators["regime_label"] = indicators["market_vol_20d_rank"].apply(regime_from_percentile)
    features = samples[["sample_id", "ticker", "event_date", "window_end_date"]].merge(
        indicators,
        left_on=["ticker", "window_end_date"],
        right_on=["ticker", "date"],
        how="inner",
    )
    drop_cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "raw_file"]
    return features.drop(columns=[c for c in drop_cols if c in features.columns])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--prices", default="data/processed/prices_mvp.parquet")
    parser.add_argument("--output", default="data/indicators/technical_features_h1_v2.parquet")
    parser.add_argument("--feature-manifest", default="outputs/manifests/technical_feature_manifest.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--task-manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    samples = pd.read_parquet(args.samples)
    prices = pd.read_parquet(args.prices)
    features = build_features(samples, prices)
    warmup_missing = features.isna().mean().sort_values(ascending=False).to_dict()
    if len(features) == 0:
        failures.append("feature output is empty")
    if len(features) > len(samples):
        failures.append("feature row count exceeds sample count")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    write_json(
        args.feature_manifest,
        {
            "features": FEATURE_SPECS,
            "sample_rows": int(len(samples)),
            "feature_rows": int(len(features)),
            "warmup_missingness": {k: float(v) for k, v in warmup_missing.items()},
            "row_count_note": "Rows may be fewer than samples if warmup/history is unavailable.",
        },
    )
    write_manifest(args.task_manifest, [args.output, args.feature_manifest], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.samples, args.prices],
        outputs_created=[args.output, args.feature_manifest, args.task_manifest, args.status],
        metrics={"sample_rows": int(len(samples)), "feature_rows": int(len(features)), "max_missing_rate": float(max(warmup_missing.values()) if warmup_missing else 0.0)},
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps({"feature_rows": len(features), "sample_rows": len(samples)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

