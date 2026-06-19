import numpy as np
import pandas as pd

from src.features.compute_technical_indicators import (
    build_features,
    compute_bollinger,
    compute_indicators_for_ticker,
    compute_rsi,
    compute_volume_zscore,
)


def price_frame(n=90):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 140, n) + np.sin(np.arange(n)), index=range(n))
    return pd.DataFrame(
        {
            "ticker": "AAA",
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adj_close": close,
            "volume": 1000,
        }
    )


def test_rsi_bounds_between_zero_and_one_hundred():
    rsi = compute_rsi(pd.Series([1, 2, 3, 2, 2, 5, 4, 4, 6, 7, 8, 7, 7, 9, 10, 11, 10]))
    valid = rsi.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_macd_columns_finite_after_warmup():
    out = compute_indicators_for_ticker(price_frame())
    macd = out[["MACD", "MACD_signal", "MACD_hist"]].iloc[35:]
    assert np.isfinite(macd.to_numpy()).all()


def test_bollinger_width_non_negative():
    bb = compute_bollinger(pd.Series(np.linspace(100, 120, 40)))
    assert (bb["Bollinger_width_20"].dropna() >= 0).all()


def test_volume_zscore_handles_zero_std():
    z = compute_volume_zscore(pd.Series([100.0] * 30))
    assert np.isfinite(z).all()
    assert float(z.iloc[-1]) == 0.0


def test_join_uses_window_end_not_future_event_date():
    prices = price_frame()
    samples = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "ticker": ["AAA"],
            "event_date": [pd.Timestamp("2020-03-15")],
            "window_end_date": [pd.Timestamp("2020-03-14")],
        }
    )
    features = build_features(samples, prices)
    assert pd.Timestamp(features.iloc[0]["window_end_date"]) < pd.Timestamp(features.iloc[0]["event_date"])

