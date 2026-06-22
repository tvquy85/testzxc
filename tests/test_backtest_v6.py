import json
import sys

import pandas as pd

from src.eval import backtest_daily_portfolio_v3 as backtest
from src.eval.backtest_daily_portfolio_v3 import compute_sharpe


def test_sharpe_zero_returns():
    assert compute_sharpe([0, 0, 0]) == 0


def test_backtest_v6_aliases_and_technical_baseline(tmp_path, monkeypatch):
    predictions = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "split": "test",
                "ticker": "AAA",
                "event_date": "2024-01-02",
                "schema_ok": True,
                "action": "long",
                "p_strong_down": 0.05,
                "p_mild_down": 0.10,
                "p_neutral": 0.20,
                "p_mild_up": 0.30,
                "p_strong_up": 0.35,
            },
            {
                "sample_id": "s2",
                "split": "test",
                "ticker": "BBB",
                "event_date": "2024-01-03",
                "schema_ok": True,
                "action": "short",
                "p_strong_down": 0.40,
                "p_mild_down": 0.20,
                "p_neutral": 0.20,
                "p_mild_up": 0.10,
                "p_strong_up": 0.10,
            },
        ]
    )
    contexts = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "ticker": "AAA",
                "event_date": "2024-01-02",
                "split": "test",
                "target_return": 0.02,
                "v6_track": "hard_event_news",
                "no_news_context_flag": True,
                "technical_event_tokens_json": json.dumps(
                    [{"direction_prior": "bullish_momentum", "strength": "strong"}]
                ),
            },
            {
                "sample_id": "s2",
                "ticker": "BBB",
                "event_date": "2024-01-03",
                "split": "test",
                "target_return": 0.01,
                "v6_track": "company_general_news",
                "no_news_context_flag": False,
                "technical_event_tokens_json": json.dumps(
                    [{"direction_prior": "bearish_momentum", "strength": "strong"}]
                ),
            },
        ]
    )

    predictions_path = tmp_path / "predictions.parquet"
    contexts_path = tmp_path / "contexts.parquet"
    metrics_path = tmp_path / "metrics.json"
    daily_path = tmp_path / "daily.csv"
    track_path = tmp_path / "track.csv"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    predictions.to_parquet(predictions_path)
    contexts.to_parquet(contexts_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_daily_portfolio_v3",
            "--predictions",
            str(predictions_path),
            "--contexts",
            str(contexts_path),
            "--output-daily",
            str(daily_path),
            "--output-track",
            str(track_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
            "--cost-bps",
            "0",
            "--slippage-bps",
            "0",
            "--short-borrow-bps",
            "0",
            "--min-trading-days",
            "1",
        ],
    )

    assert backtest.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["step"] == "15_BACKTEST_TRACK_BASELINE_V6"
    assert status["status"] == "PASS"
    assert metrics["technical_rule_num_trading_days"] == 2
    assert metrics["technical_only_baseline_name"] == "Technical_Rule"
    assert metrics["no_news_baseline_context_rows"] == 1
    assert daily_path.exists()
    assert track_path.exists()
