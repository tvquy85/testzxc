import json
import sys

import pandas as pd

from src.eval import trading_policy_variant_probe_v6 as probe


def test_trading_policy_variant_probe_end_to_end(tmp_path, monkeypatch):
    contexts = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "ticker": "AAA",
                "event_date": "2024-01-02",
                "split": "test",
                "target_return": 0.02,
                "technical_event_tokens_json": json.dumps([{"direction_prior": "bullish", "strength": "strong"}]),
            },
            {
                "sample_id": "s2",
                "ticker": "BBB",
                "event_date": "2024-01-03",
                "split": "test",
                "target_return": -0.01,
                "technical_event_tokens_json": json.dumps([{"direction_prior": "bearish", "strength": "strong"}]),
            },
        ]
    )
    dpo = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "schema_ok": True,
                "action": "long",
                "pred_label": "mild_up",
                "p_mild_up": 0.6,
                "p_strong_up": 0.1,
                "p_mild_down": 0.1,
                "p_strong_down": 0.0,
            },
            {
                "sample_id": "s2",
                "schema_ok": True,
                "action": "short",
                "pred_label": "mild_down",
                "p_mild_up": 0.1,
                "p_strong_up": 0.0,
                "p_mild_down": 0.6,
                "p_strong_down": 0.1,
            },
        ]
    )
    hybrid = pd.DataFrame({"sample_id": ["s1", "s2"], "hybrid_pred": ["mild_up", "mild_down"]})
    contexts_path = tmp_path / "contexts.parquet"
    dpo_path = tmp_path / "dpo.parquet"
    hybrid_path = tmp_path / "hybrid.parquet"
    summary_path = tmp_path / "summary.csv"
    daily_path = tmp_path / "daily.csv"
    metrics_path = tmp_path / "metrics.json"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    contexts.to_parquet(contexts_path)
    dpo.to_parquet(dpo_path)
    hybrid.to_parquet(hybrid_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trading_policy_variant_probe_v6",
            "--contexts",
            str(contexts_path),
            "--dpo-predictions",
            str(dpo_path),
            "--hybrid-predictions",
            str(hybrid_path),
            "--stacked-predictions",
            str(tmp_path / "missing_stacked.parquet"),
            "--supervised-predictions",
            str(tmp_path / "missing_supervised.parquet"),
            "--summary-output",
            str(summary_path),
            "--daily-output",
            str(daily_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
            "--n-bootstrap",
            "5",
            "--block-size",
            "1",
        ],
    )

    assert probe.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pipeline_pass"] is True
    assert metrics["claim_allowed"] is False
    assert metrics["strategy_count"] >= 3
    assert summary_path.exists()
    assert daily_path.exists()
