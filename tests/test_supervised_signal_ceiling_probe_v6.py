import json
import sys

import pandas as pd

from src.eval import supervised_signal_ceiling_probe_v6 as probe


def _context(sample_id: str, split: str, date: str, label: str, token_direction: str) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "ticker": sample_id[:2].upper(),
        "event_date": date,
        "target_label_5": label,
        "technical_event_tokens_json": json.dumps([{"direction_prior": token_direction, "strength": "strong"}]),
        "evidence_pack_json": json.dumps(
            {
                "company_evidence": [{"headline": f"{token_direction} company event", "body_excerpt": "event body"}],
                "context_evidence": [],
                "technical_signals": [{"direction_prior": token_direction, "strength": "strong"}],
            }
        ),
        "clean_context_text": f"{token_direction} context",
        "num_company_event_evidence": 1,
        "num_context_only_evidence": 0,
        "mean_evidence_quality_score": 0.8,
        "num_hard_event_evidence": 1,
        "v6_training_weight": 1.0,
        "has_company_event_news": True,
        "has_hard_event_news": True,
        "no_news_context_flag": False,
        "v6_track": "hard_event_news",
    }


def test_supervised_signal_ceiling_probe_end_to_end(tmp_path, monkeypatch):
    train_contexts = pd.DataFrame(
        [
            _context("tr1", "train", "2021-01-01", "mild_up", "bullish"),
            _context("tr2", "train", "2021-01-02", "mild_down", "bearish"),
            _context("tr3", "train", "2021-01-03", "neutral", "neutral"),
            _context("tr4", "train", "2021-01-04", "mild_up", "bullish"),
            _context("tr5", "train", "2021-01-05", "mild_down", "bearish"),
            _context("tr6", "train", "2021-01-06", "neutral", "neutral"),
        ]
    )
    val_contexts = pd.DataFrame(
        [
            _context("va1", "val", "2022-01-01", "mild_up", "bullish"),
            _context("va2", "val", "2022-01-02", "mild_down", "bearish"),
            _context("va3", "val", "2022-01-03", "neutral", "neutral"),
        ]
    )
    test_contexts = pd.DataFrame(
        [
            _context("te1", "test", "2023-01-01", "mild_up", "bullish"),
            _context("te2", "test", "2023-01-02", "mild_down", "bearish"),
            _context("te3", "test", "2023-01-03", "neutral", "neutral"),
        ]
    )
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    output_path = tmp_path / "result.csv"
    grid_path = tmp_path / "grid.csv"
    pred_path = tmp_path / "pred.parquet"
    metrics_path = tmp_path / "metrics.json"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    train_contexts.to_parquet(train_path)
    val_contexts.to_parquet(val_path)
    test_contexts.to_parquet(test_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supervised_signal_ceiling_probe_v6",
            "--train-contexts",
            str(train_path),
            "--val-contexts",
            str(val_path),
            "--test-contexts",
            str(test_path),
            "--min-train-rows",
            "3",
            "--min-df",
            "1",
            "--max-text-features",
            "20",
            "--max-token-features",
            "20",
            "--c-grid",
            "0.1,1",
            "--n-bootstrap",
            "5",
            "--output",
            str(output_path),
            "--grid-output",
            str(grid_path),
            "--predictions-output",
            str(pred_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert probe.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pipeline_pass"] is True
    assert metrics["claim_allowed"] is False
    assert metrics["train_rows"] == 6
    assert output_path.exists()
    assert grid_path.exists()
    assert pred_path.exists()
