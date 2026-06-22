import json
import sys

import pandas as pd

from src.eval import build_counterfactual_quality_filtered_v6 as quality


def _pack(headline: str, body: str, token: str = "RSI_BULLISH", direction: str = "bullish") -> str:
    return json.dumps(
        {
            "company_evidence": [{"headline": headline, "body_excerpt": body}],
            "context_evidence": [],
            "technical_signals": [{"token": token, "direction_prior": direction}],
        }
    )


def test_quality_filtered_counterfactual_tasks(tmp_path, monkeypatch):
    contexts = pd.DataFrame(
        [
            {
                "sample_id": "pos1",
                "ticker": "AAA",
                "event_date": "2024-01-02",
                "split": "test",
                "evidence_pack_json": _pack("profit beat and strong growth", "record profit and upgrade"),
                "technical_event_tokens_json": json.dumps([{"token": "RSI_BULLISH", "direction_prior": "bullish"}]),
            },
            {
                "sample_id": "neg1",
                "ticker": "BBB",
                "event_date": "2024-01-03",
                "split": "test",
                "evidence_pack_json": _pack("lawsuit warning and weak loss", "downgrade after missed profit", "RSI_BEARISH", "bearish"),
                "technical_event_tokens_json": json.dumps([{"token": "RSI_BEARISH", "direction_prior": "bearish"}]),
            },
            {
                "sample_id": "mixed1",
                "ticker": "CCC",
                "event_date": "2024-01-04",
                "split": "test",
                "evidence_pack_json": _pack("profit beat but lawsuit warning", "strong growth and weak loss"),
                "technical_event_tokens_json": json.dumps([{"token": "RSI_BULLISH", "direction_prior": "bullish"}]),
            },
        ]
    )
    contexts_path = tmp_path / "contexts.parquet"
    output_path = tmp_path / "tasks.jsonl"
    candidate_path = tmp_path / "candidates.csv"
    selected_path = tmp_path / "selected.csv"
    summary_path = tmp_path / "summary.csv"
    metrics_path = tmp_path / "metrics.json"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    contexts.to_parquet(contexts_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_counterfactual_quality_filtered_v6",
            "--contexts",
            str(contexts_path),
            "--output",
            str(output_path),
            "--candidate-output",
            str(candidate_path),
            "--selected-output",
            str(selected_path),
            "--summary-output",
            str(summary_path),
            "--metrics",
            str(metrics_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
            "--per-type-limit",
            "2",
            "--min-per-type",
            "1",
        ],
    )

    assert quality.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["pipeline_pass"] is True
    assert metrics["claim_allowed"] is False
    selected = pd.read_csv(selected_path)
    assert len(selected) >= 7
    assert selected["quality_pass"].all()
    tasks = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row["counterfactual_type"] == "remove_positive_evidence" for row in tasks)
    assert any("favorable company evidence removed" in row["counterfactual_headline"].lower() for row in tasks)


def test_semantic_neutralization_rewrites_counterfactual_text():
    task = {
        "counterfactual_type": "neutralize_negative_evidence",
        "counterfactual_headline": "lawsuit warning neutralized",
        "counterfactual_body": "loss warning neutralized",
    }
    repaired = quality.repair_semantic_neutralization(task)
    text = f"{repaired['counterfactual_headline']} {repaired['counterfactual_body']}".lower()
    assert repaired["counterfactual_type"] == "neutralize_negative_evidence"
    assert "unfavorable language rewritten" in text
    assert "lawsuit" not in text
    assert "loss" not in text
    assert "warning" not in text
