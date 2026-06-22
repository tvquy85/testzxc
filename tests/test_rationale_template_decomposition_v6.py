import json
import sys

import pandas as pd

from src.llm import audit_rationale_template_decomposition_v6 as decomp


def _row(sample_id: str, candidate_id: int, news_factor: str, tech_signal: str) -> dict:
    parsed = {
        "news_rationale": [
            {
                "evidence_id": "N1",
                "factor": news_factor,
                "direction": "positive",
                "strength": "medium",
            }
        ],
        "technical_rationale": [
            {
                "signal_id": "T1",
                "signal": tech_signal,
                "direction": "bullish",
                "strength": "strong",
            }
        ],
        "conflict_resolution": f"{news_factor} matters",
        "risk_note": "risk differs",
    }
    return {
        "sample_id": sample_id,
        "candidate_id": candidate_id,
        "parsed_json": json.dumps(parsed),
        "context_meta_json": json.dumps({"company_evidence_ids": ["N1"]}),
    }


def test_news_plus_meta_excludes_technical_signal_names():
    parsed = {
        "news_rationale": [{"evidence_id": "N1", "factor": "earnings beat"}],
        "technical_rationale": [{"signal_id": "T1", "signal": "macd bullish momentum"}],
        "conflict_resolution": "news leads",
        "risk_note": "watch volatility",
    }
    text = decomp.news_plus_meta_text(parsed)
    assert "earnings beat" in text
    assert "macd bullish momentum" not in text


def test_rationale_decomposition_end_to_end(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            _row("s1", 0, "earnings beat", "macd bullish momentum"),
            _row("s1", 1, "dividend increase", "macd bullish momentum"),
            _row("s1", 2, "analyst upgrade", "macd bullish momentum"),
            _row("s2", 0, "lawsuit risk", "macd bullish momentum"),
            _row("s2", 1, "recall warning", "macd bullish momentum"),
            _row("s2", 2, "margin pressure", "macd bullish momentum"),
        ]
    )
    input_path = tmp_path / "rationales.parquet"
    metrics_path = tmp_path / "metrics.json"
    table_path = tmp_path / "table.csv"
    samples_path = tmp_path / "samples.jsonl"
    status_path = tmp_path / "status.json"
    manifest_path = tmp_path / "manifest.json"
    df.to_parquet(input_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_rationale_template_decomposition_v6",
            "--input",
            str(input_path),
            "--metrics",
            str(metrics_path),
            "--table",
            str(table_path),
            "--samples",
            str(samples_path),
            "--status",
            str(status_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert decomp.main() == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["news_repeated_phrase_rate"] == 0.0
    assert metrics["technical_repeated_phrase_rate"] == 1.0
    assert metrics["technical_repetition_explains_overall_repetition"] is True
