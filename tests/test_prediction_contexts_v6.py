from types import SimpleNamespace

import pandas as pd

from src.eval.build_prediction_contexts_v6 import add_v6_columns, build_prediction_contexts, source_is_sufficient


def test_add_v6_columns_handles_dict_evidence_pack():
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "split": "test",
                "mean_evidence_quality_score": 0.8,
                "has_company_event_news": True,
                "evidence_pack_json": '{"company_evidence":[{"article_type":"earnings_or_guidance"}],"context_evidence":[]}',
            }
        ]
    )

    out = add_v6_columns(df)

    assert out.loc[0, "num_hard_event_evidence"] == 1
    assert out.loc[0, "v6_track"] == "hard_event_news"


def test_source_is_sufficient_requires_rows_and_days():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "split": ["test", "test"],
            "event_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        }
    )

    assert not source_is_sufficient(df, "test", min_rows=3, min_trading_days=2)
    assert not source_is_sufficient(df, "test", min_rows=2, min_trading_days=3)
    assert source_is_sufficient(df, "test", min_rows=2, min_trading_days=2)


def test_build_prediction_contexts_uses_fallback_when_source_too_small(tmp_path):
    source = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "ticker": ["AAPL"],
            "event_date": pd.to_datetime(["2023-01-01"]),
            "split": ["test"],
            "target_label_5": ["neutral"],
            "target_direction": ["flat"],
            "target_return": [0.0],
            "abnormal_return_h1": [0.0],
            "evidence_pack_json": ['{"company_evidence":[],"context_evidence":[],"technical_signals":[]}'],
            "technical_event_tokens_json": ["[]"],
            "clean_context_text": ["Ticker: AAPL"],
            "mean_evidence_quality_score": [0.0],
            "has_company_event_news": [False],
        }
    )
    fallback = pd.DataFrame(
        {
            "sample_id": ["f1", "f2", "f3"],
            "source_sample_ids": ["[]", "[]", "[]"],
            "ticker": ["AAPL", "MSFT", "NVDA"],
            "event_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "split": ["test", "test", "test"],
            "horizon": [1, 1, 1],
            "aggregated_headlines": ["earnings beat", "guidance raise", "product launch"],
            "aggregated_body": ["", "", ""],
            "context_news_count": [1, 1, 1],
            "mean_news_quality_score": [0.8, 0.8, 0.8],
            "label_5": ["neutral", "mild_up", "mild_down"],
            "abnormal_return_h1": [0.0, 0.01, -0.01],
            "technical_event_tokens_json": ["[]", "[]", "[]"],
            "target_return": [0.0, 0.01, -0.01],
            "target_label_5": ["neutral", "mild_up", "mild_down"],
            "target_direction": ["flat", "up", "down"],
        }
    )
    source_path = tmp_path / "source.parquet"
    fallback_path = tmp_path / "fallback.parquet"
    source.to_parquet(source_path, index=False)
    fallback.to_parquet(fallback_path, index=False)
    args = SimpleNamespace(
        source=str(source_path),
        fallback_contexts=str(fallback_path),
        split="test",
        min_rows=3,
        min_trading_days=3,
        seed=1,
    )

    selected, metrics, failures = build_prediction_contexts(args)

    assert failures == []
    assert metrics["used_fallback_contexts"] is True
    assert len(selected) == 3
    assert set(selected["split"]) == {"test"}
