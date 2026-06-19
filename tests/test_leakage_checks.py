import pandas as pd

from src.data.leakage_checks import check_prompt_leakage, check_split_rule, check_timestamp_rule


def test_timestamp_rule_catches_future_news_and_features():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "timestamp_utc": ["2023-01-03T00:00:00Z", "2023-01-05T00:00:00Z"],
            "event_date": ["2023-01-04", "2023-01-04"],
            "window_end_date": ["2023-01-03", "2023-01-05"],
        }
    )

    leaks = check_timestamp_rule(df)

    assert any(item["type"] == "news_after_decision" and item["sample_id"] == "b" for item in leaks)
    assert any(item["type"] == "feature_window_after_decision" and item["sample_id"] == "b" for item in leaks)


def test_split_rule_catches_duplicate_and_illegal_split():
    splits = pd.DataFrame({"sample_id": ["a", "a", "b"], "split": ["train", "val", "future"]})

    leaks = check_split_rule(splits)

    assert any(item["type"] == "duplicate_split_membership" for item in leaks)
    assert any(item["type"] == "illegal_split_value" and item["sample_id"] == "b" for item in leaks)


def test_prompt_leakage_catches_realized_return(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "bad.txt").write_text("Use the realized return as the target.", encoding="utf-8")

    leaks = check_prompt_leakage(str(prompt_dir))

    assert len(leaks) == 1
    assert leaks[0]["type"] == "prompt_leakage_pattern"
