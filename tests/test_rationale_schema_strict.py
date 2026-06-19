from src.llm.parse_and_validate_rationale import (
    normalize_distribution_if_valid,
    parse_llm_json_strict,
    validate_rationale_schema_strict,
)


VALID = {
    "news_rationale": ["news"],
    "technical_rationale": ["technical"],
    "conflict_resolution": "aligned",
    "forecast_distribution": {
        "strong_down": 0.1,
        "mild_down": 0.2,
        "neutral": 0.4,
        "mild_up": 0.2,
        "strong_up": 0.1,
    },
    "action": "hold",
    "risk_note": "risk",
}


def test_parse_invalid_json_stays_invalid():
    assert parse_llm_json_strict("not json") is None


def test_strict_validation_does_not_fill_missing_distribution():
    bad = dict(VALID)
    bad.pop("forecast_distribution")

    ok, errors = validate_rationale_schema_strict(bad)

    assert not ok
    assert any("forecast_distribution" in err for err in errors)
    assert "forecast_distribution" not in bad


def test_validation_does_not_overwrite_probabilities_from_action():
    data = dict(VALID)
    data["action"] = "long"
    data["forecast_distribution"] = {
        "strong_down": 0.8,
        "mild_down": 0.1,
        "neutral": 0.05,
        "mild_up": 0.03,
        "strong_up": 0.02,
    }

    ok, _ = validate_rationale_schema_strict(data)

    assert ok
    assert data["forecast_distribution"]["strong_down"] == 0.8


def test_normalize_distribution_only_after_valid_schema():
    data = dict(VALID)
    data["forecast_distribution"] = {
        "strong_down": 0.2,
        "mild_down": 0.2,
        "neutral": 0.2,
        "mild_up": 0.2,
        "strong_up": 0.21,
    }

    normalized = normalize_distribution_if_valid(data)

    assert abs(sum(normalized["forecast_distribution"].values()) - 1.0) < 1e-9


def test_fast_qwen3_object_schema_and_title_labels_validate():
    data = {
        "news_rationale": [{"factor": "earnings beat", "direction": "positive", "strength": "medium"}],
        "technical_rationale": [{"signal": "MACD bearish", "direction": "negative", "strength": "weak"}],
        "conflict_resolution": "News is constructive, but technical momentum is mixed.",
        "forecast_distribution": {
            "Strong Down": 0.05,
            "Mild Down": 0.15,
            "Neutral": 0.45,
            "Mild Up": 0.25,
            "Strong Up": 0.10,
        },
        "action": "hold",
        "risk_note": "Guidance may reverse sentiment.",
    }

    ok, errors = validate_rationale_schema_strict(data)

    assert ok, errors
    normalized = normalize_distribution_if_valid(data)
    assert set(normalized["forecast_distribution"]) == {"strong_down", "mild_down", "neutral", "mild_up", "strong_up"}
