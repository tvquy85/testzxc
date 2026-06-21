from src.llm.parse_and_validate_rationale_v4 import parse_llm_json_strict_v4, validate_rationale_schema_evidence_v4


VALID = {
    "news_rationale": [
        {"evidence_id": "N1", "factor": "earnings guidance improved", "direction": "positive", "strength": "medium"}
    ],
    "technical_rationale": [
        {"signal_id": "T1", "signal": "MACD_BULLISH", "direction": "positive", "strength": "strong"}
    ],
    "conflict_resolution": "News and technical signals are aligned.",
    "forecast_distribution": {
        "Strong Down": 0.05,
        "Mild Down": 0.10,
        "Neutral": 0.35,
        "Mild Up": 0.35,
        "Strong Up": 0.15,
    },
    "action": "long",
    "risk_note": "Market weakness could offset company evidence.",
}


def test_v4_schema_accepts_valid_evidence_and_signal_ids():
    ok, errors = validate_rationale_schema_evidence_v4(VALID, {"evidence_ids": ["N1"], "signal_ids": ["T1"]})
    assert ok, errors


def test_v4_schema_rejects_unknown_evidence_id():
    data = dict(VALID)
    data["news_rationale"] = [dict(VALID["news_rationale"][0], evidence_id="N9")]
    ok, errors = validate_rationale_schema_evidence_v4(data, {"evidence_ids": ["N1"], "signal_ids": ["T1"]})
    assert not ok
    assert any("unknown evidence_id" in error for error in errors)


def test_v4_schema_rejects_unknown_signal_id():
    data = dict(VALID)
    data["technical_rationale"] = [dict(VALID["technical_rationale"][0], signal_id="T9")]
    ok, errors = validate_rationale_schema_evidence_v4(data, {"evidence_ids": ["N1"], "signal_ids": ["T1"]})
    assert not ok
    assert any("unknown signal_id" in error for error in errors)


def test_v4_schema_does_not_repair_distribution_sum():
    data = dict(VALID)
    data["forecast_distribution"] = dict(VALID["forecast_distribution"], Neutral=0.40)
    ok, errors = validate_rationale_schema_evidence_v4(data, {"evidence_ids": ["N1"], "signal_ids": ["T1"]})
    assert not ok
    assert any("sum" in error for error in errors)


def test_v4_parser_invalid_json_stays_invalid():
    assert parse_llm_json_strict_v4("not-json") is None
