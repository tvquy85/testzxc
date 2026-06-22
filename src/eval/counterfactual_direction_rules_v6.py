from __future__ import annotations

EXPECTED_DIRECTIONS = {
    "remove_positive_evidence": "up_decreases",
    "neutralize_positive_evidence": "up_decreases",
    "remove_negative_evidence": "down_decreases",
    "neutralize_negative_evidence": "down_decreases",
    "neutralize_bullish_technical": "up_decreases",
    "neutralize_bearish_technical": "down_decreases",
}


def expected_direction(counterfactual_type: str, evidence_polarity_score: float | None = None) -> str:
    cf_type = str(counterfactual_type)
    if cf_type == "remove_all_company_evidence":
        if evidence_polarity_score is None:
            return "evidence_signal_decreases"
        if evidence_polarity_score > 0:
            return "up_decreases"
        if evidence_polarity_score < 0:
            return "down_decreases"
        return "evidence_signal_decreases"
    return EXPECTED_DIRECTIONS.get(cf_type, "")


def normalized_expected_direction(expected: str) -> str:
    value = str(expected or "").strip().lower()
    return {
        "up_decreases": "up_decrease",
        "up_decrease": "up_decrease",
        "down_decreases": "down_decrease",
        "down_decrease": "down_decrease",
    }.get(value, value)
