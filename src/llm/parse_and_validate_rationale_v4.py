from __future__ import annotations

import json
from typing import Any

FORECAST_CANONICAL = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
TITLE_TO_CANONICAL = {
    "Strong Down": "strong_down",
    "Mild Down": "mild_down",
    "Neutral": "neutral",
    "Mild Up": "mild_up",
    "Strong Up": "strong_up",
    "strong_down": "strong_down",
    "mild_down": "mild_down",
    "neutral": "neutral",
    "mild_up": "mild_up",
    "strong_up": "strong_up",
}
VALID_DIRECTIONS = {"positive", "negative", "neutral"}
VALID_STRENGTHS = {"weak", "medium", "strong"}
VALID_ACTIONS = {"long", "short", "hold"}


def parse_llm_json_strict_v4(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def forecast_distribution_errors(dist: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(dist, dict):
        return ["forecast_distribution must be an object"]
    mapped: dict[str, float] = {}
    unknown = sorted(set(dist) - set(TITLE_TO_CANONICAL))
    if unknown:
        errors.append(f"forecast_distribution has unknown labels: {unknown}")
    for key, value in dist.items():
        canonical = TITLE_TO_CANONICAL.get(str(key))
        if canonical is None:
            continue
        if canonical in mapped:
            errors.append(f"forecast_distribution duplicate canonical label: {canonical}")
            continue
        try:
            mapped[canonical] = float(value)
        except Exception:
            errors.append(f"forecast_distribution value for {key} is not numeric")
    missing = [key for key in FORECAST_CANONICAL if key not in mapped]
    if missing:
        errors.append(f"forecast_distribution missing labels: {missing}")
    if len(mapped) == len(FORECAST_CANONICAL):
        total = sum(mapped.values())
        if abs(total - 1.0) > 1e-3:
            errors.append(f"forecast_distribution sum {total:.6f} != 1.0")
        if any(value < 0 or value > 1 for value in mapped.values()):
            errors.append("forecast_distribution values must be in [0,1]")
    return errors


def validate_rationale_schema_evidence_v4(parsed: dict[str, Any], context_meta: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return False, ["parsed rationale must be an object"]
    context_meta = context_meta or {}
    valid_evidence_ids = set(context_meta.get("evidence_ids", []))
    valid_signal_ids = set(context_meta.get("signal_ids", []))

    news = parsed.get("news_rationale")
    if not isinstance(news, list):
        errors.append("news_rationale must be a list")
    else:
        if len(news) > 2:
            errors.append("news_rationale has more than 2 items")
        for idx, item in enumerate(news):
            if not isinstance(item, dict):
                errors.append(f"news_rationale[{idx}] must be an object")
                continue
            for key in ["evidence_id", "factor", "direction", "strength"]:
                if key not in item or item.get(key) in {"", None}:
                    errors.append(f"news_rationale[{idx}] missing {key}")
            evidence_id = str(item.get("evidence_id", ""))
            if valid_evidence_ids and evidence_id not in valid_evidence_ids:
                errors.append(f"news_rationale[{idx}] unknown evidence_id: {evidence_id}")
            if item.get("direction") not in VALID_DIRECTIONS:
                errors.append(f"news_rationale[{idx}] invalid direction")
            if item.get("strength") not in VALID_STRENGTHS:
                errors.append(f"news_rationale[{idx}] invalid strength")

    tech = parsed.get("technical_rationale")
    if not isinstance(tech, list):
        errors.append("technical_rationale must be a list")
    else:
        if len(tech) > 2:
            errors.append("technical_rationale has more than 2 items")
        for idx, item in enumerate(tech):
            if not isinstance(item, dict):
                errors.append(f"technical_rationale[{idx}] must be an object")
                continue
            for key in ["signal_id", "signal", "direction", "strength"]:
                if key not in item or item.get(key) in {"", None}:
                    errors.append(f"technical_rationale[{idx}] missing {key}")
            signal_id = str(item.get("signal_id", ""))
            if valid_signal_ids and signal_id not in valid_signal_ids:
                errors.append(f"technical_rationale[{idx}] unknown signal_id: {signal_id}")
            if item.get("direction") not in VALID_DIRECTIONS:
                errors.append(f"technical_rationale[{idx}] invalid direction")
            if item.get("strength") not in VALID_STRENGTHS:
                errors.append(f"technical_rationale[{idx}] invalid strength")

    if not isinstance(parsed.get("conflict_resolution"), str) or not parsed.get("conflict_resolution", "").strip():
        errors.append("conflict_resolution must be a non-empty string")
    errors.extend(forecast_distribution_errors(parsed.get("forecast_distribution")))
    if parsed.get("action") not in VALID_ACTIONS:
        errors.append("action must be one of long|short|hold")
    if not isinstance(parsed.get("risk_note"), str) or not parsed.get("risk_note", "").strip():
        errors.append("risk_note must be a non-empty string")
    return not errors, errors
