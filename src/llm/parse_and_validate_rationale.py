from __future__ import annotations

import copy
import json
import re
from typing import Any


FORECAST_KEYS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
FORECAST_TITLE_KEYS = ["Strong Down", "Mild Down", "Neutral", "Mild Up", "Strong Up"]
FORECAST_KEY_ALIASES = dict(zip(FORECAST_TITLE_KEYS, FORECAST_KEYS))


def canonical_forecast_key(key: str) -> str:
    if key in FORECAST_KEYS:
        return key
    if key in FORECAST_KEY_ALIASES:
        return FORECAST_KEY_ALIASES[key]
    normalized = key.strip().lower().replace(" ", "_")
    return normalized


def forecast_value(dist: dict[str, Any], key: str) -> Any:
    if key in dist:
        return dist[key]
    for raw_key, canonical in FORECAST_KEY_ALIASES.items():
        if canonical == key and raw_key in dist:
            return dist[raw_key]
    return None


def _valid_rationale_item(item: Any) -> bool:
    if isinstance(item, str):
        return bool(item.strip())
    if isinstance(item, dict):
        direction = item.get("direction")
        strength = item.get("strength")
        text = item.get("factor", item.get("signal", item.get("text")))
        return (
            isinstance(text, str)
            and bool(text.strip())
            and direction in {"positive", "negative", "neutral"}
            and strength in {"weak", "medium", "strong"}
        )
    return False


def _extract_json_candidate(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text[start:])
        return text[start : start + end]
    except json.JSONDecodeError:
        end = text.rfind("}")
        if end > start:
            return text[start : end + 1]
    return None


def parse_llm_json_strict(text: str) -> dict[str, Any] | None:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate_rationale_schema_strict(data: dict[str, Any] | None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["data is not a JSON object"]
    for key in ["news_rationale", "technical_rationale"]:
        value = data.get(key)
        if not isinstance(value, list) or not value or not all(_valid_rationale_item(x) for x in value):
            errors.append(f"{key} must be a non-empty list of valid strings or rationale objects")
    for key in ["conflict_resolution", "risk_note"]:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a non-empty string")
    if data.get("action") not in {"long", "short", "hold"}:
        errors.append("action must be one of long, short, hold")
    dist = data.get("forecast_distribution")
    if not isinstance(dist, dict):
        errors.append("forecast_distribution must be an object")
    else:
        canonical_keys = [canonical_forecast_key(str(k)) for k in dist]
        missing = [k for k in FORECAST_KEYS if k not in canonical_keys]
        extra = sorted(str(k) for k, canonical in zip(dist.keys(), canonical_keys) if canonical not in FORECAST_KEYS)
        if missing:
            errors.append(f"forecast_distribution missing keys: {missing}")
        if extra:
            errors.append(f"forecast_distribution has extra keys: {extra}")
        probs = []
        for key in FORECAST_KEYS:
            value = forecast_value(dist, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"forecast_distribution.{key} must be numeric")
                continue
            if value < 0 or value > 1:
                errors.append(f"forecast_distribution.{key} must be in [0, 1]")
            probs.append(float(value))
        if len(probs) == len(FORECAST_KEYS):
            total = sum(probs)
            if not 0.95 <= total <= 1.05:
                errors.append(f"forecast_distribution must sum to approximately 1.0, got {total:.6f}")
    return not errors, errors


def normalize_distribution_if_valid(data: dict[str, Any]) -> dict[str, Any]:
    ok, errors = validate_rationale_schema_strict(data)
    if not ok:
        raise ValueError("; ".join(errors))
    out = copy.deepcopy(data)
    dist = out["forecast_distribution"]
    total = sum(float(forecast_value(dist, k)) for k in FORECAST_KEYS)
    if total <= 0:
        raise ValueError("forecast_distribution total must be positive")
    out["forecast_distribution"] = {k: float(forecast_value(dist, k)) / total for k in FORECAST_KEYS}
    return out


def parse_llm_json(text: str) -> dict[str, Any] | None:
    return parse_llm_json_strict(text)


def validate_rationale_schema(data: dict[str, Any] | None) -> bool:
    ok, _ = validate_rationale_schema_strict(data)
    return ok
