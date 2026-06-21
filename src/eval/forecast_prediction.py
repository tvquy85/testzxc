from __future__ import annotations

import json
import math
from typing import Any

from src.llm.parse_and_validate_rationale import (
    FORECAST_KEYS,
    canonical_forecast_key,
    forecast_value,
    parse_llm_json_strict,
)
from src.llm.render_context import render_context


FORECAST_TITLE = {
    "strong_down": "Strong Down",
    "mild_down": "Mild Down",
    "neutral": "Neutral",
    "mild_up": "Mild Up",
    "strong_up": "Strong Up",
}
LABEL_FROM_KEY = {
    "strong_down": "strong_down",
    "mild_down": "mild_down",
    "neutral": "neutral",
    "mild_up": "mild_up",
    "strong_up": "strong_up",
}


def clip_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [TRUNCATED]"


def format_forecast_context(row: Any, body_chars: int = 1200, token_chars: int = 1200) -> str:
    clean_context = row.get("clean_context_text", "") if hasattr(row, "get") else ""
    if isinstance(clean_context, str) and clean_context.strip():
        ticker = row.get("ticker", "") if hasattr(row, "get") else ""
        event_date = row.get("event_date", "") if hasattr(row, "get") else ""
        return (
            f"Ticker: {clip_text(ticker, 32)}\n"
            f"Event Date: {clip_text(event_date, 64)}\n"
            f"Evidence Context:\n{clip_text(clean_context, body_chars + token_chars)}"
        )
    rendered = render_context(row)
    ticker = row.get("ticker", "") if hasattr(row, "get") else ""
    event_date = row.get("event_date", "") if hasattr(row, "get") else ""
    technical = clip_text(rendered.get("technical_event_tokens", ""), token_chars)
    return (
        f"Ticker: {clip_text(ticker, 32)}\n"
        f"Event Date: {clip_text(event_date, 64)}\n"
        f"News Headline: {clip_text(rendered.get('headline', ''), 360)}\n"
        f"News Body: {clip_text(rendered.get('body', ''), body_chars)}\n"
        f"Market Regime: {clip_text(rendered.get('regime_label', 'normal_vol'), 80)}\n"
        f"Technical Indicator Tokens:\n{technical}"
    )


def render_forecast_prompt(prompt_template: str, context: str) -> str:
    if "{context}" in prompt_template:
        return prompt_template.replace("{context}", context)
    return f"{prompt_template.rstrip()}\n\nContext:\n{context}\n\nReturn JSON only."


def validate_forecast_prediction(data: dict[str, Any] | None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["data is not a JSON object"]
    dist = data.get("forecast_distribution")
    if not isinstance(dist, dict):
        errors.append("forecast_distribution must be an object")
    else:
        canonical_keys = [canonical_forecast_key(str(key)) for key in dist]
        missing = [key for key in FORECAST_KEYS if key not in canonical_keys]
        extra = sorted(str(key) for key, canonical in zip(dist.keys(), canonical_keys) if canonical not in FORECAST_KEYS)
        if missing:
            errors.append(f"forecast_distribution missing keys: {missing}")
        if extra:
            errors.append(f"forecast_distribution has extra keys: {extra}")
        probs: list[float] = []
        for key in FORECAST_KEYS:
            value = forecast_value(dist, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"forecast_distribution.{key} must be finite numeric")
                continue
            value = float(value)
            if value < 0 or value > 1:
                errors.append(f"forecast_distribution.{key} must be in [0, 1]")
            probs.append(value)
        if len(probs) == len(FORECAST_KEYS):
            total = sum(probs)
            if not 0.99 <= total <= 1.01:
                errors.append(f"forecast_distribution must sum to 1.0 +/- 0.01, got {total:.6f}")
    if data.get("action") not in {"long", "short", "hold"}:
        errors.append("action must be one of long, short, hold")
    return not errors, errors


def parse_forecast_prediction(raw_text: str) -> tuple[dict[str, float], str, str, bool, bool, list[str]]:
    zero_dist = {key: 0.0 for key in FORECAST_KEYS}
    parsed = parse_llm_json_strict(raw_text)
    if parsed is None:
        return zero_dist, "invalid", "invalid", False, False, ["invalid_json"]
    schema_ok, errors = validate_forecast_prediction(parsed)
    if not schema_ok:
        return zero_dist, "invalid", "invalid", True, False, errors
    raw_dist = parsed["forecast_distribution"]
    total = sum(float(forecast_value(raw_dist, key)) for key in FORECAST_KEYS)
    dist = {key: float(forecast_value(raw_dist, key)) / total for key in FORECAST_KEYS}
    pred_key = max(dist, key=dist.get)
    return dist, str(parsed["action"]), LABEL_FROM_KEY[pred_key], True, True, []


def forecast_score(dist: dict[str, float]) -> float:
    return (
        -2.0 * float(dist.get("strong_down", 0.0))
        - float(dist.get("mild_down", 0.0))
        + float(dist.get("mild_up", 0.0))
        + 2.0 * float(dist.get("strong_up", 0.0))
    )


def distribution_from_row(row: Any) -> dict[str, float]:
    return {
        "strong_down": float(row.get("p_strong_down", 0.0)),
        "mild_down": float(row.get("p_mild_down", 0.0)),
        "neutral": float(row.get("p_neutral", 0.0)),
        "mild_up": float(row.get("p_mild_up", 0.0)),
        "strong_up": float(row.get("p_strong_up", 0.0)),
    }
