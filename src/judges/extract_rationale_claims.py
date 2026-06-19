from __future__ import annotations

import json
from typing import Any


def as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            data = json.loads(value)
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def rationale_item_text(item: Any) -> str:
    if isinstance(item, dict):
        subject = item.get("factor", item.get("signal", item.get("text", "")))
        direction = item.get("direction", "")
        strength = item.get("strength", "")
        parts = [str(x) for x in [subject, direction, strength] if str(x).strip()]
        return " | ".join(parts)
    return str(item)


def extract_claims(parsed_json: Any) -> list[dict[str, str]]:
    data = as_dict(parsed_json)
    if not data:
        return []
    claims: list[dict[str, str]] = []
    for text in data.get("news_rationale", []) or []:
        claims.append({"claim_type": "news_claim", "claim_text": rationale_item_text(text)})
    for text in data.get("technical_rationale", []) or []:
        claims.append({"claim_type": "technical_claim", "claim_text": rationale_item_text(text)})
    if data.get("conflict_resolution"):
        claims.append({"claim_type": "regime_claim", "claim_text": str(data["conflict_resolution"])})
    if data.get("forecast_distribution"):
        claims.append({"claim_type": "forecast_claim", "claim_text": json.dumps(data["forecast_distribution"], sort_keys=True)})
    if data.get("risk_note"):
        claims.append({"claim_type": "risk_claim", "claim_text": str(data["risk_note"])})
    return claims
