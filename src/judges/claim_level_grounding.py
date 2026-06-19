from __future__ import annotations

import json
from typing import Any


def token_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, str) and value.strip():
        try:
            data = json.loads(value)
            return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
        except Exception:
            return []
    return []


def score_technical_claim(claim: str, tokens_json: Any) -> dict[str, Any]:
    tokens = token_list(tokens_json)
    if not tokens:
        return {"score": None, "status": "not_applicable", "matched_tokens": []}
    claim_upper = claim.upper()
    matched = [t["token"] for t in tokens if str(t.get("token", "")).upper() in claim_upper]
    if matched:
        return {"score": 1.0, "status": "supported", "matched_tokens": matched}
    return {"score": 0.0, "status": "unverified", "matched_tokens": []}


def score_news_claim(claim: str, headline: str = "", body: str = "") -> dict[str, Any]:
    text = f"{headline} {body}".lower()
    words = [w.strip(".,;:!?()[]{}\"'").lower() for w in claim.split() if len(w) >= 4]
    if not words:
        return {"score": None, "status": "not_applicable"}
    overlap = sum(1 for w in words if w in text) / len(words)
    return {"score": float(overlap), "status": "supported" if overlap >= 0.35 else "unverified"}

