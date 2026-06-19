from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LABELS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
TITLE_LABELS = {
    "strong_down": "Strong Down",
    "mild_down": "Mild Down",
    "neutral": "Neutral",
    "mild_up": "Mild Up",
    "strong_up": "Strong Up",
}


def parsed_json_to_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            data = json.loads(value)
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def true_label_probability(parsed_json: Any, true_label: str | None) -> float | None:
    data = parsed_json_to_dict(parsed_json)
    if not data or true_label not in LABELS:
        return None
    dist = data.get("forecast_distribution")
    if not isinstance(dist, dict):
        return None
    value = dist.get(true_label)
    if value is None:
        value = dist.get(TITLE_LABELS.get(true_label, ""))
    return float(value) if isinstance(value, (int, float)) else None


def entropy_from_distribution(parsed_json: Any) -> float | None:
    import math

    data = parsed_json_to_dict(parsed_json)
    if not data:
        return None
    dist = data.get("forecast_distribution")
    if not isinstance(dist, dict):
        return None
    probs = [float(dist.get(label, dist.get(TITLE_LABELS[label], 0.0))) for label in LABELS]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def build_json_only_prompt(rationale_text: str, true_label: str, reversed_order: bool = False) -> str:
    labels = list(reversed(LABELS)) if reversed_order else LABELS
    return (
        "You are a strict financial forecast judge. Read the rationale and return JSON only with "
        "probability_by_label, predicted_label, and explanation. Do not infer missing probabilities.\n"
        f"Allowed labels in this order: {labels}\n"
        f"True label for scoring: {true_label}\n"
        f"Rationale:\n{rationale_text}"
    )


def resolve_snapshot(path_text: str | None) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return path_text
    refs_main = path / "refs" / "main"
    if refs_main.exists():
        snapshot = path / "snapshots" / refs_main.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return str(snapshot)
    snapshots = path / "snapshots"
    if snapshots.exists():
        candidates = sorted([p for p in snapshots.iterdir() if p.is_dir()])
        if candidates:
            return str(candidates[-1])
    return str(path)
