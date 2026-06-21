from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.parse_and_validate_rationale_v4 import validate_rationale_schema_evidence_v4
from src.utils.artifacts import write_manifest, write_status

STEP = "10_STRICT_EVIDENCE_SCHEMA_VALIDATION_V4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default=f"outputs/status/{STEP}_SMALL.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}_SMALL.manifest.json")
    args = parser.parse_args()

    valid = {
        "news_rationale": [{"evidence_id": "N1", "factor": "guidance improved", "direction": "positive", "strength": "medium"}],
        "technical_rationale": [{"signal_id": "T1", "signal": "MACD_BULLISH", "direction": "positive", "strength": "strong"}],
        "conflict_resolution": "Signals align.",
        "forecast_distribution": {"Strong Down": 0.05, "Mild Down": 0.10, "Neutral": 0.35, "Mild Up": 0.35, "Strong Up": 0.15},
        "action": "long",
        "risk_note": "Broad market weakness could offset evidence.",
    }
    ok, errors = validate_rationale_schema_evidence_v4(valid, {"evidence_ids": ["N1"], "signal_ids": ["T1"]})
    invalid = dict(valid)
    invalid["news_rationale"] = [{"evidence_id": "N9", "factor": "made up", "direction": "positive", "strength": "medium"}]
    invalid_ok, invalid_errors = validate_rationale_schema_evidence_v4(invalid, {"evidence_ids": ["N1"], "signal_ids": ["T1"]})
    failures: list[str] = []
    if not ok:
        failures.append(f"valid example failed: {errors}")
    if invalid_ok:
        failures.append("invalid unknown evidence_id example passed")
    metrics = {
        "valid_example_ok": bool(ok),
        "invalid_example_rejected": bool(not invalid_ok),
        "invalid_errors": invalid_errors,
    }
    write_manifest(args.manifest, ["src/llm/parse_and_validate_rationale_v4.py", "tests/test_rationale_schema_v4.py"], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, ["src/llm/parse_and_validate_rationale_v4.py"], [args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
