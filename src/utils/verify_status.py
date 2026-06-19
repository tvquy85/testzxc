from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import STATUS_KEYS


def validate_status(path: str | Path) -> tuple[bool, list[str]]:
    p = Path(path)
    failures: list[str] = []
    if not p.exists():
        return False, [f"status file does not exist: {p}"]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"status file is not valid JSON: {exc}"]

    missing = sorted(STATUS_KEYS - set(data))
    if missing:
        failures.append(f"missing required keys: {missing}")

    if data.get("status") not in {"PASS", "FAIL"}:
        failures.append("status must be PASS or FAIL")
    if not isinstance(data.get("inputs_checked", []), list):
        failures.append("inputs_checked must be a list")
    if not isinstance(data.get("outputs_created", []), list):
        failures.append("outputs_created must be a list")
    if not isinstance(data.get("metrics", {}), dict):
        failures.append("metrics must be an object")
    if not isinstance(data.get("failures", []), list):
        failures.append("failures must be a list")
    if not isinstance(data.get("next_step_allowed"), bool):
        failures.append("next_step_allowed must be boolean")

    outputs = data.get("outputs_created", [])
    missing_outputs = [out for out in outputs if not Path(out).exists()]
    if missing_outputs:
        failures.append(f"outputs_created missing on disk: {missing_outputs}")

    if data.get("status") == "PASS":
        if data.get("failures"):
            failures.append("PASS status cannot include failures")
        if data.get("next_step_allowed") is not True:
            failures.append("PASS status must set next_step_allowed=true")
        if missing_outputs:
            failures.append("PASS status has missing outputs")
    if data.get("status") == "FAIL" and data.get("next_step_allowed") is True:
        failures.append("FAIL status must set next_step_allowed=false")

    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    ok, failures = validate_status(args.status)
    if ok:
        print(f"PASS: {args.status}")
        return 0
    print(f"FAIL: {args.status}")
    for failure in failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
