from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import write_manifest, write_status

STEP = "09_RATIONALE_PROMPT_EVIDENCE_ID_V4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="prompts/rationale_generation_prompt_evidence_v4.txt")
    parser.add_argument("--status", default=f"outputs/status/{STEP}_SMALL.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}_SMALL.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    text = Path(args.prompt).read_text(encoding="utf-8") if Path(args.prompt).exists() else ""
    required = ["evidence_id", "signal_id", "forecast_distribution", "Do not invent", "{context}"]
    for item in required:
        if item not in text:
            failures.append(f"prompt missing required text: {item}")
    metrics = {"prompt_chars": len(text), "required_terms_present": len(required) - len(failures)}
    write_manifest(args.manifest, [args.prompt], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.prompt], [args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
