import argparse
import json
from pathlib import Path
import pandas as pd
from src.utils.artifacts import write_status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()

    failures = []
    if not Path(args.prompt).exists():
        failures.append(f"Missing prompt file: {args.prompt}")
    if not Path(args.contexts).exists():
        failures.append(f"Missing contexts file: {args.contexts}")

    metrics = {"prompt_length": 0, "rules_count": 0}

    if not failures:
        content = Path(args.prompt).read_text(encoding='utf-8')
        metrics["prompt_length"] = len(content)
        metrics["rules_count"] = content.count("- ")
        
        # Ensure rules exist in prompt
        required_rules = [
            "If Company-specific evidence contains N1/N2/N3, news_rationale must contain at least one item",
            "Do not return an empty news_rationale when company-specific evidence exists"
        ]
        for rule in required_rules:
            if rule not in content:
                failures.append(f"Prompt is missing required rule segment: '{rule}'")

    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics, 'w') as f:
        json.dump(metrics, f, indent=2)

    status_str = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        step="05_RATIONALE_PROMPT_NEWS_USAGE_V6",
        status=status_str,
        inputs_checked=[args.prompt, args.contexts],
        outputs_created=[args.metrics, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=not failures
    )

    if failures:
        print(f"FAIL: {failures}")
        raise SystemExit(1)
    print("PASS 05_RATIONALE_PROMPT_NEWS_USAGE_V6")

if __name__ == "__main__":
    main()
