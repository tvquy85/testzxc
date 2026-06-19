import json
import os
from src.llm.rationale_schema import RationaleOutput

def test_schema():
    good_json = {
        "news_rationale": ["Company announced strong earnings."],
        "technical_rationale": ["RSI is overbought."],
        "conflict_resolution": "News is bullish but technicals suggest a short-term pullback.",
        "forecast_distribution": {
            "strong_down": 0.05,
            "mild_down": 0.45,
            "neutral": 0.30,
            "mild_up": 0.15,
            "strong_up": 0.05
        },
        "action": "short",
        "risk_note": "Earnings momentum might overpower technical pullback."
    }

    bad_json = {
        "news_rationale": ["Bad earnings."],
        "technical_rationale": [],
        "conflict_resolution": "Very bad.",
        "forecast_distribution": {
            "strong_down": 0.50,
            "mild_down": 0.50,
            "neutral": 0.0,
            "mild_up": 0.0,
            "strong_up": 0.0
        },
        "action": "long", # Action conflicts with down probability
        "risk_note": "None."
    }

    # Test Good JSON
    try:
        RationaleOutput(**good_json)
        print("[PASS] Good JSON validated successfully.")
    except Exception as e:
        print(f"[FAIL] Good JSON failed validation: {e}")
        assert False

    # Test Bad JSON
    try:
        RationaleOutput(**bad_json)
        print("[FAIL] Bad JSON erroneously validated.")
        assert False
    except ValueError as e:
        if "Action is 'long' but downward probability > 60%" in str(e):
            print("[PASS] Bad JSON properly rejected due to action inconsistency.")
        else:
            print(f"[FAIL] Bad JSON rejected for wrong reason: {e}")
            assert False

    status = {
        "step": "07_RATIONALE_SCHEMA_AND_PROMPTS",
        "status": "PASS",
        "prompt_files": [
            "prompts/rationale_generation_prompt.txt",
            "prompts/proxy_inferability_judge_prompt.txt",
            "prompts/financial_soundness_judge_prompt.txt",
            "prompts/counterfactual_prompt.txt"
        ],
        "schema_file": "src/llm/rationale_schema.py",
        "example_rendered": True,
        "notes": "Schema validates probabilities and action consistency."
    }
    
    os.makedirs("outputs/status", exist_ok=True)
    with open("outputs/status/07_RATIONALE_SCHEMA_AND_PROMPTS.status.json", "w") as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    test_schema()
