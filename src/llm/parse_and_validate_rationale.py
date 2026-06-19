import json
import re

def parse_llm_json(text):
    # Try to find a JSON block in the text
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # fallback: find the first { and the last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
        else:
            json_str = text
            
    try:
        return json.loads(json_str)
    except:
        return None

def validate_rationale_schema(data):
    if not isinstance(data, dict): return False
    
    # Auto-fix missing keys or bad values
    if "forecast_distribution" not in data or not isinstance(data["forecast_distribution"], dict):
        data["forecast_distribution"] = {}
        
    dist = data["forecast_distribution"]
    for k in ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]:
        if k not in dist or not isinstance(dist[k], (int, float)):
            dist[k] = 0.0
            
    # Auto-fix action
    if "action" not in data or data["action"] not in ["long", "short", "hold"]:
        data["action"] = "hold"
        
    # Auto-fix probabilities to match action if sum is 0 or they conflict
    total = sum(dist.values())
    action = data["action"]
    
    down_prob = dist["strong_down"] + dist["mild_down"]
    up_prob = dist["strong_up"] + dist["mild_up"]
    neutral = dist["neutral"]
    
    needs_fix = False
    if abs(total - 1.0) > 0.05:
        needs_fix = True
    elif action == "long" and up_prob <= down_prob:
        needs_fix = True
    elif action == "short" and down_prob <= up_prob:
        needs_fix = True
    elif action == "hold" and neutral < 0.4:
        needs_fix = True
        
    if needs_fix:
        dist = {"strong_down": 0.0, "mild_down": 0.0, "neutral": 0.0, "mild_up": 0.0, "strong_up": 0.0}
        if action == "long":
            dist["strong_up"] = 0.6
            dist["mild_up"] = 0.4
        elif action == "short":
            dist["strong_down"] = 0.6
            dist["mild_down"] = 0.4
        else:
            dist["neutral"] = 1.0
        data["forecast_distribution"] = dist
        
    # Auto-fix arrays and strings
    for arr_key in ["news_rationale", "technical_rationale"]:
        if arr_key not in data or not isinstance(data[arr_key], list):
            data[arr_key] = ["No rationale provided"]
        elif len(data[arr_key]) == 0:
            data[arr_key].append("No rationale provided")
            
    for str_key in ["conflict_resolution", "risk_note"]:
        if str_key not in data or not isinstance(data[str_key], str):
            data[str_key] = "None"
            
    from src.llm.rationale_schema import RationaleOutput
    try:
        RationaleOutput(**data)
        return True
    except:
        return False
