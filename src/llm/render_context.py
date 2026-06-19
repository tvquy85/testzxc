import json

def render_context(row):
    """
    row is a pandas Series with news and technical indicators.
    Returns a dict with context variables to format the prompt.
    """
    
    headline = row.get("headline", "")
    body = row.get("body", "")
    regime = row.get("regime_label", "normal_vol")
    
    # technical tokens
    tokens = row.get("technical_event_tokens_json", row.get("technical_event_tokens", "[]"))
    if isinstance(tokens, str):
        try:
            tokens_list = json.loads(tokens)
            if tokens_list and isinstance(tokens_list[0], dict):
                tech_text = "\n".join(
                    f"[{item.get('token')}: value={item.get('value')}, direction={item.get('direction_prior')}, strength={item.get('strength')}, rule={item.get('rule')}]"
                    for item in tokens_list
                )
            else:
                tech_text = "\n".join(str(item) for item in tokens_list)
        except Exception:
            tech_text = tokens
    elif isinstance(tokens, list):
        tech_text = "\n".join(tokens)
    else:
        tech_text = str(tokens)
        
    return {
        "headline": headline,
        "body": body,
        "regime_label": regime,
        "technical_event_tokens": tech_text
    }
