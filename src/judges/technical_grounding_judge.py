import re

def score_technical_grounding(rationale_text: str, technical_tokens: str) -> float:
    """
    Checks what fraction of the provided technical tokens are mentioned in the rationale.
    If no tokens were provided, defaults to 1.0 (perfectly grounded since there's nothing to miss).
    """
    if not rationale_text or not isinstance(rationale_text, str):
        return 0.0
        
    if not technical_tokens or not isinstance(technical_tokens, str) or technical_tokens.strip() == "":
        return 1.0
        
    # Extract tokens like [MACD_BULLISH], [PRICE_ABOVE_SMA20]
    tokens = re.findall(r'\[[A-Z0-9_]+\]', technical_tokens)
    if not tokens:
        # Maybe tokens are comma separated without brackets
        tokens = [t.strip() for t in technical_tokens.split(',') if t.strip()]
        
    if not tokens:
        return 1.0
        
    # Check mentions (case insensitive and ignoring brackets just in case)
    # We will look for the raw token name, e.g. MACD_BULLISH
    match_count = 0
    rationale_upper = rationale_text.upper()
    for t in tokens:
        clean_token = t.replace('[', '').replace(']', '').strip().upper()
        if clean_token in rationale_upper:
            match_count += 1
            
    return match_count / len(tokens)
