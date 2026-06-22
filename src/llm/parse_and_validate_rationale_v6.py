def validate_rationale_v6(parsed, evidence_ids, signal_ids):
    errors = []
    if not isinstance(parsed, dict):
        return ["not_a_json_object"]
    
    news_rationale = parsed.get("news_rationale", [])
    if not isinstance(news_rationale, list):
        errors.append("news_rationale_not_list")
        news_rationale = []

    tech_rationale = parsed.get("technical_rationale", [])
    if not isinstance(tech_rationale, list):
        errors.append("technical_rationale_not_list")
        tech_rationale = []

    has_company_evidence = any(eid.startswith("N") for eid in evidence_ids)
    
    if has_company_evidence and len(news_rationale) == 0:
        errors.append('news_rationale empty despite company evidence')

    if has_company_evidence and len(news_rationale) == 0 and len(tech_rationale) > 0:
        errors.append("technical_rationale may not be the only rationale section when N* evidence exists")

    for item in news_rationale:
        if not isinstance(item, dict):
            errors.append("news_rationale_item_not_dict")
            continue
        eid = item.get("evidence_id")
        if not eid or eid not in evidence_ids:
            errors.append(f"invalid_evidence_id:{eid}")
        
        d = item.get("direction")
        if d not in {"positive", "negative", "neutral"}:
            errors.append(f"invalid_news_direction:{d}")
        
        s = item.get("strength")
        if s not in {"weak", "medium", "strong"}:
            errors.append(f"invalid_news_strength:{s}")

    for item in tech_rationale:
        if not isinstance(item, dict):
            errors.append("technical_rationale_item_not_dict")
            continue
        sid = item.get("signal_id")
        if not sid or sid not in signal_ids:
            errors.append(f"invalid_signal_id:{sid}")
        
        d = item.get("direction")
        if d not in {"positive", "negative", "neutral"}:
            errors.append(f"invalid_technical_direction:{d}")
        
        s = item.get("strength")
        if s not in {"weak", "medium", "strong"}:
            errors.append(f"invalid_technical_strength:{s}")

    return errors
