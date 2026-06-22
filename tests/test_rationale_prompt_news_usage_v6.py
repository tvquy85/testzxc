from src.llm.parse_and_validate_rationale_v6 import validate_rationale_v6

def test_news_required_when_company_evidence():
    parsed={'news_rationale':[],'technical_rationale':[{'signal_id':'T1','signal':'MACD bearish','direction':'negative','strength':'medium'}],'conflict_resolution':'Technicals dominate.','forecast_distribution':{'Strong Down':0.4,'Mild Down':0.3,'Neutral':0.2,'Mild Up':0.1,'Strong Up':0.0},'action':'short','risk_note':'test'}
    errors=validate_rationale_v6(parsed, evidence_ids={'N1'}, signal_ids={'T1'})
    assert any('news_rationale empty' in e for e in errors)
