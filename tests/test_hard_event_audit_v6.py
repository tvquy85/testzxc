import json
from src.data.audit_hard_event_v6 import classify_context_hardness

def test_classify_hard_event_context():
    pack=json.dumps([{'evidence_id':'N1','article_type':'earnings_or_guidance','evidence_quality_score':0.85},{'evidence_id':'M1','article_type':'macro_market'}])
    out=classify_context_hardness(pack)
    assert out['num_hard_event_evidence']==1
    assert out['has_hard_event_news'] is True


def test_classify_hard_event_context_dict_pack():
    pack=json.dumps({
        'company_evidence':[{'evidence_id':'N1','article_type':'earnings_or_guidance','evidence_quality_score':0.85}],
        'context_evidence':[{'evidence_id':'M1','article_type':'macro_market'}],
    })
    out=classify_context_hardness(pack)
    assert out['num_hard_event_evidence']==1
    assert out['has_hard_event_news'] is True
