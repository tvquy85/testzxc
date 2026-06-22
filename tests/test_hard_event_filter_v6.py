from src.data.filter_hard_event_v6 import assign_v6_track

def test_hard_event_track():
    row={'num_hard_event_evidence':1,'mean_evidence_quality_score':0.70,'has_company_event_news':True}
    assert assign_v6_track(row)=='hard_event_news'

def test_weak_track():
    row={'num_hard_event_evidence':0,'mean_evidence_quality_score':0.30,'has_company_event_news':False}
    assert assign_v6_track(row)=='weak_or_context_only'
