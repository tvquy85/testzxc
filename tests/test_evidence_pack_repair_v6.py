import json

from src.data.repair_evidence_pack_v6 import repair_evidence_pack, validate_evidence_pack

def test_reject_empty_news_evidence():
    pack=[{'evidence_id':'N1','article_type':'empty_or_weak','headline':'','body_excerpt':''}]
    assert validate_evidence_pack(pack)


def test_accept_v6_dict_pack_with_technical_signals():
    pack = {
        "company_evidence": [
            {"evidence_id": "N1", "article_type": "sector_etf", "headline": "AAPL update", "body_excerpt": "valid text"}
        ],
        "context_evidence": [],
        "technical_signals": [{"token": "MACD_BEARISH"}],
    }
    assert validate_evidence_pack(pack) == []


def test_repair_preserves_v6_dict_shape():
    pack = {
        "company_evidence": [
            {"evidence_id": "N1", "article_type": "empty_or_weak", "headline": "", "body_excerpt": ""},
            {"evidence_id": "N2", "article_type": "earnings_or_guidance", "headline": "Beat", "body_excerpt": "valid"},
        ],
        "context_evidence": [],
        "technical_signals": [{"token": "MACD_BEARISH"}],
    }
    repaired, failures = repair_evidence_pack(json.dumps(pack))
    parsed = json.loads(repaired)
    assert failures == []
    assert set(parsed) == {"company_evidence", "context_evidence", "technical_signals"}
    assert [item["evidence_id"] for item in parsed["company_evidence"]] == ["N2"]
