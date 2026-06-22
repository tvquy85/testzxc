from pathlib import Path


def test_runbook_mentions_claim_restriction():
    p = Path("RUNBOOK_currentdata_v6.md")
    assert p.exists()
    text = p.read_text(encoding="utf-8").lower()
    assert "claim" in text and ("restricted" in text or "blocked" in text)
