from src.llm.audit_rationale_diversity_v6 import jaccard

def test_jaccard_basic():
    assert jaccard('a b c','a b d') == 0.5
    assert jaccard('', 'a') == 0.0
