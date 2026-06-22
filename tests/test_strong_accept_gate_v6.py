from src.repro.currentdata_v6_strong_accept_gate import allow_flow_claim


def test_flow_claim_requires_two_wins():
    assert allow_flow_claim({"rank": True, "pair": True, "top_decile": False})
    assert not allow_flow_claim({"rank": False, "pair": True, "top_decile": False})
