import os

from src.judges import claim_level_grounding_v3 as grounding


def test_nli_label_mapping_is_fixed_for_local_deberta():
    assert grounding.NLI_ID2LABEL == {0: "contradiction", 1: "entailment", 2: "neutral"}
    assert grounding.nli_label_to_status("entailment") == "supported"
    assert grounding.nli_label_to_status("contradiction") == "contradiction"
    assert grounding.nli_label_to_status("neutral") == "unverified"


def test_load_nli_model_uses_model_id_and_hf_home(monkeypatch):
    calls = {}

    class FakeGrounder:
        def __init__(self, model_id, hf_home):
            calls["model_id"] = model_id
            calls["hf_home"] = hf_home

    monkeypatch.setattr(grounding, "NLIGrounder", FakeGrounder)

    nli, info = grounding.load_nli_model("cross-encoder/nli-deberta-v3-small", "E:/huggingface")

    assert nli is not None
    assert calls == {"model_id": "cross-encoder/nli-deberta-v3-small", "hf_home": "E:/huggingface"}
    assert os.environ["HF_HOME"] == "E:/huggingface"
    assert info["nli_backend"] is True
    assert info["nli_loader"] == "transformers_model_id"
    assert info["nli_local_files_only"] is True
    assert info["nli_failure"] is None


def test_failed_nli_load_does_not_claim_backend(monkeypatch):
    class BrokenGrounder:
        def __init__(self, model_id, hf_home):
            raise RuntimeError("local cache missing")

    monkeypatch.setattr(grounding, "NLIGrounder", BrokenGrounder)

    nli, info = grounding.load_nli_model("cross-encoder/nli-deberta-v3-small", "E:/huggingface")

    assert nli is None
    assert info["nli_backend"] is False
    assert info["nli_loader"] is None
    assert "local cache missing" in info["nli_failure"]


def test_required_nli_fails_without_lexical_escape_hatch():
    assert grounding.nli_gate_failures(None, require_nli=True, allow_lexical_fallback=False) == [
        "required NLI backend unavailable"
    ]
    assert grounding.nli_gate_failures(None, require_nli=False, allow_lexical_fallback=False) == [
        "NLI backend unavailable and lexical fallback was not explicitly allowed"
    ]
    assert grounding.nli_gate_failures(None, require_nli=False, allow_lexical_fallback=True) == []
