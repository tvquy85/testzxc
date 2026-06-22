import json

from src.alignment.summarize_training_v6 import summarize_child


def test_summarize_child_requires_min_steps_and_adapter(tmp_path):
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    status = {
        "status": "PASS",
        "metrics": {
            "max_steps": 2,
            "losses": [1.0, 0.9],
            "loss_first": 1.0,
            "loss_last": 0.9,
        },
    }

    summary, failures = summarize_child("RWSFT", status, str(adapter), min_steps=800)

    assert summary["adapter_exists"] is True
    assert any("max_steps 2 < 800" in failure for failure in failures)


def test_summarize_child_passes_with_adapter_steps_and_finite_losses(tmp_path):
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    status = {
        "status": "PASS",
        "metrics": {
            "max_steps": 800,
            "dpo_losses": [0.7, 0.6],
            "dpo_loss_first": 0.7,
            "dpo_loss_last": 0.6,
        },
    }

    summary, failures = summarize_child("DPO", status, str(adapter), min_steps=800)

    assert failures == []
    assert summary["max_steps"] == 800
    assert summary["adapter_sha256"]
