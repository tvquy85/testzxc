from pathlib import Path


def test_v6_adapters_exist():
    assert Path("outputs/models/qwen3_current_v6_rwsft_adapter/adapter_model.safetensors").exists()
    assert Path("outputs/models/qwen3_current_v6_dpo_adapter/adapter_model.safetensors").exists()
