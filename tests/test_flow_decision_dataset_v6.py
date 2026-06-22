from pathlib import Path

import torch

from src.reward.build_flow_decision_dataset_v6 import action_from_distribution


def test_flow_decision_dataset_shape():
    p = Path("data/reward/current_v6_flow_decision_dataset.pt")
    assert p.exists()
    data = torch.load(p, map_location="cpu", weights_only=False)
    assert data["target"].shape[1] == 5
    assert data["cond"].shape[0] == data["target"].shape[0]
    assert set(data["split"]).issuperset({"train", "val"})


def test_flow_decision_dataset_auxiliary_contract():
    p = Path("data/reward/current_v6_flow_decision_dataset.pt")
    assert p.exists()
    data = torch.load(p, map_location="cpu", weights_only=False)
    assert data["metadata"]["target_source"] == "calibrated_debiased_judge_distribution"
    assert data["metadata"]["utility_is_auxiliary"] is True
    assert data["auxiliary"]["judge_reliability_weight"]
    assert min(data["auxiliary"]["judge_reliability_weight"]) >= 0.0
    assert max(data["auxiliary"]["judge_reliability_weight"]) <= 1.0


def test_action_from_distribution_prefers_neutral_on_tie():
    row = {
        "p_strong_down": 0.05,
        "p_mild_down": 0.20,
        "p_neutral": 0.50,
        "p_mild_up": 0.20,
        "p_strong_up": 0.05,
    }

    assert action_from_distribution(row) == "hold"
