import pandas as pd

from src.alignment.build_alignment_v6 import (
    accept_pair,
    build_dpo_records,
    build_rwsft_records,
    semantic_distance,
    valid_candidates,
)


def test_accept_pair_gap_and_distance():
    assert accept_pair(0.70, 0.60, 0.20)
    assert not accept_pair(0.70, 0.68, 0.20)
    assert not accept_pair(0.70, 0.60, 0.05)


def test_semantic_distance_ignores_schema_tokens():
    left = '{"news_rationale": [{"factor": "margin pressure", "direction": "down"}], "action": "short"}'
    right = '{"news_rationale": [{"factor": "margin pressure", "direction": "down"}], "action": "short"}'
    other = '{"news_rationale": [{"factor": "cloud revenue acceleration", "direction": "up"}], "action": "long"}'

    assert semantic_distance(left, right) == 0.0
    assert semantic_distance(left, other) >= 0.15


def test_build_records_keep_train_format_and_pair_constraints():
    rows = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "candidate_id": 0,
                "split": "train",
                "prompt": "prompt",
                "raw_output": "margin pressure weaker sales",
                "parse_ok": True,
                "schema_ok": True,
                "judge_schema_ok": True,
                "target_label_5": "mild_down",
                "true_label_probability_ensemble": 0.70,
                "argmax_consistency_ensemble": 1.0,
                "final_reward": 0.70,
                "reward_source": "proxy_true_label_probability_ensemble",
                "p_strong_down": 0.1,
                "p_mild_down": 0.7,
                "p_neutral": 0.1,
                "p_mild_up": 0.05,
                "p_strong_up": 0.05,
            },
            {
                "sample_id": "s1",
                "candidate_id": 1,
                "split": "train",
                "prompt": "prompt",
                "raw_output": "cloud revenue acceleration backlog",
                "parse_ok": True,
                "schema_ok": True,
                "judge_schema_ok": True,
                "target_label_5": "mild_down",
                "true_label_probability_ensemble": 0.60,
                "argmax_consistency_ensemble": 1.0,
                "final_reward": 0.60,
                "reward_source": "proxy_true_label_probability_ensemble",
                "p_strong_down": 0.1,
                "p_mild_down": 0.6,
                "p_neutral": 0.2,
                "p_mild_up": 0.05,
                "p_strong_up": 0.05,
            },
        ]
    )

    valid = valid_candidates(rows)
    rwsft = build_rwsft_records(valid)
    dpo = build_dpo_records(valid, min_reward_gap=0.05, min_semantic_distance=0.15)

    assert len(rwsft) == 2
    assert rwsft[0]["split"] == "train"
    assert len(rwsft[0]["messages"]) == 2
    assert len(dpo) == 1
    assert dpo[0]["chosen_reward"] == 0.70
    assert dpo[0]["rejected_reward"] == 0.60
    assert dpo[0]["semantic_distance"] >= 0.15
