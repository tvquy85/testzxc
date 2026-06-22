# 12 — Build Alignment Dataset with Minimum Reward Gap and Semantic Diversity

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Avoid DPO pairs that differ only by tiny probability tweaks or templates.

## Outputs
```text
data/alignment/current_v6_rwsft.jsonl
data/alignment/current_v6_dpo_pairs.jsonl
outputs/metrics/12_v6_alignment_dataset.json
review_samples/currentdata_v6/12_dpo_pair_samples.jsonl
outputs/status/12_ALIGNMENT_PAIRS_MIN_GAP_AND_DIVERSITY.status.json
```

## Pair selection
`chosen = highest reward valid candidate`; `rejected = lower reward candidate with reward gap >= 0.05 and semantic distance >= 0.15`. Use proxy reward if Flow V6 fails.

## Test case
```python
from src.alignment.build_alignment_v6 import accept_pair

def test_accept_pair_gap_and_distance():
    assert accept_pair(0.70,0.60,0.20)
    assert not accept_pair(0.70,0.68,0.20)
    assert not accept_pair(0.70,0.60,0.05)
```

## Commands
```bash
python -m src.alignment.build_alignment_v6 --rationales data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet --judge data/judges/current_v6_independent_judge_ensemble.parquet --flow-metrics outputs/metrics/11_v6_flow_vs_proxy_raw_utility.json --rwsft-output data/alignment/current_v6_rwsft.jsonl --dpo-output data/alignment/current_v6_dpo_pairs.jsonl --metrics outputs/metrics/12_v6_alignment_dataset.json --samples review_samples/currentdata_v6/12_dpo_pair_samples.jsonl --status outputs/status/12_ALIGNMENT_PAIRS_MIN_GAP_AND_DIVERSITY.status.json
python -m pytest -q tests/test_alignment_pair_selection_v6.py tests
```

## Acceptance
RWSFT >=1000, DPO >=300, mean gap >=0.035 diagnostic or >=0.05 preferred, semantic distance >=0.15.

## Progress Update 2026-06-22
Status: `stage_2_alignment_dataset PASS`; Step 12 used the repaired rationale artifact and the full Step 08 judge ensemble. Status file reports `PASS` and `next_step_allowed=true`.

Artifacts verified:
```text
data/alignment/current_v6_rwsft.jsonl
data/alignment/current_v6_dpo_pairs.jsonl
outputs/metrics/12_v6_alignment_dataset.json
review_samples/currentdata_v6/12_dpo_pair_samples.jsonl
outputs/manifests/12_ALIGNMENT_PAIRS_MIN_GAP_AND_DIVERSITY.manifest.json
outputs/status/12_ALIGNMENT_PAIRS_MIN_GAP_AND_DIVERSITY.status.json
```

Reward policy:
```text
Flow V6 claim was blocked in Step 11, so Step 12 did not use Flow scores.
reward_source: proxy_true_label_probability_ensemble
flow_claim_allowed: false
flow_reward_improvement: false
flow_scores_used: false
```

Selection rules implemented:
```text
valid candidate: rationale parse_ok/schema_ok, judge_schema_ok, finite probability simplex, train split
RWSFT: all valid train candidates, sorted by reward descending for the train loader
DPO chosen: highest proxy reward valid candidate per sample
DPO rejected: lower reward valid candidate with reward_gap >= 0.05 and semantic_distance >= 0.15
semantic distance: Jaccard distance over content tokens, excluding JSON schema/control tokens
```

Gate metrics from `outputs/metrics/12_v6_alignment_dataset.json`:
```text
rationale_rows: 3000
judge_rows: 3000
merged_rows: 3000
valid_candidate_rows: 2976
valid_unique_samples: 998
rwsft_examples: 2976
dpo_pairs: 316
dpo_unique_samples: 206
mean_reward_gap: 0.1737
min_observed_reward_gap: 0.0500
mean_semantic_distance: 0.3560
min_observed_semantic_distance: 0.1538
train_only: true
```

Acceptance result:
```text
RWSFT gate: PASS (2976 >= 1000)
DPO gate: PASS (316 >= 300)
mean gap diagnostic: PASS (0.1737 >= 0.035)
minimum pair gap: PASS (min 0.0500 >= 0.05)
minimum semantic distance: PASS (min 0.1538 >= 0.15)
status JSON gate: PASS
unit tests/full tests: PASS with D:\LOBProj\LOBExp\.venv\Scripts\python.exe
```

Verification commands run:
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_alignment_pair_selection_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.alignment.build_alignment_v6 --rationales data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet --judge data/judges/current_v6_independent_judge_ensemble.parquet --flow-metrics outputs/metrics/11_v6_flow_vs_proxy_raw_utility.json --rwsft-output data/alignment/current_v6_rwsft.jsonl --dpo-output data/alignment/current_v6_dpo_pairs.jsonl --metrics outputs/metrics/12_v6_alignment_dataset.json --samples review_samples/currentdata_v6/12_dpo_pair_samples.jsonl --status outputs/status/12_ALIGNMENT_PAIRS_MIN_GAP_AND_DIVERSITY.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/12_ALIGNMENT_PAIRS_MIN_GAP_AND_DIVERSITY.status.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests
```

Method basis:
```text
DPO requires prompt/chosen/rejected preference triplets; Step 12 builds this exact contract for train_dpo_v2.py.
Because Step 11 blocked Flow, the reward source is the independent judge proxy rather than a failed learned Flow reward.
The semantic filter removes near-duplicate/template pairs by measuring content-token Jaccard distance, not JSON schema-token overlap.
Label-order robustness from Step 08 remains enforced by requiring judge_schema_ok for alignment candidates.
```

Scientific caution: Step 12 proves only that clean RWSFT/DPO training data is now available at the required scale. It does not prove alignment improvement; Step 13 must train adapters and Step 14 must compare base/SFT/RWSFT/DPO before any forecast claim is allowed.
