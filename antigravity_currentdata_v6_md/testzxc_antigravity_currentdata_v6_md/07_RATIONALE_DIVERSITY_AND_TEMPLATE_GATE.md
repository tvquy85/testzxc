# 07 — Rationale Diversity and Template-Heavy Gate V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Measure and reduce template-heavy outputs. Medium showed high within-sample Jaccard and repeated templates.

## Outputs
```text
outputs/metrics/07_v6_rationale_diversity.json
outputs/tables/07_v6_template_clusters.csv
review_samples/currentdata_v6/07_template_heavy_examples.jsonl
outputs/status/07_RATIONALE_DIVERSITY_AND_TEMPLATE_GATE.status.json
```

## Metrics
`mean_within_sample_jaccard`, `repeated_template_cluster_rate`, `technical_only_phrase_rate`, `news_rationale_empty_when_N_rate`, `evidence_citation_rate`, `semantic_diversity_score`.

## Gates
Preferred: Jaccard <=0.68, template cluster <=0.25, technical-only phrase <=0.15. Required: news empty with N evidence <=0.05.

## Test case
```python
from src.llm.audit_rationale_diversity_v6 import jaccard

def test_jaccard_basic():
    assert jaccard('a b c','a b d') == 0.5
    assert jaccard('', 'a') == 0.0
```

## Commands
```bash
python -m src.llm.audit_rationale_diversity_v6 --input data/rationales/parsed/current_v6_train_qwen3_1000x3.parquet --metrics outputs/metrics/07_v6_rationale_diversity.json --clusters outputs/tables/07_v6_template_clusters.csv --samples review_samples/currentdata_v6/07_template_heavy_examples.jsonl --status outputs/status/07_RATIONALE_DIVERSITY_AND_TEMPLATE_GATE.status.json
python -m pytest -q tests/test_rationale_diversity_v6.py tests
```
