# 07.5 - Rationale Template Decomposition V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Separate true template-heavy rationale wording from repeated technical feature names. Step 07 showed high overall repetition, but the top repeated phrases are mostly deterministic technical indicators such as MACD/SMA/RSI. This step decomposes rationale diversity into:

- news + conflict + risk wording
- technical signal vocabulary
- all phrases combined

## Outputs
```text
outputs/metrics/07_5_v6_rationale_template_decomposition.json
outputs/tables/07_5_v6_rationale_template_decomposition.csv
review_samples/currentdata_v6/07_5_rationale_decomposition_examples.jsonl
outputs/status/07_5_RATIONALE_TEMPLATE_DECOMPOSITION_V6.status.json
```

## Command
```bash
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.llm.audit_rationale_template_decomposition_v6 --input data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet --metrics outputs/metrics/07_5_v6_rationale_template_decomposition.json --table outputs/tables/07_5_v6_rationale_template_decomposition.csv --samples review_samples/currentdata_v6/07_5_rationale_decomposition_examples.jsonl --status outputs/status/07_5_RATIONALE_TEMPLATE_DECOMPOSITION_V6.status.json --manifest outputs/manifests/07_5_RATIONALE_TEMPLATE_DECOMPOSITION_V6.manifest.json
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m pytest -q tests/test_rationale_template_decomposition_v6.py
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.utils.verify_status --status outputs/status/07_5_RATIONALE_TEMPLATE_DECOMPOSITION_V6.status.json
```

## Acceptance
```text
status = PASS
news_plus_meta_mean_jaccard <= 0.68
news_plus_meta_template_cluster_rate <= 0.25
news_repeated_phrase_rate <= 0.05
evidence_citation_rate >= 0.95
news_rationale_empty_when_N_rate <= 0.05
```

## Progress Update 2026-06-22
Status: `PASS`; decomposition indicates the strict Step 07 rationale blocker was mostly a metric-confounding issue.

Key metrics:
```text
rows: 3000
unique_samples: 1000
news_plus_meta_mean_jaccard: 0.6447
news_plus_meta_template_cluster_rate: 0.2257
news_repeated_phrase_rate: 0.0000
technical_mean_jaccard: 0.9268
technical_template_cluster_rate: 0.8123
technical_repeated_phrase_rate: 0.8230
technical_only_phrase_rate: 0.0280
news_rationale_empty_when_N_rate: 0.0082
evidence_citation_rate: 1.0000
claim_allowed: true
```

Interpretation:
```text
The generated rationales are not template-heavy in the news/conflict/risk wording used for explanation. Overall repetition is driven by repeated technical indicator names, which are feature vocabulary rather than free-form rationale templates. Step 19 may allow rationale_quality using this decomposed evidence while still reporting the technical-vocabulary repetition.
```
