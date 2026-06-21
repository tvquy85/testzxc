# 06 — Rationale Diversity Template Audit

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Quantify whether rationales are template-heavy or genuinely diverse.

## Why this is needed
Clean V4 samples looked schema-stable but sometimes mapped technical tokens to repetitive narratives. DPO pairs are weak if candidates differ only in tiny wording.

## Files to create or modify

Create `src/llm/audit_rationale_diversity_v4.py`.
Use lexical Jaccard among candidates from the same `sample_id`, phrase matching, and citation coverage.


## Inputs
```text
data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet
data/processed/medium_clean_v4_contexts_gated.parquet
```

## Outputs
```text
outputs/metrics/06_rationale_diversity_template_audit.json
review_samples/medium_clean_v4/rationale_template_audit_samples.jsonl
outputs/status/06_RATIONALE_DIVERSITY_TEMPLATE_AUDIT.status.json
```

## Commands
```bash
python -m src.llm.audit_rationale_diversity_v4 \
  --rationales data/rationales/parsed/medium_clean_v4_qwen3_4b_candidates.parquet \
  --contexts data/processed/medium_clean_v4_contexts_gated.parquet \
  --output outputs/metrics/06_rationale_diversity_template_audit.json \
  --samples-output review_samples/medium_clean_v4/rationale_template_audit_samples.jsonl \
  --status outputs/status/06_RATIONALE_DIVERSITY_TEMPLATE_AUDIT.status.json
```

## Verification
```bash
python - <<'PY'
import json
m=json.load(open('outputs/metrics/06_rationale_diversity_template_audit.json'))
for k in ['technical_only_phrase_rate','news_evidence_citation_rate','repeated_template_cluster_rate']:
    assert k in m
print('PASS diversity audit')
PY
```

## Acceptance criteria
- Reports template metrics; does not hide them.
- Fails only if `news_evidence_citation_rate < 0.70` for contexts with company evidence or `repeated_template_cluster_rate > 0.70`.

## Status JSON contract
Write a status file with this shape:
```json
{
  "step": "06_RATIONALE_DIVERSITY_TEMPLATE_AUDIT",
  "status": "PASS|FAIL",
  "pipeline_pass": true,
  "claim_allowed": false,
  "inputs": [],
  "outputs": [],
  "metrics": {},
  "failures": [],
  "warnings": []
}
```
