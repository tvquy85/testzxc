# 00 — Master Clean V4 Medium Order

> **Scope:** work on branch `currentdata-aaai-fix-v2`, current-data only. Do **not** use SN2. Do **not** expand to full FNSPID. Build on existing DataClean V4 artifacts and write new artifacts with suffix `_medium_v5` or `_clean_v4_medium`.  
> **Codex rule:** execute one file at a time, verify, write status JSON, stop on FAIL. PASS means the step ran; it does not automatically allow a scientific claim.


## Goal
Upgrade the current-data Clean V4 pipeline from small-scale validation to medium-scale validation before full-scale FNSPID expansion.

## Current evidence to respect
The Clean V4 report says the pipeline is `GO_SMALL` with `CLAIM_RESTRICTED`; trading alpha and AAAI-main readiness remain blocked. The previous reviewer analysis also found template-heavy rationales, weak news counterfactuals, small Flow data, and lack of V4 adapter evaluation.

## Medium-scale minimum targets
```text
Train rationale samples: >= 500 unique sample_ids × 3 candidates
Preferred rationale scale: 1000 unique sample_ids × 3 candidates
Independent judge: all candidates, >= 2 label-order variants
Flow dataset: >= 1500 rows minimum; preferred >= 3000
RWSFT: >= 1000 examples
DPO: >= 300 pairs; preferred >= 1000
Test predictions: >= 150 rows minimum; preferred >= 500
Backtest days: >= 30 for diagnostics; >= 60 to consider alpha claim
```

## Global stop rules
- If `schema_ok_rate < 0.95` during rationale generation, stop and fix prompt/parser.
- If independent judge mean true-label probability <= 0.20, stop and fix judge/prompt.
- If DPO pairs < 300, do not train DPO.
- If adapter files are missing, do not run prediction/backtest.
- If tests fail, stop.

## Always run
```bash
python -m pytest -q tests
```

## Final decision logic
`pipeline_go_medium` can be true even if scientific claims are false. `aaai_main_ready_allowed` should remain false at medium stage unless full statistical validation exists.
