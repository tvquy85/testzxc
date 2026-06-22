# 06.5 — Post-Processing & Rationale Repair V6

> Scope: branch `currentdata-aaai-fix-v2`; current-data only.

## Goal
Fix raw outputs from Step 06 that suffered from truncation, empty `news_rationale` when N evidence exists, and inconsistent `action` mapping.

## Outputs
```text
data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet
outputs/metrics/06_5_post_processing_repair.json
outputs/status/06_5_POST_PROCESSING_AND_REPAIR_V6.status.json
```

## Intervention Logic

1. **Filtering Errors**: 
   A rationale is invalid if:
   - `parse_ok == False` (e.g. truncated JSON)
   - `schema_ok == False`
   - `has_N_evidence == True` but `len(parsed_json['news_rationale']) == 0`
   
2. **Action Inconsistency Fix**:
   Derive `action` from `forecast_distribution` for all valid rationales.
   `score = p_strong_up*2 + p_mild_up*1 + p_neutral*0 + p_mild_down*(-1) + p_strong_down*(-2)`
   `derived_action` = 'long' if score >= 0.2 else 'short' if score <= -0.2 else 'hold'.

3. **Top-Up Generation**:
   Re-generate only the failed candidates with `max_new_tokens=384` and `temperature=0.65`.
   
4. **Merge**:
   Merge the fixed top-up candidates with the original valid ones to form a complete, fully valid 3000-candidate dataset.

## Commands
```bash
python -m src.repro.repair_rationales_v6
```
