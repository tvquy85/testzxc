# 11 — Evidence-Based Grounding Judge V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Judge whether each rationale claim is supported by the cited evidence item or technical signal.

## Inputs
- `data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet`
- `data/processed/ticker_date_evidence_contexts_h1_v4.parquet`
- local `cross-encoder/nli-deberta-v3-small` if available.

## Outputs
- `data/judges/claim_grounding_evidence_v4.parquet`
- `outputs/metrics/claim_grounding_evidence_v4.json`
- `outputs/status/11_EVIDENCE_GROUNDING_JUDGE_V4.status.json`

## Core logic
```python
def validate_news_claim(claim, evidence_pack, nli_model=None):
    eid = claim.get('evidence_id')
    if not eid:
        return {'status':'unsupported','reason':'missing_evidence_id','score':0.0}
    ev = next((x for x in evidence_pack if x.get('evidence_id') == eid), None)
    if ev is None:
        return {'status':'unsupported','reason':'unknown_evidence_id','score':0.0}
    premise = f"{ev.get('headline','')} {ev.get('body_excerpt','')}"
    hyp = str(claim.get('factor',''))
    lexical = lexical_overlap_score(premise, hyp)
    nli = nli_entailment_score(premise, hyp, nli_model) if nli_model else 0.0
    score = max(lexical, nli)
    if score >= 0.55: return {'status':'supported','reason':'evidence_support','score':score}
    if score >= 0.25: return {'status':'unverified','reason':'weak_support','score':score}
    return {'status':'unsupported','reason':'not_supported_by_cited_evidence','score':score}
```

## Command
```bash
python -m src.judges.claim_level_grounding_v4 \
  --rationales data/rationales/parsed/current_clean_train_qwen3_4b_v4.parquet \
  --contexts data/processed/ticker_date_evidence_contexts_h1_v4.parquet \
  --output data/judges/claim_grounding_evidence_v4.parquet \
  --metrics outputs/metrics/claim_grounding_evidence_v4.json \
  --status outputs/status/11_EVIDENCE_GROUNDING_JUDGE_V4.status.json \
  --use-nli true
```

## Gates
- missing_evidence_id_rate <= 0.02.
- unknown_evidence_id_rate == 0.
- technical unknown signal rate <= 0.02.
- unsupported news claim rate reported; if >0.30 then mechanics may PASS but claim must be blocked.
