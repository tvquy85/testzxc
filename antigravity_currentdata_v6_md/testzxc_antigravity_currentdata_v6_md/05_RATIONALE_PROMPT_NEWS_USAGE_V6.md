# 05 — Update Rationale Prompt to Force News Usage When Valid Evidence Exists

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Prevent technical-template rationales. If a context has valid `N*` company evidence, the model must use at least one news rationale item.

## Outputs
```text
prompts/rationale_generation_prompt_evidence_v6.txt
src/llm/parse_and_validate_rationale_v6.py
outputs/metrics/05_v6_prompt_validation.json
outputs/status/05_RATIONALE_PROMPT_NEWS_USAGE_V6.status.json
```

## Prompt rules to add exactly
```text
If Company-specific evidence contains N1/N2/N3, news_rationale must contain at least one item citing one of those IDs.
Do not return an empty news_rationale when company-specific evidence exists.
If you think company evidence is weak, cite it with direction="neutral" and explain uncertainty in risk_note.
technical_rationale may not be the only rationale section when N* evidence exists.
Return JSON only.
```

## Parser validation
```python
if has_company_evidence and len(news_rationale) == 0:
    errors.append('news_rationale empty despite company evidence')
```

## Test case
```python
from src.llm.parse_and_validate_rationale_v6 import validate_rationale_v6

def test_news_required_when_company_evidence():
    parsed={'news_rationale':[],'technical_rationale':[{'signal_id':'T1','signal':'MACD bearish','direction':'negative','strength':'medium'}],'conflict_resolution':'Technicals dominate.','forecast_distribution':{'Strong Down':0.4,'Mild Down':0.3,'Neutral':0.2,'Mild Up':0.1,'Strong Up':0.0},'action':'short','risk_note':'test'}
    errors=validate_rationale_v6(parsed, evidence_ids={'N1'}, signal_ids={'T1'})
    assert any('news_rationale empty' in e for e in errors)
```

## Commands
```bash
python -m src.llm.validate_prompt_v6 --prompt prompts/rationale_generation_prompt_evidence_v6.txt --contexts data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --metrics outputs/metrics/05_v6_prompt_validation.json --status outputs/status/05_RATIONALE_PROMPT_NEWS_USAGE_V6.status.json
python -m pytest -q tests/test_rationale_prompt_news_usage_v6.py tests
```
