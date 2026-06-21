# 10 — Strict Evidence Schema Validation V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Update rationale validation to check evidence IDs and signal IDs. Do not auto-fix invalid outputs.

## Files
- `src/llm/parse_and_validate_rationale_v4.py` or extend existing parser.
- `tests/test_rationale_schema_v4.py`.

## Required function
```python
def validate_rationale_schema_evidence_v4(parsed: dict, context_meta: dict | None = None) -> tuple[bool, list[str]]:
    """Validate JSON schema, forecast distribution, action, evidence_id, and signal_id."""
```

## Required checks
- `news_rationale` items must have `evidence_id`, `factor`, `direction`, `strength`.
- `technical_rationale` items must have `signal_id`, `signal`, `direction`, `strength`.
- Evidence/signal IDs must exist in `context_meta` when provided.
- Forecast distribution has exactly five labels and sums to 1 ± 1e-3.
- `action` in `long|short|hold`.
- Invalid output fails; no repair.

## Tests
Run:
```bash
python -m pytest -q tests/test_rationale_schema_v4.py
```

## Acceptance criteria
New and existing tests pass. Parser does not silently repair forecast distribution.
