# 08 — Strict rationale schema, no auto-fix in main metrics

## Goal

Replace auto-fixing parser behavior with strict raw-output validation.

## Current problem

`src/llm/parse_and_validate_rationale.py` auto-fixes missing distributions and overwrites probabilities based on action. This contaminates forecast-distribution and calibration metrics.

## Files to modify

```text
src/llm/parse_and_validate_rationale.py
src/llm/rationale_schema.py
```

## Codex task

1. Preserve the old function name only if needed for backward compatibility.
2. Add strict functions:

```python
parse_llm_json_strict(text) -> dict | None
validate_rationale_schema_strict(data) -> tuple[bool, list[str]]
normalize_distribution_if_valid(data) -> dict
```

3. Do not overwrite forecast probabilities based on action.
4. Do not fill `"No rationale provided"` in main parsed outputs.
5. Save two outputs:
   - raw generation text;
   - strict parsed JSON or parse error.
6. If a repair mode is necessary, it must be saved under:

```text
data/rationales/repaired/
```

and never used for main metrics.

## Verification commands

```bash
pytest -q tests/test_rationale_schema_strict.py
python - <<'PY'
from src.llm.parse_and_validate_rationale import parse_llm_json_strict
bad='not json'
assert parse_llm_json_strict(bad) is None
PY
```

## Acceptance criteria

- Invalid JSON remains invalid.
- Probabilities are not replaced by long/short/hold templates.
- Main metrics can report raw JSON-valid rate separately.
- Existing downstream scripts are updated to consume strict parsed files.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "08_STRICT_RATIONALE_SCHEMA_NO_AUTOFIX",
  "status": "PASS|FAIL",
  "inputs_checked": [],
  "outputs_created": [],
  "metrics": {},
  "failures": [],
  "next_step_allowed": true
}
```

Save it to:

```text
outputs/status/08_STRICT_RATIONALE_SCHEMA_NO_AUTOFIX.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
