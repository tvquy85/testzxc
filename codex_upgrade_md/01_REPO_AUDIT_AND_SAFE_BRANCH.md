# 01 — Repo audit and safe upgrade branch

## Goal

Create a safe branch, audit current code/results, and write a machine-readable baseline report before making changes.

## Files to inspect

```text
TongHop.md
configs/local_paths.yaml
src/llm/parse_and_validate_rationale.py
src/judges/inferability_judge.py
src/judges/technical_grounding_judge.py
src/eval/backtest_long_short_hold.py
src/eval/evaluate_counterfactual_consistency.py
src/reward/flow_model_lite.py
src/reward/flow_dataset.py
src/eval/build_paper_tables.py
outputs/status/*.json
outputs/metrics/*.json
outputs/tables/*.csv
```

## Codex task

1. Create a new branch:

```bash
git checkout -b upgrade-aaai-reproducibility
```

2. Create `src/utils/audit_repo_state.py`.
3. The script must:
   - list all tracked source files under `src/`, `configs/`, `prompts/`;
   - compute SHA256 for each file;
   - check whether key outputs exist;
   - parse `TongHop.md` if present;
   - flag known risk patterns:
     - hard-coded Windows paths like `e:/`;
     - dummy tables;
     - auto-fixing forecast distributions;
     - status `PASS` with missing outputs;
     - parse fallback to neutral distribution;
     - backtest using per-news returns instead of daily portfolio returns.

4. Create `src/utils/verify_status.py` to validate status JSON files.

## Output files

```text
outputs/audit/repo_audit_initial.json
outputs/audit/repo_file_hashes_initial.csv
outputs/status/01_REPO_AUDIT_AND_SAFE_BRANCH.status.json
```

## Verification commands

```bash
python src/utils/audit_repo_state.py --output outputs/audit/repo_audit_initial.json
python -m src.utils.verify_status --status outputs/status/01_REPO_AUDIT_AND_SAFE_BRANCH.status.json
```

## Acceptance criteria

- `repo_audit_initial.json` exists and includes `risk_flags`.
- File hash CSV exists and has at least the columns `path,sha256,size_bytes`.
- Audit must not modify any existing experiment outputs.
- Status is `PASS` only if audit script runs without exception.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "01_REPO_AUDIT_AND_SAFE_BRANCH",
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
outputs/status/01_REPO_AUDIT_AND_SAFE_BRANCH.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
