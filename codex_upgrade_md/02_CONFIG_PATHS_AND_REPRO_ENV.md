# 02 — Config paths and reproducible environment

## Goal

Remove hard-coded local paths and make the repository reproducible on another machine.

## Current problem

`configs/local_paths.yaml` contains machine-specific paths such as `e:/huggingface/...`. This is unacceptable for a reproducible AAAI package.

## Files to create or modify

```text
configs/default_paths.yaml
configs/local_paths.template.yaml
src/utils/config.py
requirements.lock.txt
environment.yml
```

## Codex task

1. Create `configs/default_paths.yaml` with relative defaults:

```yaml
project_root: .
hf_home: $HF_HOME
data_root: data
output_root: outputs
model_root: $HF_HOME/hub
```

2. Create `configs/local_paths.template.yaml` with placeholders only. Do not include the user's real machine path.
3. Create `src/utils/config.py` with:
   - `load_config(path=None)`;
   - environment variable expansion;
   - path validation;
   - helpful errors for missing models/data.
4. Modify scripts that read `configs/local_paths.yaml` so they also accept:

```bash
--config configs/default_paths.yaml
```

5. Create `requirements.lock.txt` by pinning currently needed libraries. Include at least:

```text
torch
transformers
peft
trl
accelerate
bitsandbytes
pandas
pyarrow
numpy
scikit-learn
lightgbm
sentence-transformers
matplotlib
seaborn
tqdm
pyyaml
pytest
```

Use exact versions installed in the environment if detectable.

## Verification commands

```bash
python - <<'PY'
from src.utils.config import load_config
cfg = load_config("configs/default_paths.yaml")
print(cfg)
PY

grep -R "e:/huggingface\|C:\\Users" -n configs src prompts || true
```

## Acceptance criteria

- No source/config file except `outputs/audit/*` contains hard-coded `e:/huggingface` or `C:\Users`.
- Scripts still run from repo root.
- Missing model paths produce explicit errors, not silent fallback.


## Status JSON contract

When this task is complete, write:

```json
{
  "step": "02_CONFIG_PATHS_AND_REPRO_ENV",
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
outputs/status/02_CONFIG_PATHS_AND_REPRO_ENV.status.json
```

`next_step_allowed` must be `false` if any acceptance criterion fails.
