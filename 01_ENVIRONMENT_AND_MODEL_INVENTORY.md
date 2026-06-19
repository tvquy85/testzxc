# Step 01 — Environment and Local Model Inventory

> **Use with Antigravity / Gemini Pro 3.1 High**  
> Treat this file as one bounded task. Do not implement later steps unless this file explicitly asks for them.  
> Always create small scripts, run verification, and save a short status JSON before stopping.

## Goal
Create the project folder, install minimum dependencies, detect GPU, detect local Hugging Face cache, and produce a model inventory. Do not download data in this step.

## Inputs
The machine reportedly has:

- RTX 3090.
- Local FNSPID cache: `datasets--Zihan1004--FNSPID`.
- Local models including Qwen, FinGPT, FinBERT, MOMENT, Chronos-Bolt, TTM, NLI cross-encoder.

## Tasks

### 1. Create project structure
Create the `firefin/` layout from `00_README_EXECUTION_ORDER.md`.

### 2. Create Python environment files
Create:

```text
firefin/requirements.txt
firefin/configs/local_paths.yaml
firefin/src/utils/env_report.py
```

`requirements.txt` must include at minimum:

```text
pandas
numpy
pyarrow
scikit-learn
scipy
tqdm
pydantic
pyyaml
python-dateutil
matplotlib
seaborn
plotly
polars
pandas-market-calendars
pandas-ta
transformers>=4.45.0
accelerate
peft
trl
bitsandbytes
sentence-transformers
huggingface_hub
datasets
safetensors
torch
lightgbm
xgboost
statsmodels
```

If `pandas-ta` fails on the platform, fall back to implementing RSI, MACD, Bollinger, ATR manually in Step 05.

### 3. Detect GPU
Implement `src/utils/env_report.py` to print and save:

```json
{
  "python_version": "...",
  "torch_version": "...",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "gpu_vram_gb": 24,
  "transformers_version": "...",
  "hf_home": "..."
}
```

### 4. Detect model cache
Implement `src/utils/model_inventory.py`.

It must search these likely roots:

```text
$HF_HOME/hub
~/.cache/huggingface/hub
./models
../models
D:/models
E:/models
```

It must find directories whose names contain:

```text
Zihan1004--FNSPID
Qwen2.5-3B
Qwen3-4B
DeepSeek-R1-Distill-Qwen-1.5B
FinGPT
ProsusAI--finbert
cross-encoder--nli-deberta-v3-small
AutonLab--MOMENT-1-small
amazon--chronos-bolt-small
ibm-granite--granite-timeseries-ttm-r2
```

Save:

```text
outputs/status/model_inventory.json
```

### 5. Create `configs/local_paths.yaml`
Use auto-detected paths. If a path is missing, set it to `null`, do not guess.

Example:

```yaml
project_root: ./firefin
hf_home: null
models:
  main_explanation_llm: null
  qwen3_judge: null
  deepseek_reasoning_judge: null
  fingpt_forecaster: null
  finbert: null
  nli_cross_encoder: null
  moment_small: null
  chronos_bolt_small: null
  ttm_r2: null
datasets:
  fnspid_cache: null
```

## Verification
Run:

```bash
cd firefin
python src/utils/env_report.py --output outputs/status/env_report.json
python src/utils/model_inventory.py --output outputs/status/model_inventory.json --config configs/local_paths.yaml
```

## Acceptance criteria
PASS only if:

- `outputs/status/env_report.json` exists.
- `outputs/status/model_inventory.json` exists.
- GPU is detected or the report explicitly says CUDA unavailable.
- At least one of `Qwen2.5-3B`, `Qwen3-4B`, or `Llama-3.2-3B` is found.
- `ProsusAI--finbert` is found or marked null.
- `Zihan1004--FNSPID` is found or marked null.

## Status file
Create:

```text
outputs/status/01_ENVIRONMENT_AND_MODEL_INVENTORY.status.json
```

with:

```json
{
  "step": "01_ENVIRONMENT_AND_MODEL_INVENTORY",
  "status": "PASS|FAIL",
  "created_files": [
    "requirements.txt",
    "configs/local_paths.yaml",
    "outputs/status/env_report.json",
    "outputs/status/model_inventory.json"
  ],
  "notes": "..."
}
```
