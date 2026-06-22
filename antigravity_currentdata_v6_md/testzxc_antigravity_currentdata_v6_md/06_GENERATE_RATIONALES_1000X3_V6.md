# 06 — Generate V6 Rationales on 1000×3 Current-Data Candidates

> Scope: branch `currentdata-aaai-fix-v2`; current-data only; do not use SN2; do not expand full FNSPID. Execute one file at a time. Every step must write a status JSON and must stop on FAIL. Pipeline PASS is not a scientific claim.

## Goal
Generate enough clean current-data rationales to improve judge, flow, and alignment. Target 1000 unique sample IDs × 3 candidates.

## Outputs
```text
data/rationales/raw/current_v6_train_qwen3_1000x3.jsonl
data/rationales/parsed/current_v6_train_qwen3_1000x3.parquet
outputs/metrics/06_v6_generate_rationales_1000x3.json
outputs/status/06_GENERATE_RATIONALES_1000X3_V6.status.json
```

## Command
```bash
python -m src.llm.generate_rationales --input data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet --prompt prompts/rationale_generation_prompt_evidence_v6.txt --config configs/local_paths.yaml --split train --model-key main_explanation_llm --limit 1000 --num-candidates 3 --max-new-tokens 280 --max-input-tokens 3072 --temperature 0.45 --top-p 0.85 --top-k 40 --batch-size 4 --sort-by-length --save-every 50 --resume --stage current_v6_1000x3 --raw-output data/rationales/raw/current_v6_train_qwen3_1000x3.jsonl --parsed-output data/rationales/parsed/current_v6_train_qwen3_1000x3.parquet --status outputs/status/06_GENERATE_RATIONALES_1000X3_V6.status.json
```

## Test case
```python
import pandas as pd
from pathlib import Path

def test_v6_rationale_counts_and_schema():
    p=Path('data/rationales/parsed/current_v6_train_qwen3_1000x3.parquet')
    assert p.exists()
    df=pd.read_parquet(p)
    assert df['sample_id'].nunique()>=900
    assert len(df)>=2700
    assert df['schema_ok'].mean()>=0.95
```

## Acceptance
Unique IDs >=900, rows >=2700, parse/schema >=0.95, empty news rationale with N evidence <=0.05.
