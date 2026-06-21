# 08 — Render Evidence Context V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Add a renderer so Qwen receives clean evidence IDs rather than noisy raw body concatenation.

## File to add/modify
- Prefer adding `src/llm/render_context_evidence_v4.py`.
- Modify `src/llm/generate_rationales.py` only to optionally call evidence renderer when input has `evidence_pack_json`.

## Output sample
- `outputs/data_samples/rendered_context_evidence_v4.txt`
- `outputs/status/08_RENDER_CONTEXT_EVIDENCE_V4.status.json`

## Required render format
```text
Ticker: AAPL
Date: 2021-05-03

Company-specific evidence:
[N1] type=earnings_or_guidance quality=0.86
Headline: ...
Excerpt: ...

Context-only evidence:
[M1] type=macro_market quality=0.51
Headline: ...
Excerpt: ...

Technical signals:
[T1] token=MACD_BEARISH direction=negative strength=strong
```

## Verification command
```bash
python -m src.llm.render_context_evidence_v4 \
  --input data/processed/ticker_date_evidence_contexts_h1_v4.parquet \
  --num-samples 10 \
  --output outputs/data_samples/rendered_context_evidence_v4.txt \
  --status outputs/status/08_RENDER_CONTEXT_EVIDENCE_V4.status.json
```

## Acceptance criteria
- Sample file contains evidence IDs or explicitly says company evidence is None.
- Average rendered context <= 3500 chars.
- Technical tokens are rendered with `T1/T2/...` IDs.
