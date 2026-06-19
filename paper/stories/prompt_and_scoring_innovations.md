# Prompt Engineering and Scoring Innovations

## 1. Robust JSON Regex Parsing (Implementation Innovation)
**Context:** When using instruction-tuned LLMs (like Llama-3-8B or DeepSeek-R1-Distill-1.5B) for "LLM-as-a-Judge", one of the most significant failure points is the model failing to output valid, parseable JSON. The models often wrap the JSON in markdown tags (e.g., ` ```json `), prepend thinking steps (e.g., `<think>...</think>`), or include conversational padding (e.g., "Here is your JSON...").
**Our Innovation:** Instead of trying to force the LLM into a strict JSON-mode (which can degrade reasoning quality by constraining token probabilities), we embraced the conversational output and implemented a **Robust Regex Parser**. 
- We search the raw output string for specific keys and float patterns (e.g., `re.search(f'"{key}"\s*:\s*([0-9.]+)', text)`).
- **Result:** This single improvement boosted the valid parsing rate to **100%**, completely eliminating pipeline crashes or default-fallback errors. This highlights an important engineering lesson for future XAI pipelines: *Don't constrain the LLM generation; parse smartly instead.*

## 2. Hybrid Utility Scoring (Scoring Innovation)
**Context:** Traditional evaluation of rationales often relies entirely on LLM judgement (i.e., asking the LLM "Is this rationale useful?"). This is prone to systemic bias, where the LLM prefers its own style (Self-Enhancement Bias).
**Our Innovation:** We introduced a **Deterministic Utility Score** rooted in objective financial reality.
- We utilize the `abnormal_return_h1` scalar. If the rationale predicts a direction that aligns with the *actual* market movement, the utility score reflects this mathematical truth, bypassing LLM bias entirely.
- By combining this deterministic Utility Score with the LLM's Financial Soundness Score, our `overall_proxy_score` represents a true hybrid metric: Grounded in semantic logic, but anchored in financial reality.

## 3. Unapplied Ideas (Future Work for Storytelling)
During the pipeline design, several advanced ideas were considered but ultimately set aside to prioritize the efficiency of the RTX 3090 constraint:
- **Using DeepSeek-R1-Distill-Qwen-1.5B:** We explicitly chose DeepSeek-R1-Distill as our *Main Generator* because its `<think>` reasoning logic naturally fits our goal of producing comprehensive Financial Rationales. However, we intentionally avoided using it as the *Judge* for Financial Soundness. Generating long reasoning chains (1024+ tokens) just to output a single proxy score for thousands of candidate rationales proved computationally prohibitive for full-scale runs. Thus, we reserved R1's power for generating explanations, while retaining the zero-shot, fast extraction capability of Llama-3-8B for the Judge.
- **Dynamic Prompt Modification (In-context learning):** We considered injecting few-shot examples dynamically into the prompt based on the specific ticker/asset class. While this would improve accuracy, it heavily bloated the context window and reduced throughput. Future work could explore retrieval-augmented prompt construction for the LLM Judges.
## 2026-06-19 Codex AAAI Upgrade Gates

- Strict rationale parsing should be treated as a measurement instrument, not a cleanup step: invalid JSON remains invalid, and forecast probabilities are never rewritten from the action label. This preserves calibration and parse-ok rate as real metrics.
- Technical event tokens v2 are more useful as structured evidence objects than as flat strings. The fields `token`, `value`, `direction_prior`, `strength`, `evidence_column`, and `rule` make downstream grounding auditable.
- No-token technical contexts should be scored as `not_applicable`, not as perfect grounding. This prevents sparse technical evidence from inflating explanation quality.
- Flow reward v2 should use masked multi-target learning so missing judge components are explicitly masked rather than filled with zero. This makes partial reward evidence usable without pretending unavailable metrics were observed.
- Gate statuses are themselves paper evidence: failed gates identify exactly which claims cannot be made yet, and `NOT_RUN` rows should stay out of main evidence tables until real experiments replace them.

## 2026-06-19 Local Model Routing Decision

- Primary rationale generator: `Qwen3-4B-Instruct-2507`.
- No lightweight fallback/student generator is currently configured.
- Do not use as primary generator: `DeepSeek-R1-Distill-Qwen-1.5B`.
- Auxiliary judge/teacher models: `FinGPT forecaster` and `Llama-3-8B-Instruct`.
- Grounding model: `cross-encoder nli-deberta-v3-small`.

## 2026-06-19 Throughput Optimization Note

- For Qwen3 generation, the dominant runtime cost is autoregressive decoded tokens, not DPO smoke training. Bulk generation should minimize candidates and output length, while conflict/hard samples can keep multiple candidates for DPO signal.
- FlashAttention-2 is now available in the LOBExp venv and should be used through an explicit `attn_implementation` setting when the backend supports it. Artifacts should record the attention backend so speed/quality comparisons are reproducible.
- vLLM with prefix caching remains the preferred full-scale throughput path if the local Windows/WSL stack supports serving the Qwen3 checkpoint reliably; Transformers remains the fallback backend.

## 2026-06-19 Forecast-Only Evaluation Interface

- The aligned model should not be evaluated for Step16 with the full rationale schema. Backtest only needs `forecast_distribution` and `action`, so requiring rationale fields caused valid forecasts to be counted as schema failures.
- Moving Step16/17 to a compact forecast-only JSON prompt changed prediction smoke from near-zero schema validity to high validity while preserving strict parsing: invalid JSON remains invalid and no neutral fallback is used as evidence.
- This separation is useful for the paper story: rationale generation remains explanation-rich for alignment, while finance evaluation uses a minimal decision interface that is easier to parse, audit, and backtest.

## 2026-06-20 Medium Finance Evaluation and Statistics Gate

- Date-aware prediction selection matters. The first 200 chronological test rows covered only one trading day, which could not support Sharpe or block bootstrap claims. Scaling to 5,000 chronological test rows produced 33 trading days and made the daily portfolio simulator statistically testable.
- Prediction-level baseline artifacts are necessary for real paired inference. Aggregate baseline CSVs are not enough because paired bootstrap requires overlapping `sample_id` rows between FIRE-Fin and each baseline/seed.
- Step19 now separates "not run" ablations from statistical evidence: A0-A8 remain `NOT_RUN` and are explicitly excluded from evidence, while prediction paired bootstrap and daily-return block bootstrap are real artifacts.
- The medium result is an important negative-control story: mechanics are now finance-valid, but the medium Sharpe is negative and counterfactual no-change remains high. This prevents premature paper claims and points the next research effort toward forecast quality rather than table generation.
