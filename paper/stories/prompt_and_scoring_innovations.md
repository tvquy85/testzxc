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
