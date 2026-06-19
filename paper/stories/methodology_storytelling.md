# Evaluation Methodology Storytelling

## 1. The Rationale for a Hybrid Judge Pipeline
When evaluating the quality of financial rationales, purely relying on generative Large Language Models (LLMs) poses risks of hallucination and lacks strict objective verification. To address this, our methodology employs a **Hybrid Judge Pipeline** that combines the nuanced semantic understanding of LLMs with deterministic, rule-based heuristics and specialized state-of-the-art models.
- **NLI Grounding Judge (`DeBERTa-v3-small` Cross-Encoder):** Verifying whether the generated rationale contradicts the original financial news headline requires highly precise textual entailment capabilities. We use a dedicated, fine-tuned Cross-Encoder model rather than a general-purpose LLM to ensure SOTA accuracy in detecting contradictions, establishing a strong grounding penalty.
- **Technical Grounding & Utility:** We apply deterministic exact-match evaluations to ensure technical indicator tokens are correctly woven into the rationale. Additionally, the utility judge utilizes the ground-truth abnormal returns (`abnormal_return_h1`), anchoring the evaluation completely in objective reality rather than simulated payoffs.

## 2. Leveraging Open-Weights for LLM-as-a-Judge
Following the foundational study on "Judging LLM-as-a-Judge" (Zheng et al., 2023), it is well established that open-weights models with strong instruction-following capabilities can serve as highly reliable evaluators when provided with a strict rubric.
- We utilize **`Meta-Llama-3-8B-Instruct`** to judge the *Inferability* and *Financial Soundness* of the generated rationales.
- Despite having only 8 billion parameters, Llama-3-8B exhibits semantic and reasoning capabilities competitive with much larger models on constrained zero-shot extraction tasks. By imposing a strict JSON schema and explicit criteria (e.g., extracting probabilistic distributions and scoring financial logic), the 8B model delivers precise, nuanced scoring distributions without the overhead of massive parameter counts.

## 3. The Imperative of 100% Reproducibility and Data Privacy
A critical flaw in many contemporary NLP and Finance evaluation pipelines is the reliance on proprietary, closed-source APIs (such as OpenAI's GPT-4).
- **Data Drift:** Closed APIs undergo silent updates, causing evaluation scores to drift over time. This creates a severe reproducibility crisis in ML research, as independent researchers cannot replicate the evaluation exactness months later.
- **Strict Determinism:** By hosting `Llama-3-8B-Instruct` and `DeBERTa-v3-small` locally and inferencing with `temperature=0.0`, our evaluation pipeline is **100% deterministic**. Any researcher downloading our codebase and model weights will achieve identical proxy scores.
- **Financial Security Standard:** Financial text processing often involves sensitive or proprietary data. A 100% offline, local evaluation pipeline establishes an industry-standard methodology for privacy-preserving algorithmic trading research, avoiding data transmission to external commercial servers.

## Verdict
The combination of `Meta-Llama-3-8B-Instruct`, `DeBERTa-v3-small`, and deterministic utility scoring creates a robust, highly reproducible, and scientifically rigorous evaluation standard that completely avoids the black-box nature of commercial LLM APIs.

## 4. Weak-to-Strong Alignment Paradigm: DeepSeek-R1-Distill as Generator

While our Hybrid Judge Pipeline employs a rigorous and computationally intensive evaluator (`Meta-Llama-3-8B-Instruct`), the rationale generation phase (Candidate Generation) intentionally utilizes a much smaller, highly efficient, yet logically capable model: **`DeepSeek-R1-Distill-Qwen-1.5B`**.

- **Extreme Generation Efficiency & Inherent Reasoning:** To achieve statistical significance, our framework demands generating thousands of candidate rationales (e.g., 3 candidates per sample for extensive datasets). At 1.5B parameters, the DeepSeek-R1-Distill model provides lightning-fast batched generation while bringing inherent Chain-of-Thought (CoT) reasoning capabilities. This allows us to build a massive pool of candidate rationales without prohibitive computational bottlenecks.
- **Weak-to-Strong Generalization on Financial Noise:** This design choice is deeply rooted in the "Weak-to-Strong Generalization" paradigm (Burns et al., 2023). Although R1-Distill possesses general logic, it is still "weak" in handling extremely noisy financial contexts and counterfactuals. By generating candidates with this fast model and rigorously scoring/filtering them with a "strong" but slower model (Llama-3-8B), we distill the sophisticated financial reasoning and rubric-following capabilities of the 8B Judge down into the 1.5B model's alignment dataset. This effectively teaches a small, deployable reasoning engine to mimic the rigor of a much larger evaluator specifically for noisy financial events.
- **Multi-Model Robustness (Llama-3.2-3B Baseline):** To prove our framework is model-agnostic and does not solely rely on DeepSeek-R1's pre-trained logic, we also validate our Regime-Conditioned Flow Reward on `Llama-3.2-3B-Instruct`. The consistent performance gains across both model families confirm the general applicability of our alignment approach.
