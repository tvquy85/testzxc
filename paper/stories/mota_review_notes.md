# MoTa Review Notes

Created alongside `MoTa.md` as the current review-facing story for the FIRE-Fin AAAI upgrade.

Key decisions captured:

- Primary rationale generator is `Qwen3-4B-Instruct-2507`.
- No unavailable student-generator fallback is used; local-cache availability is required before a model can enter the project.
- `DeepSeek-R1-Distill-Qwen-1.5B` is not positioned as the main generator.
- The paper story should emphasize gated, train-only rationale alignment, strict schema, claim-level grounding, Flow Reward v2, and finance-valid evaluation.
- Acceptance claims should stay conditional on Step14-21 evidence, especially real RWSFT/DPO data, Flow Reward validation, daily portfolio backtest, baselines, ablations, and statistics.
- Stage-1 now has Step09-15 RWSFT smoke PASS evidence: 12,500 train-only rationales, 7,336 RWSFT examples, 3,362 DPO pairs, and a 10-step QLoRA RWSFT adapter smoke using LOBExp venv plus existing packages from C:/Python/Python311.

Use `MoTa.md` as the human-readable review draft for domain-knowledge feedback.
