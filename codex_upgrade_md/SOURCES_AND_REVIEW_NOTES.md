# Sources and reviewer notes used to design the upgrade tasks

## Repo evidence

Starting file:

```text
https://github.com/tvquy85/testzxc/blob/main/TongHop.md
```

Key current risks observed from repo source:

```text
configs/local_paths.yaml
- hard-coded machine paths.

src/llm/parse_and_validate_rationale.py
- auto-fixes forecast distribution based on action.
- fills missing rationale text.

src/judges/inferability_judge.py
- parse failure can default to neutral distribution.

src/judges/technical_grounding_judge.py
- checks whether token names are mentioned, not whether claims are true.

src/eval/evaluate_counterfactual_consistency.py
- computes flip-rate rather than direction-aware counterfactual response.
- passes empty context to proxy judge in the inspected implementation.

src/eval/build_paper_tables.py
- creates dummy zero-filled tables for some metrics.

src/eval/backtest_long_short_hold.py
- current backtest must be upgraded to daily portfolio returns.
```

## External standards motivating the upgrade

1. AAAI submissions require a reproducibility checklist and supplementary materials/code can be used to assess reproducibility.
2. FNSPID is much larger than the prototype currently summarized in `TongHop.md`; an AAAI-grade paper should not overclaim scale.
3. Stock forecasting evaluation should not rely only on sample-level accuracy. It needs robust backtests, transaction costs, and uncertainty/statistical testing.
4. LLM-as-a-judge pipelines need bias controls and parse-failure reporting.
5. Flow reward must be evaluated against proxy averaging; otherwise the flow component is not established as necessary.

## Design philosophy

The upgrade tasks are deliberately short. Each task has:
- exact files to modify;
- exact outputs;
- verification commands;
- acceptance criteria;
- status JSON contract.

This structure is intended for Codex execution without overloading the agent.
