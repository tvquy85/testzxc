# 09 — Evidence-ID Rationale Prompt V4

> Scope: Upgrade `tvquy85/testzxc` branch `currentdata-aaai-fix-v2` on the already-built current dataset. Do not use SN2. Do not expand to full FNSPID. Do not overwrite v3 artifacts; all new artifacts must use `_v4` or `current_clean_v4`.
> Codex rule: implement only the task in this file, run verification, write status JSON, and never PASS if required outputs are missing or empty.

## Goal
Create a strict prompt that forces evidence citations and prevents invented news claims.

## File to create
`prompts/rationale_generation_prompt_evidence_v4.txt`

## Required prompt
```text
You are a financial reasoning model.

Task:
Given a ticker-date evidence pack and technical signals, generate a concise JSON rationale for short-horizon stock movement.

Rules:
1. Return valid JSON only. No markdown.
2. Every news_rationale item must cite an evidence_id from the evidence section.
3. If Company-specific evidence is None, set news_rationale to [] and mention uncertainty in risk_note.
4. Every technical_rationale item must cite a signal_id such as T1, T2, T3.
5. Do not invent events, analyst actions, earnings, guidance, volume, RSI, MACD, or macro information not shown.
6. Do not explicitly reveal the realized label.
7. Use at most 2 news_rationale items and at most 2 technical_rationale items.
8. forecast_distribution must sum to 1.0.

Context:
{context}

Return JSON:
{
  "news_rationale": [{"evidence_id":"N1","factor":"...","direction":"positive|negative|neutral","strength":"weak|medium|strong"}],
  "technical_rationale": [{"signal_id":"T1","signal":"...","direction":"positive|negative|neutral","strength":"weak|medium|strong"}],
  "conflict_resolution": "<=35 words",
  "forecast_distribution": {"Strong Down":0.0,"Mild Down":0.0,"Neutral":0.0,"Mild Up":0.0,"Strong Up":0.0},
  "action": "long|short|hold",
  "risk_note": "<=25 words"
}
```

## Verification
Assert the prompt contains `evidence_id`, `signal_id`, `forecast_distribution`, and `Do not invent`.
