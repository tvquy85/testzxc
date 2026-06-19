from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.judges.claim_level_grounding import score_news_claim, score_technical_claim
from src.judges.extract_rationale_claims import extract_claims
from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "11_CLAIM_LEVEL_GROUNDING_JUDGES"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", default="data/rationales/parsed/train_candidates_strict.parquet")
    parser.add_argument("--samples", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--tokens", default="data/indicators/technical_event_tokens_h1_v2.parquet")
    parser.add_argument("--output", default="data/judges/claim_grounding_scores.parquet")
    parser.add_argument("--summary", default="outputs/metrics/claim_grounding_summary.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    failures: list[str] = []
    if not Path(args.rationales).exists():
        failures.append(f"rationales missing: {args.rationales}")
        rationales = pd.DataFrame()
    else:
        rationales = pd.read_parquet(args.rationales)
    samples = pd.read_parquet(args.samples) if Path(args.samples).exists() else pd.DataFrame()
    tokens = pd.read_parquet(args.tokens) if Path(args.tokens).exists() else pd.DataFrame()
    if rationales.empty:
        failures.append("rationales input is empty")

    rows = []
    if not rationales.empty:
        df = rationales
        if not samples.empty:
            df = df.merge(samples[["sample_id", "headline", "body"]], on="sample_id", how="left")
        if not tokens.empty:
            df = df.merge(tokens[["sample_id", "technical_event_tokens_json"]], on="sample_id", how="left")
        for _, row in df.iterrows():
            for idx, claim in enumerate(extract_claims(row.get("parsed_json"))):
                if claim["claim_type"] == "technical_claim":
                    score = score_technical_claim(claim["claim_text"], row.get("technical_event_tokens_json"))
                    news_score = None
                    tech_score = score.get("score")
                    status_text = score.get("status")
                elif claim["claim_type"] == "news_claim":
                    score = score_news_claim(claim["claim_text"], row.get("headline", ""), row.get("body", ""))
                    news_score = score.get("score")
                    tech_score = None
                    status_text = score.get("status")
                else:
                    news_score = None
                    tech_score = None
                    status_text = "not_applicable"
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "candidate_id": row.get("candidate_id"),
                        "claim_id": idx,
                        "claim_type": claim["claim_type"],
                        "claim_text": claim["claim_text"],
                        "news_grounding_score": news_score,
                        "technical_grounding_score": tech_score,
                        "grounding_status": status_text,
                    }
                )
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    summary = {
        "rows": int(len(out)),
        "claim_type_counts": out["claim_type"].value_counts().to_dict() if len(out) else {},
        "status_counts": out["grounding_status"].value_counts().to_dict() if len(out) else {},
        "technical_contradiction_examples_saved": False,
        "news_contradiction_examples_saved": False,
    }
    if len(out) == 0:
        failures.append("no claim rows produced")
    write_json(args.summary, summary)
    write_manifest(args.manifest, [args.output, args.summary], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.rationales, args.samples, args.tokens],
        outputs_created=[args.output, args.summary, args.manifest, args.status],
        metrics=summary,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

