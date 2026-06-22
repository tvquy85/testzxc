from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataclean_v4_utils import clean_string, write_json
from src.utils.artifacts import write_manifest, write_status

STEP = "07_RATIONALE_DIVERSITY_AND_TEMPLATE_GATE"


def parse_json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(clean_string(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower()))


def jaccard(a: str, b: str) -> float:
    left, right = tokens(a), tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def compact_rationale(parsed: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in parsed.get("news_rationale", []) if isinstance(parsed.get("news_rationale"), list) else []:
        if isinstance(item, dict):
            parts.append(" ".join(clean_string(item.get(key)) for key in ["evidence_id", "factor", "direction", "strength"]))
    for item in parsed.get("technical_rationale", []) if isinstance(parsed.get("technical_rationale"), list) else []:
        if isinstance(item, dict):
            parts.append(" ".join(clean_string(item.get(key)) for key in ["signal_id", "signal", "direction", "strength"]))
    parts.append(clean_string(parsed.get("conflict_resolution")))
    parts.append(clean_string(parsed.get("risk_note")))
    return " ".join(parts)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metrics", default="outputs/metrics/07_v6_rationale_diversity.json")
    parser.add_argument("--clusters", default="outputs/tables/07_v6_template_clusters.csv")
    parser.add_argument("--samples", default="review_samples/currentdata_v6/07_template_heavy_examples.jsonl")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    df = pd.read_parquet(args.input) if Path(args.input).exists() else pd.DataFrame()
    if df.empty:
        failures.append(f"rationales missing or empty: {args.input}")
        
    parsed = [parse_json_obj(value) for value in df.get("parsed_json", [])]
    compact = [compact_rationale(item) for item in parsed]
    df["_compact"] = compact
    
    news_citations = 0
    news_claims = 0
    company_news_claims_possible = 0
    technical_only_phrases = 0
    news_rationale_empty_when_N = 0
    phrase_counter: Counter[str] = Counter()
    
    for idx, item in enumerate(parsed):
        news = item.get("news_rationale", []) if isinstance(item.get("news_rationale"), list) else []
        tech = item.get("technical_rationale", []) if isinstance(item.get("technical_rationale"), list) else []
        
        context_meta = parse_json_obj(df.iloc[idx].get("context_meta_json", "{}"))
        company_evidence = context_meta.get("company_evidence_ids", [])
        has_N = any(eid.startswith("N") for eid in company_evidence)
        
        if has_N:
            company_news_claims_possible += 1
            if not news:
                news_rationale_empty_when_N += 1
                
        if news:
            news_claims += len(news)
            for claim in news:
                if isinstance(claim, dict) and clean_string(claim.get("evidence_id")):
                    news_citations += 1
        if tech and not news:
            technical_only_phrases += 1
            
        for claim in [*news, *tech]:
            if isinstance(claim, dict):
                phrase = clean_string(claim.get("factor", claim.get("signal"))).lower()
                if phrase:
                    phrase_counter[phrase] += 1

    within_jaccards: list[float] = []
    bad_samples: list[dict[str, Any]] = []
    
    for sample_id, group in df.groupby("sample_id"):
        texts = group["_compact"].tolist()
        pair_scores = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                score = jaccard(texts[i], texts[j])
                pair_scores.append(score)
                within_jaccards.append(score)
        if pair_scores and max(pair_scores) > 0.82:
            bad_samples.append(
                {
                    "sample_id": str(sample_id),
                    "max_pair_jaccard": max(pair_scores),
                    "candidate_ids": group.get("candidate_id", pd.Series([], dtype=int)).tolist(),
                    "rationales": texts[:3],
                }
            )

    repeated_phrases = sum(count for count in phrase_counter.values() if count >= max(5, len(df) * 0.05))
    total_phrases = sum(phrase_counter.values())
    
    metrics = {
        "pipeline_pass": True,
        "claim_allowed": False,
        "rows": int(len(df)),
        "unique_samples": int(df["sample_id"].nunique()) if len(df) else 0,
        "mean_candidates_per_sample": float(len(df) / max(1, df["sample_id"].nunique())) if len(df) else 0.0,
        "mean_within_sample_jaccard": float(sum(within_jaccards) / max(1, len(within_jaccards))),
        "repeated_template_cluster_rate": float(sum(score > 0.82 for score in within_jaccards) / max(1, len(within_jaccards))),
        "technical_only_phrase_rate": float(technical_only_phrases / max(1, len(df))),
        "news_rationale_empty_when_N_rate": float(news_rationale_empty_when_N / max(1, company_news_claims_possible)),
        "evidence_citation_rate": float(news_citations / max(1, news_claims)),
        "company_context_news_usage_rate": float(news_claims / max(1, company_news_claims_possible)),
        "top_repeated_phrases": phrase_counter.most_common(12),
        "repeated_phrase_rate": float(repeated_phrases / max(1, total_phrases)),
        "semantic_diversity_score": 1.0 - (sum(within_jaccards) / max(1, len(within_jaccards))),
        "bad_sample_count": len(bad_samples),
    }
    
    if metrics["news_rationale_empty_when_N_rate"] > 0.05:
        failures.append(f"news_rationale_empty_when_N_rate {metrics['news_rationale_empty_when_N_rate']:.4f} > 0.05")
    
    if company_news_claims_possible and metrics["evidence_citation_rate"] < 0.70:
        failures.append(f"evidence_citation_rate {metrics['evidence_citation_rate']:.4f} < 0.70")
    if metrics["repeated_template_cluster_rate"] > 0.70:
        failures.append(f"repeated_template_cluster_rate {metrics['repeated_template_cluster_rate']:.4f} > 0.70")

    write_json(args.metrics, metrics)
    write_jsonl(args.samples, bad_samples[:50])
    
    # Save clusters to CSV
    Path(args.clusters).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bad_samples).to_csv(args.clusters, index=False)
    
    write_manifest(args.manifest, [args.metrics, args.samples, args.clusters], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.input], [args.metrics, args.samples, args.clusters, args.manifest, args.status], metrics, failures, status == "PASS")
    
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
