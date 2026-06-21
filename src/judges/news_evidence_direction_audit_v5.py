from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.judges.claim_level_grounding_v4 import load_nli, score_row, safe_rate
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "09_GROUNDING_NEWS_NEGATIVE_FIX"
NEGATIVE_WORDS = {"negative", "downgrade", "weak", "lower", "decline", "miss", "missed", "loss", "bearish", "down"}


def claim_details(value: Any) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_negative_news_claim(item: dict[str, Any]) -> bool:
    if item.get("claim_type") != "news":
        return False
    text = " ".join(str(item.get(key, "")).lower() for key in ["direction", "factor", "strength"])
    return any(word in text for word in NEGATIVE_WORDS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", default="outputs/judges/medium_clean_v4_claim_grounding.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/09_grounding_news_negative_fix.json")
    parser.add_argument("--failures-output", default="review_samples/medium_clean_v4/news_negative_evidence_failures.jsonl")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--hf-home", default="E:/huggingface")
    parser.add_argument("--nli-model-id", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--nli-loader", default="auto", choices=["auto", "transformers", "sentence_transformers"])
    parser.add_argument("--require-nli", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if Path(args.rationales).exists() else pd.DataFrame()
    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    if rationales.empty:
        failures.append(f"rationales missing or empty: {args.rationales}")
    if contexts.empty:
        failures.append(f"contexts missing or empty: {args.contexts}")
    nli, nli_info = load_nli(args.nli_model_id, args.hf_home, True, args.nli_loader)
    if args.require_nli and nli is None:
        failures.append("required NLI backend unavailable")

    cols = [c for c in ["sample_id", "target_label_5", "target_return", "split", "track", "news_reasoning_track", "evidence_pack_json", "technical_event_tokens_json", "mean_evidence_quality_score"] if c in contexts.columns]
    merged = rationales.merge(contexts[cols], on="sample_id", how="inner", suffixes=("", "_context")) if not rationales.empty and not contexts.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    negative_news_claim_count = 0
    negative_news_supported = 0
    for _, row in merged.iterrows():
        scored = score_row(row, nli)
        details = claim_details(scored["claim_details_json"])
        for item in details:
            if is_negative_news_claim(item):
                negative_news_claim_count += 1
                negative_news_supported += int(item.get("status") == "supported")
            if item.get("claim_type") == "news" and item.get("status") in {"unsupported", "unverified", "contradiction"}:
                bad.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "candidate_id": int(row.get("candidate_id", 0)),
                        "status": item.get("status"),
                        "reason": item.get("reason"),
                        "claim": {k: item.get(k) for k in ["evidence_id", "factor", "direction", "strength"]},
                    }
                )
        rows.append({"sample_id": row.get("sample_id"), "candidate_id": int(row.get("candidate_id", 0)), "split": row.get("split"), "track": row.get("track"), **scored})

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    write_jsonl(args.failures_output, bad[:200])
    news_claims = int(out["news_claim_count"].sum()) if len(out) else 0
    unsupported_news = int(out["unsupported_news_claim_count"].sum()) if len(out) else 0
    metrics = {
        "pipeline_pass": True,
        "claim_allowed": False,
        "rows": int(len(out)),
        "total_claims": int(out["total_claims"].sum()) if len(out) else 0,
        "news_claims": news_claims,
        "technical_claims": int(out["technical_claim_count"].sum()) if len(out) else 0,
        "negative_news_claim_count": int(negative_news_claim_count),
        "negative_news_supported_rate": safe_rate(negative_news_supported, negative_news_claim_count),
        "unsupported_news_claim_rate": safe_rate(unsupported_news, news_claims),
        "row_status_counts": out["status"].value_counts(dropna=False).to_dict() if len(out) else {},
        "bad_examples_saved": min(200, len(bad)),
        **nli_info,
    }
    if len(out) == 0:
        failures.append("grounding output is empty")
    if metrics["unsupported_news_claim_rate"] > 0.30:
        failures.append(f"unsupported_news_claim_rate {metrics['unsupported_news_claim_rate']:.4f} > 0.30")
    if negative_news_claim_count <= 0:
        failures.append("negative_news_claim_count == 0")

    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics, args.failures_output], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.rationales, args.contexts], [args.output, args.metrics, args.failures_output, args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
