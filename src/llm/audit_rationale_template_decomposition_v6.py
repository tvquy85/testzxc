from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataclean_v4_utils import clean_string, write_json
from src.llm.audit_rationale_diversity_v6 import jaccard, parse_json_obj
from src.utils.artifacts import write_manifest, write_status

STEP = "07_5_RATIONALE_TEMPLATE_DECOMPOSITION_V6"


def rationale_items(parsed: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items = parsed.get(key, [])
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def phrase_list(parsed: dict[str, Any], kind: str) -> list[str]:
    items: list[dict[str, Any]] = []
    if kind in {"news", "all"}:
        items.extend(rationale_items(parsed, "news_rationale"))
    if kind in {"technical", "all"}:
        items.extend(rationale_items(parsed, "technical_rationale"))
    phrases: list[str] = []
    for item in items:
        phrase = clean_string(item.get("factor", item.get("signal"))).lower()
        if phrase:
            phrases.append(phrase)
    return phrases


def repeated_phrase_rate(phrases: list[str], row_count: int) -> tuple[float, list[tuple[str, int]], int]:
    counts = Counter(phrases)
    threshold = max(5, row_count * 0.05)
    repeated = sum(count for count in counts.values() if count >= threshold)
    total = sum(counts.values())
    return float(repeated / max(1, total)), counts.most_common(12), int(len(counts))


def news_plus_meta_text(parsed: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in rationale_items(parsed, "news_rationale"):
        parts.append(" ".join(clean_string(item.get(key)) for key in ["evidence_id", "factor", "direction", "strength"]))
    parts.append(clean_string(parsed.get("conflict_resolution")))
    parts.append(clean_string(parsed.get("risk_note")))
    return " ".join(part for part in parts if part)


def technical_text(parsed: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in rationale_items(parsed, "technical_rationale"):
        parts.append(" ".join(clean_string(item.get(key)) for key in ["signal_id", "signal", "direction", "strength"]))
    return " ".join(part for part in parts if part)


def grouped_jaccard_metrics(df: pd.DataFrame, texts: list[str]) -> tuple[float, float, list[dict[str, Any]]]:
    tmp = df[["sample_id", "candidate_id"]].copy()
    tmp["_text"] = texts
    scores: list[float] = []
    bad: list[dict[str, Any]] = []
    for sample_id, group in tmp.groupby("sample_id"):
        group_texts = group["_text"].tolist()
        pair_scores: list[float] = []
        for i in range(len(group_texts)):
            for j in range(i + 1, len(group_texts)):
                score = jaccard(group_texts[i], group_texts[j])
                scores.append(score)
                pair_scores.append(score)
        if pair_scores and max(pair_scores) > 0.82:
            bad.append(
                {
                    "sample_id": str(sample_id),
                    "max_pair_jaccard": float(max(pair_scores)),
                    "candidate_ids": group["candidate_id"].tolist(),
                    "texts": group_texts[:3],
                }
            )
    mean = float(sum(scores) / max(1, len(scores)))
    cluster_rate = float(sum(score > 0.82 for score in scores) / max(1, len(scores)))
    return mean, cluster_rate, bad


def evidence_usage_metrics(df: pd.DataFrame, parsed_rows: list[dict[str, Any]]) -> dict[str, float]:
    news_citations = 0
    news_claims = 0
    technical_only = 0
    empty_news_when_company_news = 0
    company_news_possible = 0
    for idx, parsed in enumerate(parsed_rows):
        news = rationale_items(parsed, "news_rationale")
        tech = rationale_items(parsed, "technical_rationale")
        context_meta = parse_json_obj(df.iloc[idx].get("context_meta_json", "{}"))
        company_evidence = context_meta.get("company_evidence_ids", [])
        has_company_news = any(str(eid).startswith("N") for eid in company_evidence)
        if has_company_news:
            company_news_possible += 1
            if not news:
                empty_news_when_company_news += 1
        if news:
            news_claims += len(news)
            news_citations += sum(1 for claim in news if clean_string(claim.get("evidence_id")))
        if tech and not news:
            technical_only += 1
    return {
        "technical_only_phrase_rate": float(technical_only / max(1, len(df))),
        "news_rationale_empty_when_N_rate": float(empty_news_when_company_news / max(1, company_news_possible)),
        "evidence_citation_rate": float(news_citations / max(1, news_claims)),
        "company_context_news_usage_rate": float(news_claims / max(1, company_news_possible)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metrics", default="outputs/metrics/07_5_v6_rationale_template_decomposition.json")
    parser.add_argument("--table", default="outputs/tables/07_5_v6_rationale_template_decomposition.csv")
    parser.add_argument("--samples", default="review_samples/currentdata_v6/07_5_rationale_decomposition_examples.jsonl")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    df = pd.read_parquet(args.input) if Path(args.input).exists() else pd.DataFrame()
    if df.empty:
        failures.append(f"rationale input missing or empty: {args.input}")
    if not failures and "parsed_json" not in df.columns:
        failures.append("rationale input missing parsed_json")
    if not failures and "sample_id" not in df.columns:
        failures.append("rationale input missing sample_id")
    if not failures and "candidate_id" not in df.columns:
        failures.append("rationale input missing candidate_id")

    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    table = pd.DataFrame()
    bad_examples: list[dict[str, Any]] = []
    if not failures:
        parsed_rows = [parse_json_obj(value) for value in df["parsed_json"]]
        news_texts = [news_plus_meta_text(parsed) for parsed in parsed_rows]
        tech_texts = [technical_text(parsed) for parsed in parsed_rows]
        news_mean_jaccard, news_cluster_rate, news_bad = grouped_jaccard_metrics(df, news_texts)
        tech_mean_jaccard, tech_cluster_rate, tech_bad = grouped_jaccard_metrics(df, tech_texts)
        phrase_metrics: dict[str, Any] = {}
        table_rows: list[dict[str, Any]] = []
        for kind in ["news", "technical", "all"]:
            phrases: list[str] = []
            for parsed in parsed_rows:
                phrases.extend(phrase_list(parsed, kind))
            rate, top, unique_count = repeated_phrase_rate(phrases, len(df))
            phrase_metrics[f"{kind}_repeated_phrase_rate"] = rate
            phrase_metrics[f"{kind}_unique_phrase_count"] = unique_count
            table_rows.extend(
                {"component": kind, "phrase": phrase, "count": count}
                for phrase, count in top
            )
        usage = evidence_usage_metrics(df, parsed_rows)
        claim_allowed = bool(
            usage["news_rationale_empty_when_N_rate"] <= 0.05
            and usage["evidence_citation_rate"] >= 0.95
            and usage["technical_only_phrase_rate"] <= 0.15
            and news_mean_jaccard <= 0.68
            and news_cluster_rate <= 0.25
            and phrase_metrics["news_repeated_phrase_rate"] <= 0.05
        )
        metrics = {
            "pipeline_pass": True,
            "claim_allowed": claim_allowed,
            "rows": int(len(df)),
            "unique_samples": int(df["sample_id"].nunique()),
            "mean_candidates_per_sample": float(len(df) / max(1, df["sample_id"].nunique())),
            "news_plus_meta_mean_jaccard": news_mean_jaccard,
            "news_plus_meta_template_cluster_rate": news_cluster_rate,
            "technical_mean_jaccard": tech_mean_jaccard,
            "technical_template_cluster_rate": tech_cluster_rate,
            **phrase_metrics,
            **usage,
            "technical_repetition_explains_overall_repetition": bool(
                phrase_metrics["technical_repeated_phrase_rate"] > 0.5
                and phrase_metrics["news_repeated_phrase_rate"] <= 0.05
            ),
            "claim_boundary": (
                "Rationale quality allowed only for non-technical/news/meta wording; "
                "technical signal-name repetition is reported separately as feature vocabulary repetition."
            ),
        }
        table = pd.DataFrame(table_rows)
        bad_examples = news_bad[:25] + [{"technical_example": item} for item in tech_bad[:25]]

    Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table, index=False)
    Path(args.samples).parent.mkdir(parents=True, exist_ok=True)
    with open(args.samples, "w", encoding="utf-8") as f:
        for row in bad_examples[:50]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.input, args.metrics, args.table, args.samples], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.input],
        [args.metrics, args.table, args.samples, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
