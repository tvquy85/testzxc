from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataclean_v4_utils import clean_string
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "11_EVIDENCE_GROUNDING_JUDGE_V4"
DEFAULT_HF_HOME = "E:/huggingface"
DEFAULT_NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-small"
NLI_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}

STOPWORDS = {
    "about",
    "after",
    "also",
    "from",
    "have",
    "into",
    "near",
    "term",
    "that",
    "the",
    "their",
    "this",
    "with",
    "while",
    "which",
    "will",
    "stock",
    "share",
    "shares",
    "company",
}


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = clean_string(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def tokens_from_text(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", clean_string(text).lower().replace("_", " "))
        if len(token) > 3 and token not in STOPWORDS
    }


def safe_rate(num: int | float, den: int | float) -> float:
    return 0.0 if not den else float(num) / float(den)


def direction_family(value: Any) -> str:
    text = clean_string(value).lower()
    if any(term in text for term in ["positive", "bullish", "up", "oversold"]):
        return "positive"
    if any(term in text for term in ["negative", "bearish", "down", "overbought"]):
        return "negative"
    return "neutral"


class NLIGrounder:
    def __init__(self, model_id: str, hf_home: str):
        os.environ["HF_HOME"] = hf_home
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True).to(self.device)
        self.model.eval()

    def score(self, premise: str, hypothesis: str) -> dict[str, Any]:
        if not premise.strip() or not hypothesis.strip():
            return {"label": "neutral", "entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            probs = self.torch.softmax(self.model(**inputs).logits, dim=1)[0].detach().cpu()
        label = NLI_ID2LABEL.get(int(probs.argmax().item()), "neutral")
        return {
            "label": label,
            "contradiction": float(probs[0]),
            "entailment": float(probs[1]),
            "neutral": float(probs[2]),
        }


class SentenceTransformerNLIGrounder:
    def __init__(self, model_id: str, hf_home: str):
        os.environ["HF_HOME"] = hf_home
        import numpy as np
        from sentence_transformers import CrossEncoder

        self.np = np
        self.model_id = model_id
        self.model = CrossEncoder(model_id, local_files_only=True)

    def score(self, premise: str, hypothesis: str) -> dict[str, Any]:
        if not premise.strip() or not hypothesis.strip():
            return {"label": "neutral", "entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
        try:
            raw = self.model.predict([(premise, hypothesis)], apply_softmax=True)
        except TypeError:
            raw = self.model.predict([(premise, hypothesis)])
        arr = self.np.asarray(raw)
        if arr.ndim == 2:
            arr = arr[0]
        arr = arr.astype(float)
        if arr.size != 3:
            return {"label": "neutral", "entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
        total = float(arr.sum())
        if not 0.99 <= total <= 1.01:
            arr = self.np.exp(arr - arr.max())
            arr = arr / arr.sum()
        label = NLI_ID2LABEL.get(int(arr.argmax()), "neutral")
        return {
            "label": label,
            "contradiction": float(arr[0]),
            "entailment": float(arr[1]),
            "neutral": float(arr[2]),
        }


def load_nli(model_id: str, hf_home: str, use_nli: bool, loader: str = "transformers") -> tuple[Any | None, dict[str, Any]]:
    info = {
        "nli_backend": False,
        "nli_loader": None,
        "nli_model_id": model_id,
        "nli_hf_home": hf_home,
        "nli_local_files_only": True,
        "nli_failure": None,
    }
    if not use_nli:
        return None, info
    loaders = ["sentence_transformers", "transformers"] if loader == "auto" else [loader]
    failures: list[str] = []
    for candidate in loaders:
        try:
            if candidate == "sentence_transformers":
                nli = SentenceTransformerNLIGrounder(model_id, hf_home)
                info.update({"nli_backend": True, "nli_loader": "sentence_transformers_cross_encoder"})
                return nli, info
            if candidate == "transformers":
                nli = NLIGrounder(model_id, hf_home)
                info.update({"nli_backend": True, "nli_loader": "transformers_model_id"})
                return nli, info
            failures.append(f"unknown_loader:{candidate}")
        except Exception as exc:
            failures.append(f"{candidate}:{type(exc).__name__}: {exc}")
    info["nli_failure"] = " | ".join(failures)
    return None, info


def evidence_index(pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for section in ["company_evidence", "context_evidence"]:
        items = pack.get(section, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    eid = clean_string(item.get("evidence_id"))
                    if eid:
                        out[eid] = item
    return out


def signal_index(pack: dict[str, Any], fallback_tokens: Any = None) -> dict[str, dict[str, Any]]:
    items = pack.get("technical_signals", [])
    if not isinstance(items, list) or not items:
        try:
            parsed = json.loads(clean_string(fallback_tokens))
            items = parsed if isinstance(parsed, list) else []
        except Exception:
            items = []
    return {f"T{idx}": item for idx, item in enumerate(items, start=1) if isinstance(item, dict)}


def lexical_overlap_score(premise: str, hypothesis: str) -> float:
    premise_terms = tokens_from_text(premise)
    hypothesis_terms = tokens_from_text(hypothesis)
    if not premise_terms or not hypothesis_terms:
        return 0.0
    recall = len(premise_terms & hypothesis_terms) / len(hypothesis_terms)
    precision = len(premise_terms & hypothesis_terms) / len(premise_terms)
    return float(max(recall, math.sqrt(max(0.0, recall * precision))))


def validate_news_claim(claim: dict[str, Any], evidence: dict[str, dict[str, Any]], nli: NLIGrounder | None) -> dict[str, Any]:
    eid = clean_string(claim.get("evidence_id"))
    if not eid:
        return {**claim, "status": "unsupported", "reason": "missing_evidence_id", "score": 0.0}
    ev = evidence.get(eid)
    if ev is None:
        return {**claim, "status": "unsupported", "reason": "unknown_evidence_id", "score": 0.0}
    premise = f"{clean_string(ev.get('headline'))} {clean_string(ev.get('body_excerpt'))}".strip()
    hypothesis = clean_string(claim.get("factor"))
    lexical = lexical_overlap_score(premise, hypothesis)
    nli_info = nli.score(premise, hypothesis) if nli is not None else {"label": "not_loaded", "entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    score = max(float(lexical), float(nli_info.get("entailment", 0.0)))
    if float(nli_info.get("contradiction", 0.0)) >= 0.70 and score < 0.55:
        status, reason = "contradiction", "nli_contradiction"
    elif score >= 0.55:
        status, reason = "supported", "evidence_support"
    elif score >= 0.25:
        status, reason = "unverified", "weak_support"
    else:
        status, reason = "unsupported", "not_supported_by_cited_evidence"
    return {
        **claim,
        "status": status,
        "reason": reason,
        "score": float(score),
        "lexical_score": float(lexical),
        "nli_label": nli_info.get("label"),
        "nli_entailment": float(nli_info.get("entailment", 0.0)),
        "nli_contradiction": float(nli_info.get("contradiction", 0.0)),
    }


def validate_technical_claim(claim: dict[str, Any], signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sid = clean_string(claim.get("signal_id"))
    if not sid:
        return {**claim, "status": "unsupported", "reason": "missing_signal_id", "score": 0.0}
    signal = signals.get(sid)
    if signal is None:
        return {**claim, "status": "unsupported", "reason": "unknown_signal_id", "score": 0.0}
    claim_direction = direction_family(claim.get("direction"))
    signal_direction = direction_family(signal.get("direction_prior", signal.get("direction")))
    claim_terms = tokens_from_text(f"{claim.get('signal')} {claim.get('strength')}")
    signal_terms = tokens_from_text(" ".join(clean_string(signal.get(key)) for key in ["token", "rule", "strength", "direction_prior", "evidence_column"]))
    overlap = len(claim_terms & signal_terms)
    if claim_direction != "neutral" and signal_direction != "neutral" and claim_direction != signal_direction:
        return {**claim, "status": "contradiction", "reason": "technical_direction_mismatch", "score": 0.20}
    score = 0.65 + min(0.25, 0.08 * overlap)
    return {**claim, "status": "supported", "reason": "signal_id_support", "score": float(score), "matched_signal": signal}


def score_row(row: pd.Series, nli: NLIGrounder | None) -> dict[str, Any]:
    parsed = parse_json_object(row.get("parsed_json"))
    pack = parse_json_object(row.get("evidence_pack_json"))
    evidence = evidence_index(pack)
    signals = signal_index(pack, row.get("technical_event_tokens_json"))
    scored: list[dict[str, Any]] = []
    for claim in parsed.get("news_rationale", []) if isinstance(parsed.get("news_rationale"), list) else []:
        if isinstance(claim, dict):
            scored.append({"claim_type": "news", **validate_news_claim(claim, evidence, nli)})
    for claim in parsed.get("technical_rationale", []) if isinstance(parsed.get("technical_rationale"), list) else []:
        if isinstance(claim, dict):
            scored.append({"claim_type": "technical", **validate_technical_claim(claim, signals)})
    statuses = [item.get("status") for item in scored]
    total = len(scored)
    supported = sum(status == "supported" for status in statuses)
    unsupported = sum(status == "unsupported" for status in statuses)
    unverified = sum(status == "unverified" for status in statuses)
    contradiction = sum(status == "contradiction" for status in statuses)
    final_status = "not_applicable"
    if contradiction:
        final_status = "contradiction"
    elif unsupported:
        final_status = "unsupported"
    elif unverified:
        final_status = "unverified"
    elif supported:
        final_status = "supported"
    news_items = [item for item in scored if item.get("claim_type") == "news"]
    tech_items = [item for item in scored if item.get("claim_type") == "technical"]
    return {
        "status": final_status,
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "unverified_claims": unverified,
        "contradiction_claims": contradiction,
        "news_grounding_score": safe_rate(sum(item.get("status") == "supported" for item in news_items), len(news_items)) if news_items else None,
        "technical_grounding_score": safe_rate(sum(item.get("status") == "supported" for item in tech_items), len(tech_items)) if tech_items else None,
        "missing_evidence_id_count": sum(item.get("reason") == "missing_evidence_id" for item in scored),
        "unknown_evidence_id_count": sum(item.get("reason") == "unknown_evidence_id" for item in scored),
        "missing_signal_id_count": sum(item.get("reason") == "missing_signal_id" for item in scored),
        "unknown_signal_id_count": sum(item.get("reason") == "unknown_signal_id" for item in scored),
        "unsupported_news_claim_count": sum(item.get("claim_type") == "news" and item.get("status") == "unsupported" for item in scored),
        "news_claim_count": len(news_items),
        "technical_claim_count": len(tech_items),
        "claim_details_json": json.dumps(scored, ensure_ascii=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--examples", default="outputs/data_samples/claim_grounding_evidence_v4_examples.json")
    parser.add_argument("--manifest", default="outputs/manifests/11_EVIDENCE_GROUNDING_JUDGE_V4.manifest.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument("--nli-model-id", default=DEFAULT_NLI_MODEL_ID)
    parser.add_argument("--nli-loader", default="transformers", choices=["transformers", "sentence_transformers", "auto"])
    parser.add_argument("--use-nli", default="true", choices=["true", "false"])
    parser.add_argument("--require-nli", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if Path(args.rationales).exists() else pd.DataFrame()
    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    if args.limit and args.limit > 0:
        rationales = rationales.head(args.limit).copy()
    if rationales.empty:
        failures.append(f"rationales missing or empty: {args.rationales}")
    if contexts.empty:
        failures.append(f"contexts missing or empty: {args.contexts}")

    nli, nli_info = load_nli(args.nli_model_id, args.hf_home, args.use_nli == "true", args.nli_loader)
    if args.require_nli and nli is None:
        failures.append("required NLI backend unavailable")

    merged = rationales.merge(
        contexts[
            [
                col
                for col in [
                    "sample_id",
                    "target_label_5",
                    "target_return",
                    "split",
                    "track",
                    "evidence_pack_json",
                    "technical_event_tokens_json",
                    "mean_evidence_quality_score",
                ]
                if col in contexts.columns
            ]
        ],
        on="sample_id",
        how="inner",
        suffixes=("", "_context"),
    )
    if merged.empty and not failures:
        failures.append("rationales and contexts do not join on sample_id")

    rows: list[dict[str, Any]] = []
    bad_examples: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        scored = score_row(row, nli)
        detail = json.loads(scored["claim_details_json"])
        for item in detail:
            if item.get("status") in {"unsupported", "unverified", "contradiction"}:
                bad_examples.append(
                    {
                        "sample_id": row["sample_id"],
                        "candidate_id": int(row.get("candidate_id", 0)),
                        "claim_type": item.get("claim_type"),
                        "status": item.get("status"),
                        "reason": item.get("reason"),
                        "claim": {key: item.get(key) for key in ["evidence_id", "signal_id", "factor", "signal", "direction", "strength"]},
                    }
                )
        rows.append(
            {
                "sample_id": row["sample_id"],
                "candidate_id": int(row.get("candidate_id", 0)),
                "split": row.get("split", row.get("split_context")),
                "track": row.get("track"),
                **scored,
            }
        )

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    write_json(args.examples, bad_examples[:100])

    total_claims = int(out["total_claims"].sum()) if len(out) and "total_claims" in out else 0
    news_claims = int(out["news_claim_count"].sum()) if len(out) and "news_claim_count" in out else 0
    tech_claims = int(out["technical_claim_count"].sum()) if len(out) and "technical_claim_count" in out else 0
    missing_evidence = int(out["missing_evidence_id_count"].sum()) if len(out) else 0
    unknown_evidence = int(out["unknown_evidence_id_count"].sum()) if len(out) else 0
    unknown_signal = int(out["unknown_signal_id_count"].sum()) if len(out) else 0
    unsupported_news = int(out["unsupported_news_claim_count"].sum()) if len(out) else 0
    metrics = {
        "rows": int(len(out)),
        "total_claims": total_claims,
        "news_claims": news_claims,
        "technical_claims": tech_claims,
        "row_status_counts": out["status"].value_counts(dropna=False).to_dict() if len(out) else {},
        "claim_status_counts": {
            "supported": int(out["supported_claims"].sum()) if len(out) else 0,
            "unsupported": int(out["unsupported_claims"].sum()) if len(out) else 0,
            "unverified": int(out["unverified_claims"].sum()) if len(out) else 0,
            "contradiction": int(out["contradiction_claims"].sum()) if len(out) else 0,
        },
        "missing_evidence_id_rate": safe_rate(missing_evidence, news_claims),
        "unknown_evidence_id_rate": safe_rate(unknown_evidence, news_claims),
        "technical_unknown_signal_rate": safe_rate(unknown_signal, tech_claims),
        "unsupported_news_claim_rate": safe_rate(unsupported_news, news_claims),
        "mean_news_grounding_score": float(pd.to_numeric(out.get("news_grounding_score"), errors="coerce").mean()) if len(out) else 0.0,
        "mean_technical_grounding_score": float(pd.to_numeric(out.get("technical_grounding_score"), errors="coerce").mean()) if len(out) else 0.0,
        "bad_examples_saved": len(bad_examples[:100]),
        **nli_info,
        "require_nli": bool(args.require_nli),
    }
    if len(out) == 0:
        failures.append("grounding output is empty")
    if total_claims == 0:
        failures.append("no claims extracted")
    if metrics["missing_evidence_id_rate"] > 0.02:
        failures.append(f"missing_evidence_id_rate {metrics['missing_evidence_id_rate']:.4f} > 0.0200")
    if metrics["unknown_evidence_id_rate"] != 0.0:
        failures.append(f"unknown_evidence_id_rate {metrics['unknown_evidence_id_rate']:.4f} != 0")
    if metrics["technical_unknown_signal_rate"] > 0.02:
        failures.append(f"technical_unknown_signal_rate {metrics['technical_unknown_signal_rate']:.4f} > 0.0200")

    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics, args.examples], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.rationales, args.contexts],
        outputs_created=[args.output, args.metrics, args.examples, args.manifest, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
