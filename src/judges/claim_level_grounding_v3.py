from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import write_json, write_manifest, write_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STEP = "09_CLAIM_EXTRACTION_GROUNDING_V2"
DEFAULT_HF_HOME = "E:/huggingface"
DEFAULT_NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-small"
NLI_LOADER = "transformers_model_id"
NLI_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}

PLACEHOLDER_PATTERNS = (
    "no significant news",
    "no material news",
    "no relevant news",
    "no significant technical",
    "no material technical",
    "no technical signals",
)
STOPWORDS = {
    "about",
    "after",
    "also",
    "from",
    "have",
    "into",
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
    "price",
}
TECH_KEYWORDS = {
    "bearish",
    "bullish",
    "macd",
    "rsi",
    "sma",
    "bollinger",
    "volume",
    "volatility",
    "trend",
    "momentum",
    "overbought",
    "oversold",
    "regime",
    "relative",
    "strength",
}


class NLIGrounder:
    def __init__(self, model_id: str, hf_home: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if hf_home:
            os.environ["HF_HOME"] = hf_home
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.hf_home = os.environ.get("HF_HOME")
        logging.info("Loading NLI model %s from HF_HOME=%s on %s", model_id, self.hf_home, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True).to(self.device)
        self.model.eval()
        self.id2label = NLI_ID2LABEL.copy()

    def check_claim(self, premise: str, hypothesis: str) -> str:
        if not premise.strip():
            return "not_applicable"
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            probs = self.torch.softmax(self.model(**inputs).logits, dim=1)[0].detach().cpu()
        label = self.id2label.get(int(probs.argmax().item()), "neutral")
        return nli_label_to_status(label)


def nli_label_to_status(label: str) -> str:
    label = str(label).lower()
    if "entail" in label:
        return "supported"
    if "contradict" in label:
        return "contradiction"
    return "unverified"


def read_models_config(path: str = "configs/default_paths.yaml") -> dict[str, Any]:
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        models = cfg.get("models", {})
        return models if isinstance(models, dict) else {}
    except Exception as exc:
        logging.warning("Could not read model config from %s: %s", path, exc)
        return {}


def resolve_hf_home(value: str | None = None) -> str:
    if value and value != "$HF_HOME":
        return value
    env_value = os.environ.get("HF_HOME")
    if env_value:
        return env_value
    return DEFAULT_HF_HOME


def resolve_nli_model_id(value: str | None = None, config_path: str = "configs/default_paths.yaml") -> str:
    if value:
        return value
    models = read_models_config(config_path)
    model_id = models.get("nli_judge_model_id")
    if model_id:
        return str(model_id)
    return DEFAULT_NLI_MODEL_ID


def empty_nli_load_info(model_id: str, hf_home: str) -> dict[str, Any]:
    return {
        "nli_backend": False,
        "nli_loader": None,
        "nli_model_id": model_id,
        "nli_hf_home": hf_home,
        "nli_local_files_only": True,
        "nli_failure": None,
    }


def load_nli_model(model_id: str | None = None, hf_home: str | None = None) -> tuple[NLIGrounder | None, dict[str, Any]]:
    resolved_model_id = resolve_nli_model_id(model_id)
    resolved_hf_home = resolve_hf_home(hf_home)
    os.environ["HF_HOME"] = resolved_hf_home
    info = empty_nli_load_info(resolved_model_id, resolved_hf_home)
    try:
        nli = NLIGrounder(resolved_model_id, resolved_hf_home)
        info.update({"nli_backend": True, "nli_loader": NLI_LOADER, "nli_failure": None})
        return nli, info
    except Exception as exc:
        info["nli_failure"] = f"{type(exc).__name__}: {exc}"
        logging.warning("NLI unavailable with model_id=%s HF_HOME=%s: %s", resolved_model_id, resolved_hf_home, exc)
        return None, info


def nli_gate_failures(nli: NLIGrounder | None, require_nli: bool, allow_lexical_fallback: bool) -> list[str]:
    if nli is not None:
        return []
    if require_nli:
        return ["required NLI backend unavailable"]
    if not allow_lexical_fallback:
        return ["NLI backend unavailable and lexical fallback was not explicitly allowed"]
    return []


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def row_get(row: Any, *names: str, default: str = "") -> str:
    for name in names:
        if name in row:
            value = clean_text(row.get(name))
            if value:
                return value
    return default


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def is_placeholder(text: str) -> bool:
    low = text.strip().lower()
    return not low or any(pattern in low for pattern in PLACEHOLDER_PATTERNS)


def tokens_from_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower().replace("_", " "))
        if len(token) > 3 and token not in STOPWORDS
    }


def parse_technical_tokens(value: Any) -> tuple[str, set[str]]:
    raw = clean_text(value)
    if not raw:
        return "", set()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = raw
    pieces: list[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                pieces.extend(
                    clean_text(item.get(key))
                    for key in ("token", "direction_prior", "strength", "rule", "evidence_column")
                    if clean_text(item.get(key))
                )
            else:
                pieces.append(clean_text(item))
    else:
        pieces.append(raw)
    text = " ".join(pieces)
    return text, tokens_from_text(text)


def extract_claims(parsed_json_str: Any) -> list[dict[str, str]]:
    data = parse_json_object(parsed_json_str)
    claims: list[dict[str, str]] = []
    for item in data.get("news_rationale", []) if isinstance(data.get("news_rationale"), list) else []:
        text = clean_text(item)
        claims.append({"claim_type": "news", "claim": text})
    for item in data.get("technical_rationale", []) if isinstance(data.get("technical_rationale"), list) else []:
        text = clean_text(item)
        ctype = "regime" if "regime" in text.lower() or "volatility" in text.lower() else "technical"
        claims.append({"claim_type": ctype, "claim": text})
    if clean_text(data.get("conflict_resolution")):
        claims.append({"claim_type": "forecast", "claim": clean_text(data.get("conflict_resolution"))})
    if clean_text(data.get("risk_note")):
        claims.append({"claim_type": "risk", "claim": clean_text(data.get("risk_note"))})
    return claims


def lexical_news_status(premise: str, claim: str) -> str:
    if not premise.strip():
        return "not_applicable"
    premise_terms = tokens_from_text(premise)
    claim_terms = tokens_from_text(claim)
    if not claim_terms:
        return "not_applicable"
    overlap = premise_terms & claim_terms
    if len(overlap) >= max(2, min(4, len(claim_terms) // 3)):
        return "supported"
    return "unverified"


def technical_status(claim: str, token_text: str, token_terms: set[str]) -> str:
    if not token_text.strip():
        return "not_applicable"
    claim_terms = tokens_from_text(claim)
    if not claim_terms:
        return "not_applicable"
    direct = claim_terms & token_terms
    semantic = (claim_terms & TECH_KEYWORDS) and (token_terms & TECH_KEYWORDS)
    direction_match = any(term in token_terms for term in ("bearish", "bullish", "down", "up")) and any(
        term in claim_terms for term in ("bearish", "bullish", "down", "up")
    )
    return "supported" if direct or semantic or direction_match else "unverified"


def score_claim(row: Any, claim: dict[str, str], nli: NLIGrounder | None) -> dict[str, Any]:
    claim_type = claim["claim_type"]
    text = claim["claim"]
    if is_placeholder(text):
        return {**claim, "status": "not_applicable"}
    headline = row_get(row, "headline", "aggregated_headlines")
    body = row_get(row, "body", "aggregated_body")
    premise = f"{headline}\n{body}".strip()
    token_text, token_terms = parse_technical_tokens(row_get(row, "technical_event_tokens_json", "technical_event_tokens"))
    if claim_type == "news":
        status = nli.check_claim(premise, text) if nli is not None else lexical_news_status(premise, text)
    elif claim_type in {"technical", "regime"}:
        status = technical_status(text, token_text, token_terms)
    else:
        status = "not_applicable"
    return {**claim, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-supported-rate", type=float, default=0.98)
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--nli-model-id", default=None)
    parser.add_argument("--require-nli", action="store_true")
    parser.add_argument("--allow-lexical-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    contexts_df = pd.read_parquet(args.contexts)
    rationales_df = pd.read_parquet(args.rationales)
    tokens_df = pd.read_parquet(args.tokens) if Path(args.tokens).exists() else pd.DataFrame()
    if not tokens_df.empty and "technical_event_tokens_json" not in contexts_df.columns:
        token_cols = [c for c in ["sample_id", "technical_event_tokens_json"] if c in tokens_df.columns]
        if token_cols:
            contexts_df = contexts_df.merge(
                tokens_df[token_cols].drop_duplicates("sample_id"),
                on="sample_id",
                how="left",
            )
    merged = rationales_df.merge(contexts_df, on="sample_id", how="inner", suffixes=("", "_context"))
    if args.limit and args.limit > 0:
        merged = merged.head(args.limit).copy()

    existing_df = None
    existing_keys: set[tuple[str, int]] = set()
    if args.resume and Path(args.output).exists():
        existing_df = pd.read_parquet(args.output)
        for _, row in existing_df.iterrows():
            existing_keys.add((str(row["sample_id"]), int(row.get("candidate_id", 0))))

    nli, nli_info = load_nli_model(args.nli_model_id, args.hf_home)
    rows: list[dict[str, Any]] = []
    bad_examples: list[dict[str, Any]] = []
    claim_status_counts = {"supported": 0, "unverified": 0, "contradiction": 0, "not_applicable": 0}
    claim_type_counts: dict[str, int] = {}

    for _, row in merged.iterrows():
        cand_id = int(row.get("candidate_id", 0))
        key = (str(row["sample_id"]), cand_id)
        if key in existing_keys:
            continue
        scored = [score_claim(row, claim, nli) for claim in extract_claims(row.get("parsed_json", ""))]
        for item in scored:
            claim_status_counts[item["status"]] = claim_status_counts.get(item["status"], 0) + 1
            claim_type_counts[item["claim_type"]] = claim_type_counts.get(item["claim_type"], 0) + 1
            if item["status"] in {"contradiction", "unverified"}:
                bad_examples.append(
                    {
                        "sample_id": row["sample_id"],
                        "candidate_id": cand_id,
                        "claim_type": item["claim_type"],
                        "claim": item["claim"],
                        "status": item["status"],
                    }
                )
        statuses = [item["status"] for item in scored]
        total_claims = len(statuses)
        supported_claims = statuses.count("supported")
        unverified_claims = statuses.count("unverified")
        contradiction_claims = statuses.count("contradiction")
        not_applicable_claims = statuses.count("not_applicable")
        if contradiction_claims:
            final_status = "contradiction"
        elif unverified_claims:
            final_status = "unverified"
        elif supported_claims:
            final_status = "supported"
        else:
            final_status = "not_applicable"
        news_claims = [item for item in scored if item["claim_type"] == "news"]
        tech_claims = [item for item in scored if item["claim_type"] in {"technical", "regime"}]
        news_supported = sum(1 for item in news_claims if item["status"] == "supported")
        tech_supported = sum(1 for item in tech_claims if item["status"] == "supported")
        rows.append(
            {
                "sample_id": row["sample_id"],
                "candidate_id": cand_id,
                "status": final_status,
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "unverified_claims": unverified_claims,
                "contradiction_claims": contradiction_claims,
                "not_applicable_claims": not_applicable_claims,
                "news_grounding_score": news_supported / len(news_claims) if news_claims else None,
                "technical_grounding_score": tech_supported / len(tech_claims) if tech_claims else None,
                "claim_details_json": json.dumps(scored, ensure_ascii=False),
            }
        )

    out_df = pd.DataFrame(rows)
    if existing_df is not None and not out_df.empty:
        out_df = pd.concat([existing_df, out_df], ignore_index=True)
    elif existing_df is not None:
        out_df = existing_df

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    write_json(args.examples, bad_examples[:100])

    row_status_counts = out_df["status"].value_counts(dropna=False).to_dict() if len(out_df) else {}
    total_claim_count = sum(claim_status_counts.values())
    supported_rate = claim_status_counts.get("supported", 0) / total_claim_count if total_claim_count else 0.0
    metrics = {
        "total_evaluated": int(len(out_df)),
        "total_claims": int(total_claim_count),
        "supported_rate": float(supported_rate),
        "contradiction_rate": float(claim_status_counts.get("contradiction", 0) / total_claim_count) if total_claim_count else 0.0,
        "unverified_rate": float(claim_status_counts.get("unverified", 0) / total_claim_count) if total_claim_count else 0.0,
        "not_applicable_rate": float(claim_status_counts.get("not_applicable", 0) / total_claim_count) if total_claim_count else 0.0,
        "row_status_counts": row_status_counts,
        "claim_status_counts": claim_status_counts,
        "claim_type_counts": claim_type_counts,
        "bad_examples_saved": len(bad_examples[:100]),
        **nli_info,
        "require_nli": bool(args.require_nli),
        "allow_lexical_fallback": bool(args.allow_lexical_fallback),
        "max_supported_rate": args.max_supported_rate,
    }

    failures: list[str] = nli_gate_failures(nli, args.require_nli, args.allow_lexical_fallback)
    if len(out_df) == 0:
        failures.append("grounding output is empty")
    if total_claim_count == 0:
        failures.append("no claims extracted")
    if supported_rate > args.max_supported_rate and claim_status_counts.get("unverified", 0) == 0:
        failures.append(f"supported_rate {supported_rate:.4f} > {args.max_supported_rate:.4f} with no unverified claims")
    if claim_status_counts.get("not_applicable", 0) == total_claim_count and total_claim_count:
        failures.append("all claims are not_applicable")

    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics, args.examples], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.contexts, args.rationales, args.tokens],
        outputs_created=[args.output, args.metrics, args.examples, args.manifest, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    logging.info("Grounding status %s with supported_rate %.3f", status, supported_rate)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
