from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.parse_and_validate_rationale import FORECAST_KEYS, canonical_forecast_key, forecast_value, parse_llm_json_strict
from src.utils.artifacts import write_json, write_manifest, write_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STEP = "08_INDEPENDENT_JUDGE_ENSEMBLE_V6"
STRICT_SUM_MIN = 0.99
STRICT_SUM_MAX = 1.01
REPAIR_SUM_MIN = 0.95
REPAIR_SUM_MAX = 1.05
MIN_USABLE_SCHEMA_OK_RATE = 0.95
MAX_REPAIR_RATE = 0.35
MAX_REPAIR_ARGMAX_CHANGE_RATE = 0.0
MIN_TRUE_LABEL_PROBABILITY = 0.22
MIN_ARGMAX_CONSISTENCY = 0.75

PROMPT_TEMPLATE = """You are an independent, objective financial judge. Infer the probability distribution of the stock's NEXT TRADING DAY abnormal return.
DO NOT use any external or future information. You must rely ONLY on the provided context and the extracted rationale.
Do not copy probabilities from the analyst rationale. Re-evaluate from evidence.
Class meanings: strong_down (<-3%), mild_down (-3% to -0.75%), neutral (-0.75% to 0.75%), mild_up (0.75% to 3%), strong_up (>3%).

News Headline: {headline}
News Body: {body}
Market Regime: {regime_label}
Technical Indicator Tokens:
{technical_event_tokens}

---
Extracted Rationale from Analyst:
News Rationale: {news_rationale}
Technical Rationale: {technical_rationale}
Conflict Resolution: {conflict_resolution}
---

Label-order robustness audit:
For this run, review the label definitions in this diagnostic order: {label_order}.
This order is only for robustness auditing; it must NOT change the canonical JSON key order below.

Return canonical JSON keys in the schema below. Choose exactly ONE calibrated probability template and copy its five probability values exactly:
- strong_down evidence: strong_down=0.55, mild_down=0.25, neutral=0.10, mild_up=0.05, strong_up=0.05
- mild_down evidence: strong_down=0.15, mild_down=0.45, neutral=0.25, mild_up=0.10, strong_up=0.05
- neutral or mixed evidence: strong_down=0.05, mild_down=0.20, neutral=0.50, mild_up=0.20, strong_up=0.05
- mild_up evidence: strong_down=0.05, mild_down=0.10, neutral=0.25, mild_up=0.45, strong_up=0.15
- strong_up evidence: strong_down=0.05, mild_down=0.05, neutral=0.10, mild_up=0.25, strong_up=0.55
Use strong_down/strong_up templates only when evidence supports a tail move beyond 3%; otherwise prefer mild or neutral/mixed templates.

Output STRICTLY valid JSON matching this schema:
{{
  "evidence_direction": "down|neutral|up|mixed",
  "confidence": "low|medium|high",
  "forecast_distribution": {{
    "strong_down": 0.0,
    "mild_down": 0.0,
    "neutral": 0.0,
    "mild_up": 0.0,
    "strong_up": 0.0
  }}
}}"""


def format_rationale_list(lst: Any) -> str:
    if not isinstance(lst, list) or not lst:
        return "None"
    return "\n- " + "\n- ".join(str(x) for x in lst)


def clip_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [TRUNCATED]"


def average_distributions(dists: list[dict[str, float]]) -> dict[str, float]:
    keys = set()
    for d in dists:
        keys.update(d.keys())
    out = {}
    for k in keys:
        out[k] = sum(d.get(k, 0.0) for d in dists) / len(dists)
    return out


def argmax_label(dist: dict[str, float]) -> str:
    return max(FORECAST_KEYS, key=lambda k: dist.get(k, 0.0))


def distribution_from_probs(probs: list[float]) -> dict[str, float]:
    return {k: float(probs[i]) for i, k in enumerate(FORECAST_KEYS)}


def normalize_probs(probs: list[float]) -> list[float]:
    total = float(sum(probs))
    if total <= 0:
        raise ValueError("probability total must be positive")
    return [float(p) / total for p in probs]


def project_to_probability_simplex(probs: list[float]) -> list[float]:
    """L2 projection onto the probability simplex."""
    if not probs:
        return []
    values = [float(p) for p in probs]
    sorted_values = sorted(values, reverse=True)
    cumulative = 0.0
    theta = 0.0
    for idx, value in enumerate(sorted_values, start=1):
        cumulative += value
        candidate_theta = (cumulative - 1.0) / idx
        if value - candidate_theta > 0:
            theta = candidate_theta
    projected = [max(value - theta, 0.0) for value in values]
    total = sum(projected)
    if total <= 0:
        raise ValueError("simplex projection produced zero total")
    return normalize_probs(projected)


def generate_stable_random_order(sample_id: str) -> list[str]:
    # Stable random seed based on sample_id string hash
    seed = sum(ord(c) for c in str(sample_id))
    rng = random.Random(seed)
    keys = list(FORECAST_KEYS)
    rng.shuffle(keys)
    return keys


def build_prompt(context_row: pd.Series, rationale_row: pd.Series, variant: str) -> str:
    ctx = {
        "headline": clip_text(context_row.get("headline", context_row.get("aggregated_headlines", "")), 500),
        "body": clip_text(context_row.get("body", context_row.get("aggregated_body", "")), 1500),
        "regime_label": clip_text(context_row.get("regime_label", "normal_vol"), 120),
        "technical_event_tokens": clip_text(
            context_row.get("technical_event_tokens", context_row.get("technical_event_tokens_json", "")),
            800,
        )
    }

    parsed_json = {}
    if pd.notna(rationale_row.get("parsed_json")):
        try:
            parsed_json = json.loads(rationale_row["parsed_json"])
        except Exception:
            pass

    ctx["news_rationale"] = format_rationale_list(parsed_json.get("news_rationale", []))
    ctx["technical_rationale"] = format_rationale_list(parsed_json.get("technical_rationale", []))
    ctx["conflict_resolution"] = parsed_json.get("conflict_resolution", "None")

    if variant == "normal":
        order = list(FORECAST_KEYS)
    elif variant == "reversed":
        order = list(FORECAST_KEYS)[::-1]
    elif variant == "stable_random":
        order = generate_stable_random_order(rationale_row["sample_id"])
    else:
        order = list(FORECAST_KEYS)

    ctx["label_order"] = ", ".join(order)

    return PROMPT_TEMPLATE.format(**ctx)


def generate_judgments(model_path: str, items: list[dict], batch_size: int = 4, max_new_tokens: int = 160) -> Iterable[dict]:
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()

    for start in range(0, len(items), max(1, batch_size)):
        batch = items[start:start + batch_size]
        chat_texts = []
        for item in batch:
            messages = [
                {"role": "system", "content": "You generate concise financial JSON. Return JSON only."},
                {"role": "user", "content": item["prompt"]}
            ]
            chat_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        input_width = inputs["input_ids"].shape[-1]
        for item, generated_ids in zip(batch, generated):
            output_ids = generated_ids[input_width:]
            item["raw_judge_output"] = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            yield item


def parse_judge_output(raw_text: str) -> dict:
    result = {
        "parse_ok": False,
        "schema_ok": False,
        "raw_schema_ok": False,
        "usable_schema_ok": False,
        "error_type": "invalid_json",
        "raw_error_type": "invalid_json",
        "dist": {},
        "raw_dist": {},
        "usable_dist": {},
        "prob_sum": 0.0,
        "raw_prob_sum": 0.0,
        "usable_prob_sum": 0.0,
        "normalized": False,
        "repaired": False,
        "repair_type": "none",
        "repair_argmax_changed": False,
        "raw_keys": [],
        "evidence_direction": "",
        "confidence": ""
    }
    parsed = parse_llm_json_strict(raw_text)
    if not parsed:
        return result

    result["parse_ok"] = True
    result["error_type"] = "schema_mismatch"
    result["raw_error_type"] = "schema_mismatch"
    
    result["evidence_direction"] = parsed.get("evidence_direction", "")
    result["confidence"] = parsed.get("confidence", "")

    dist = parsed.get("forecast_distribution", {})
    if not isinstance(dist, dict):
        return result

    try:
        keys = list(FORECAST_KEYS)
        canonical_keys = [canonical_forecast_key(str(k)) for k in dist]
        result["raw_keys"] = canonical_keys
        
        if sorted(canonical_keys) != sorted(keys):
            return result
            
        probs = [float(forecast_value(dist, k)) for k in keys]
        total = sum(probs)
        result["prob_sum"] = total
        result["raw_prob_sum"] = total
        
        if not (all(np.isfinite(probs)) and all(0 <= p <= 1 for p in probs)):
            result["error_type"] = "invalid_probs"
            result["raw_error_type"] = "invalid_probs"
            return result
            
        raw_dist = {k: probs[i] for i, k in enumerate(keys)}
        result["dist"] = raw_dist
        result["raw_dist"] = raw_dist
        
        if STRICT_SUM_MIN <= total <= STRICT_SUM_MAX:
            result["raw_schema_ok"] = True
            result["raw_error_type"] = "none"
            if abs(total - 1.0) > 1e-6:
                normalized_probs = normalize_probs(probs)
                result["normalized"] = True
            else:
                normalized_probs = probs

            result["usable_dist"] = distribution_from_probs(normalized_probs)
            result["usable_prob_sum"] = float(sum(normalized_probs))
            result["dist"] = result["usable_dist"]
            result["schema_ok"] = True
            result["usable_schema_ok"] = True
            result["error_type"] = "none"
        elif REPAIR_SUM_MIN <= total <= REPAIR_SUM_MAX and total > 0:
            projected_probs = project_to_probability_simplex(probs)
            projected_dist = distribution_from_probs(projected_probs)
            raw_argmax = argmax_label(raw_dist)
            repaired_argmax = argmax_label(projected_dist)
            result["repair_argmax_changed"] = raw_argmax != repaired_argmax
            if result["repair_argmax_changed"]:
                result["error_type"] = "repair_argmax_changed"
                result["raw_error_type"] = "prob_sum_out_of_range"
            else:
                result["usable_dist"] = projected_dist
                result["usable_prob_sum"] = float(sum(projected_probs))
                result["dist"] = projected_dist
                result["schema_ok"] = True
                result["usable_schema_ok"] = True
                result["normalized"] = True
                result["repaired"] = True
                result["repair_type"] = "simplex_projection"
                result["error_type"] = "prob_sum_repaired"
                result["raw_error_type"] = "prob_sum_out_of_range"
                
        else:
            result["error_type"] = "prob_sum_out_of_range"
            result["raw_error_type"] = "prob_sum_out_of_range"
            
    except Exception as e:
        result["error_type"] = f"exception: {str(e)}"
        result["raw_error_type"] = result["error_type"]
        
    return result


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    q = np.clip(q, 1e-9, 1.0)
    return float(np.sum(p * np.log(p / q)))


def aggregate_results(results: dict[str, dict], variants: list[str]) -> tuple[list[dict], list[dict]]:
    final_rows = []
    sample_audit_rows = []

    for _, data in results.items():
        row = {"sample_id": data["sample_id"], "candidate_id": data["candidate_id"], "target_label_5": data["target_label_5"]}
        dists = []
        raw_schema_oks = 0
        usable_schema_oks = 0

        for v in variants:
            pr = data["parse_results"].get(v, {})
            raw_ok = bool(pr.get("raw_schema_ok", False))
            usable_ok = bool(pr.get("usable_schema_ok", pr.get("schema_ok", False)))
            row[f"parse_ok_{v}"] = bool(pr.get("parse_ok", False))
            row[f"raw_schema_ok_{v}"] = raw_ok
            row[f"usable_schema_ok_{v}"] = usable_ok
            row[f"schema_ok_{v}"] = usable_ok
            row[f"error_type_{v}"] = pr.get("error_type", "missing")
            row[f"raw_error_type_{v}"] = pr.get("raw_error_type", "missing")
            row[f"prob_sum_{v}"] = pr.get("raw_prob_sum", pr.get("prob_sum", 0.0))
            row[f"usable_prob_sum_{v}"] = pr.get("usable_prob_sum", 0.0)
            row[f"repaired_{v}"] = bool(pr.get("repaired", False))
            row[f"repair_type_{v}"] = pr.get("repair_type", "none")
            row[f"repair_argmax_changed_{v}"] = bool(pr.get("repair_argmax_changed", False))

            raw_dist = pr.get("raw_dist") or pr.get("dist", {})
            usable_dist = pr.get("usable_dist") or (pr.get("dist", {}) if usable_ok else {})
            for k in FORECAST_KEYS:
                row[f"p_{k}_{v}_raw"] = raw_dist.get(k, 0.0)
                row[f"p_{k}_{v}_usable"] = usable_dist.get(k, 0.0)

            if raw_ok:
                raw_schema_oks += 1
            if usable_ok and usable_dist:
                dists.append(usable_dist)
                usable_schema_oks += 1

        row["raw_valid_variant_count"] = raw_schema_oks
        row["valid_variant_count"] = usable_schema_oks
        row["raw_judge_schema_ok"] = raw_schema_oks == len(variants)
        row["judge_schema_ok"] = usable_schema_oks == len(variants)

        if dists:
            avg_dist = average_distributions(dists)
            for k in FORECAST_KEYS:
                row[f"p_{k}"] = avg_dist.get(k, 0.0)

            argmaxes = [argmax_label(d) for d in dists]
            most_common = max(set(argmaxes), key=argmaxes.count)
            row["argmax_consistency_ensemble"] = argmaxes.count(most_common) / len(dists)

            target = data["target_label_5"]
            if target in FORECAST_KEYS:
                row["true_label_probability_ensemble"] = avg_dist.get(target, 0.0)
            else:
                row["true_label_probability_ensemble"] = avg_dist.get("neutral", 0.0)

            if len(dists) > 1:
                mean_arr = np.array([avg_dist.get(k, 0.0) for k in FORECAST_KEYS])
                kls = []
                for d in dists:
                    d_arr = np.array([d.get(k, 0.0) for k in FORECAST_KEYS])
                    kls.append(kl_divergence(d_arr, mean_arr))
                row["label_order_kl_mean"] = float(np.mean(kls))
                row["judge_disagreement_entropy"] = row["label_order_kl_mean"]
            else:
                row["label_order_kl_mean"] = 0.0
                row["judge_disagreement_entropy"] = 0.0
        else:
            for k in FORECAST_KEYS:
                row[f"p_{k}"] = 0.0
            row["argmax_consistency_ensemble"] = np.nan
            row["true_label_probability_ensemble"] = np.nan
            row["label_order_kl_mean"] = np.nan
            row["judge_disagreement_entropy"] = np.nan

        final_rows.append(row)

        if len(sample_audit_rows) < 50:
            sample_audit_rows.append({
                "sample_id": data["sample_id"],
                "candidate_id": data["candidate_id"],
                "variants": [
                    {
                        "variant": v,
                        "prompt": data["prompts"].get(v, ""),
                        "raw_judge_output": data["raw_judgments"].get(v, ""),
                        "parse_result": data["parse_results"].get(v, {})
                    } for v in variants
                ]
            })

    return final_rows, sample_audit_rows


def compute_metrics(out_df: pd.DataFrame, variants: list[str]) -> dict:
    total_rows = len(out_df)
    total_variants = max(1, total_rows * len(variants))
    repaired_count = int(sum(out_df.get(f"repaired_{v}", pd.Series(dtype=bool)).fillna(False).sum() for v in variants))
    repair_argmax_change_count = int(sum(out_df.get(f"repair_argmax_changed_{v}", pd.Series(dtype=bool)).fillna(False).sum() for v in variants))
    raw_variant_ok_count = int(sum(out_df.get(f"raw_schema_ok_{v}", pd.Series(dtype=bool)).fillna(False).sum() for v in variants))
    usable_variant_ok_count = int(sum(out_df.get(f"usable_schema_ok_{v}", pd.Series(dtype=bool)).fillna(False).sum() for v in variants))

    error_counts: Counter[str] = Counter()
    raw_error_counts: Counter[str] = Counter()
    prob_sum_bins: Counter[str] = Counter()
    for v in variants:
        if f"error_type_{v}" in out_df:
            error_counts.update(str(x) for x in out_df[f"error_type_{v}"].fillna("missing").tolist())
        if f"raw_error_type_{v}" in out_df:
            raw_error_counts.update(str(x) for x in out_df[f"raw_error_type_{v}"].fillna("missing").tolist())
        if f"prob_sum_{v}" in out_df:
            for value in out_df[f"prob_sum_{v}"].dropna().tolist():
                prob_sum_bins[f"{float(value):.2f}"] += 1

    raw_schema_ok_rate = float(out_df["raw_judge_schema_ok"].mean()) if total_rows else 0.0
    usable_schema_ok_rate = float(out_df["judge_schema_ok"].mean()) if total_rows else 0.0
    mean_consistency = out_df["argmax_consistency_ensemble"].mean() if total_rows and not out_df["argmax_consistency_ensemble"].isna().all() else 0.0
    mean_true_prob = out_df["true_label_probability_ensemble"].mean() if total_rows and not out_df["true_label_probability_ensemble"].isna().all() else 0.0

    return {
        "rows": total_rows,
        "raw_schema_ok_rate": raw_schema_ok_rate,
        "usable_schema_ok_rate": usable_schema_ok_rate,
        "judge_schema_ok_rate": usable_schema_ok_rate,
        "raw_variant_schema_ok_rate": raw_variant_ok_count / total_variants,
        "usable_variant_schema_ok_rate": usable_variant_ok_count / total_variants,
        "repair_rate": repaired_count / total_variants,
        "repair_count": repaired_count,
        "repair_argmax_change_rate": repair_argmax_change_count / total_variants,
        "repair_argmax_change_count": repair_argmax_change_count,
        "error_type_counts": dict(sorted(error_counts.items())),
        "raw_error_type_counts": dict(sorted(raw_error_counts.items())),
        "prob_sum_bins": dict(sorted(prob_sum_bins.items())),
        "mean_argmax_consistency_ensemble": float(mean_consistency),
        "mean_true_label_probability_ensemble": float(mean_true_prob),
    }


def evaluate_gate(metrics: dict[str, Any]) -> list[str]:
    failures = []
    usable_schema_ok_rate = float(metrics.get("usable_schema_ok_rate", metrics.get("judge_schema_ok_rate", 0.0)))
    repair_rate = float(metrics.get("repair_rate", 0.0))
    repair_argmax_change_rate = float(metrics.get("repair_argmax_change_rate", 0.0))
    mean_consistency = float(metrics.get("mean_argmax_consistency_ensemble", 0.0))
    mean_true_prob = float(metrics.get("mean_true_label_probability_ensemble", 0.0))

    if usable_schema_ok_rate < MIN_USABLE_SCHEMA_OK_RATE:
        failures.append(f"usable_schema_ok_rate {usable_schema_ok_rate:.3f} < {MIN_USABLE_SCHEMA_OK_RATE:.2f}")
    if repair_argmax_change_rate > MAX_REPAIR_ARGMAX_CHANGE_RATE:
        failures.append(f"repair_argmax_change_rate {repair_argmax_change_rate:.3f} > {MAX_REPAIR_ARGMAX_CHANGE_RATE:.2f}")
    if repair_rate > MAX_REPAIR_RATE:
        failures.append(f"repair_rate {repair_rate:.3f} > {MAX_REPAIR_RATE:.2f}")
    if mean_true_prob <= MIN_TRUE_LABEL_PROBABILITY:
        failures.append(f"true_label_prob {mean_true_prob:.3f} <= {MIN_TRUE_LABEL_PROBABILITY:.2f}")
    if mean_consistency < MIN_ARGMAX_CONSISTENCY:
        failures.append(f"consistency {mean_consistency:.3f} < {MIN_ARGMAX_CONSISTENCY:.2f}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--label-order-variants", default="normal,reversed,stable_random")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of sample candidates (for smoke test)")
    args = parser.parse_args()

    # Load data
    logging.info("Loading datasets")
    contexts_df = pd.read_parquet(args.contexts)
    rationales_df = pd.read_parquet(args.rationales)

    # Merge
    merged = pd.merge(rationales_df, contexts_df, on="sample_id", how="inner")
    
    if merged.empty:
        logging.error("No data to process")
        sys.exit(1)
        
    if args.limit > 0:
        merged = merged.head(args.limit)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    hf_home = os.environ.get("HF_HOME", "E:\\huggingface")
    model_path = cfg["models"].get(args.model_key, args.model_key).replace("$HF_HOME", hf_home)
    logging.info(f"Using model {model_path}")

    variants = [v.strip() for v in args.label_order_variants.split(",")]
    logging.info(f"Using variants: {variants}")

    # Resume logic
    tmp_path = args.output + ".tmp.jsonl"
    existing_keys = set()
    if args.resume and os.path.exists(tmp_path):
        with open(tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_keys.add(f"{obj['sample_id']}_{obj['candidate_id']}_{obj['variant']}")
                except Exception:
                    pass
        logging.info(f"Resuming: found {len(existing_keys)} existing inferences.")

    # To optimize LLM calls, we will flatten items, so one item per row * variant
    items = []
    for _, row in merged.iterrows():
        cand_id = int(row.get("candidate_id", 0))
        for v in variants:
            key_check = f"{row['sample_id']}_{cand_id}_{v}"
            if key_check in existing_keys:
                continue
            prompt = build_prompt(row, row, v)
            items.append({
                "sample_id": str(row["sample_id"]),
                "candidate_id": cand_id,
                "variant": v,
                "prompt": prompt,
                "target_label_5": row.get("target_label_5")
            })

    logging.info(f"Generating {len(items)} remaining inferences...")
    
    tmp_mode = "a" if args.resume else "w"
    tmp_f = open(tmp_path, tmp_mode, encoding="utf-8") if len(items) > 0 else None
    
    processed = 0
    start_time = time.time()
    total_items = len(items)
    for res in generate_judgments(model_path, items, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens):
        s_id = str(res["sample_id"])
        c_id = res["candidate_id"]
        v = res["variant"]
        
        parsed = parse_judge_output(res["raw_judge_output"])
        out_obj = {
            "sample_id": s_id,
            "candidate_id": c_id,
            "variant": v,
            "target_label_5": res["target_label_5"],
            "raw_judge_output": res["raw_judge_output"],
            "prompt": res["prompt"],
            "parse_result": parsed
        }
        if tmp_f:
            tmp_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            tmp_f.flush()
        processed += 1
        if processed == 1 or processed % 100 == 0 or processed == total_items:
            elapsed = max(time.time() - start_time, 1e-9)
            rate = processed / elapsed
            remaining = (total_items - processed) / rate if rate > 0 else 0.0
            logging.info(
                "Judge inference progress: %d/%d new calls (%.2f calls/s, eta %.1f min)",
                processed,
                total_items,
                rate,
                remaining / 60.0,
            )

    if tmp_f:
        tmp_f.close()
        
    # Read everything back
    results = {}
    with open(tmp_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                res = json.loads(line)
                s_id = str(res["sample_id"])
                c_id = res["candidate_id"]
                key = f"{s_id}_{c_id}"
                
                if key not in results:
                    results[key] = {
                        "sample_id": s_id,
                        "candidate_id": c_id,
                        "target_label_5": res["target_label_5"],
                        "raw_judgments": {},
                        "parse_results": {},
                        "prompts": {}
                    }
                    
                v = res["variant"]
                results[key]["raw_judgments"][v] = res["raw_judge_output"]
                results[key]["parse_results"][v] = res.get("parse_result", {})
                results[key]["prompts"][v] = res.get("prompt", "")
            except Exception:
                pass
        
    final_rows, sample_audit_rows = aggregate_results(results, variants)

    out_df = pd.DataFrame(final_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    
    metrics = compute_metrics(out_df, variants)
    failures = evaluate_gate(metrics)
    
    write_json(args.metrics, metrics)
    write_jsonl(args.samples, sample_audit_rows)
    
    write_manifest(args.manifest, [args.output, args.metrics, args.samples], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.rationales, args.contexts], [args.output, args.metrics, args.samples, args.manifest, args.status], metrics, failures, status == "PASS")
    
    logging.info(
        "Done. Raw schema OK: %.2f, Usable schema OK: %.2f, Consistency: %.2f",
        metrics["raw_schema_ok_rate"],
        metrics["usable_schema_ok_rate"],
        metrics["mean_argmax_consistency_ensemble"],
    )
    return 0 if status == "PASS" else 1

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    sys.exit(main())
