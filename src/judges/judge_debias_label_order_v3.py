import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.parse_and_validate_rationale import FORECAST_KEYS, canonical_forecast_key, forecast_value, parse_llm_json_strict
from src.utils.artifacts import write_json, write_manifest, write_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STEP = "08_JUDGE_DEBIAS_LABEL_ORDER_RANDOMIZATION"

PROMPT_TEMPLATE_REVERSED = """You are an independent, objective financial judge. Infer the probability distribution of the stock's NEXT TRADING DAY abnormal return.
DO NOT use any external or future information. You must rely ONLY on the provided context and the extracted rationale.
Do not copy probabilities from the analyst rationale. Re-evaluate from evidence.

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

Evaluate labels in this displayed order: strong_up, mild_up, neutral, mild_down, strong_down.
Return canonical JSON keys in the schema below. The probabilities must sum to exactly 1.0.

Output STRICTLY valid JSON matching this schema:
{{
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

def build_prompt_reversed(context_row: pd.Series, rationale_row: pd.Series) -> str:
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
    
    return PROMPT_TEMPLATE_REVERSED.format(**ctx)

def generate_judgments(model_path: str, items: list[dict], batch_size: int = 4, max_input_tokens: int = 2048, max_new_tokens: int = 96, attn_implementation: str = "flash_attention_2") -> Iterable[dict]:
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
        attn_implementation=attn_implementation
    )
    model.eval()
    
    for start in range(0, len(items), max(1, batch_size)):
        batch = items[start:start+batch_size]
        chat_texts = []
        for item in batch:
            messages = [
                {"role": "system", "content": "You generate concise financial JSON. Return JSON only."},
                {"role": "user", "content": item["prompt"]}
            ]
            chat_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens)
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
            raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            item["raw_judge_output"] = raw_output
            yield item

def parse_judge_output(raw_text: str) -> dict:
    res = {
        "judge_schema_ok_reversed": False,
        "p_strong_down_reversed": 0.0,
        "p_mild_down_reversed": 0.0,
        "p_neutral_reversed": 0.0,
        "p_mild_up_reversed": 0.0,
        "p_strong_up_reversed": 0.0
    }
    parsed = parse_llm_json_strict(raw_text)
    if not parsed:
        return res
    
    dist = parsed.get("forecast_distribution", {})
    if not isinstance(dist, dict):
        return res
        
    try:
        keys = list(FORECAST_KEYS)
        canonical_keys = [canonical_forecast_key(str(k)) for k in dist]
        if sorted(canonical_keys) != sorted(keys):
            return res
        probs = [float(forecast_value(dist, k)) for k in keys]
        total = sum(probs)
        if 0.99 <= total <= 1.01 and all(np.isfinite(probs)) and all(p >= 0 for p in probs):
            res["judge_schema_ok_reversed"] = True
            probs = [p / total for p in probs]
            for i, k in enumerate(keys):
                res[f"p_{k}_reversed"] = probs[i]
    except Exception:
        pass
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--base-judge", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-consistency", type=float, default=0.55)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    # Load data
    logging.info("Loading datasets")
    contexts_df = pd.read_parquet(args.contexts)
    rationales_df = pd.read_parquet(args.rationales)
    base_judge_df = pd.read_parquet(args.base_judge)

    # Merge on both sample_id and candidate_id to avoid cartesian explosion
    if "candidate_id" in rationales_df.columns and "candidate_id" in base_judge_df.columns:
        merged_base = pd.merge(base_judge_df, rationales_df, on=["sample_id", "candidate_id"], how="inner")
    else:
        merged_base = pd.merge(base_judge_df, rationales_df, on="sample_id", how="inner")
        
    merged = pd.merge(merged_base, contexts_df, on="sample_id", how="inner")
    
    if args.limit > 0:
        merged = merged.head(args.limit)
        
    if merged.empty:
        logging.error("No data to process")
        sys.exit(1)

    model_path = args.judge_model
    if model_path == "qwen3_4b":
        model_path = "qwen3_judge"
    if not os.path.exists(model_path):
        import yaml
        with open("configs/default_paths.yaml") as f:
            cfg = yaml.safe_load(f)
        hf_home = os.environ.get("HF_HOME", "E:\\huggingface")
        model_path = cfg["models"].get(model_path, model_path).replace("$HF_HOME", hf_home)
        
    logging.info(f"Using model {model_path}")

    existing_df = None
    existing_keys = set()
    if args.resume and os.path.exists(args.output):
        try:
            existing_df = pd.read_parquet(args.output)
            for _, r in existing_df.iterrows():
                existing_keys.add((str(r["sample_id"]), int(r.get("candidate_id", 0))))
            logging.info(f"Resuming: found {len(existing_keys)} existing records in {args.output}")
        except Exception as e:
            logging.warning(f"Could not read existing {args.output}: {e}")

    items = []
    for _, row in merged.iterrows():
        cand_id = int(row.get("candidate_id", 0))
        if (str(row["sample_id"]), cand_id) in existing_keys:
            continue
        items.append({
            "sample_id": row["sample_id"],
            "candidate_id": cand_id,
            "target_label_5": row.get("target_label_5"),
            "prompt": build_prompt_reversed(row, row),
            "original_row": row
        })

    logging.info(f"Generating reversed-order judgments for {len(items)} samples")
    results = []
    keys = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
    
    for res, item in zip(
        generate_judgments(
            model_path,
            items,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
        ),
        items,
    ):
        parsed = parse_judge_output(res["raw_judge_output"])
        
        row_dict = item["original_row"].to_dict()
        row_dict.update({
            "raw_judge_output_reversed": res["raw_judge_output"],
            **parsed
        })
        
        if parsed["judge_schema_ok_reversed"] and row_dict.get("judge_schema_ok", False):
            base_probs = np.array([row_dict.get(f"p_{k}", 0.0) for k in keys])
            rev_probs = np.array([row_dict.get(f"p_{k}_reversed", 0.0) for k in keys])
            
            avg_probs = (base_probs + rev_probs) / 2.0
            for i, k in enumerate(keys):
                row_dict[f"p_{k}_debiased"] = avg_probs[i]
                
            base_argmax = np.argmax(base_probs)
            rev_argmax = np.argmax(rev_probs)
            
            row_dict["argmax_consistency"] = int(base_argmax == rev_argmax)
            row_dict["l1_probability_delta"] = np.sum(np.abs(base_probs - rev_probs))
            
            true_label = row_dict.get("target_label_5")
            if true_label in keys:
                idx = keys.index(true_label)
                row_dict["true_label_prob_delta"] = abs(base_probs[idx] - rev_probs[idx])
                row_dict["true_label_probability_debiased"] = avg_probs[idx]
        else:
            row_dict["argmax_consistency"] = np.nan
            row_dict["l1_probability_delta"] = np.nan
            
        results.append(row_dict)

    out_df = pd.DataFrame(results)
    if existing_df is not None and not out_df.empty:
        out_df = pd.concat([existing_df, out_df], ignore_index=True)
    elif existing_df is not None and out_df.empty:
        out_df = existing_df
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    
    valid_mask = out_df["argmax_consistency"].notna()
    if valid_mask.sum() == 0:
        argmax_consistency = 0.0
        mean_l1_delta = 1.0
    else:
        argmax_consistency = out_df.loc[valid_mask, "argmax_consistency"].mean()
        mean_l1_delta = out_df.loc[valid_mask, "l1_probability_delta"].mean()
        
    metrics = {
        "evaluated_rows": len(out_df),
        "valid_debiased_rows": int(valid_mask.sum()),
        "argmax_consistency": float(argmax_consistency),
        "mean_l1_probability_delta": float(mean_l1_delta),
        "mean_true_label_prob_delta": float(out_df["true_label_prob_delta"].mean()) if "true_label_prob_delta" in out_df else 0.0,
        "reversed_schema_ok_rate": float(out_df["judge_schema_ok_reversed"].mean()) if len(out_df) and "judge_schema_ok_reversed" in out_df else 0.0,
        "min_consistency_required": args.min_consistency,
        "debias_reward_source_allowed": bool(argmax_consistency >= args.min_consistency),
    }
    
    write_json(args.metrics, metrics)
        
    failures = []
    if argmax_consistency < args.min_consistency:
        failures.append(f"argmax_consistency {argmax_consistency:.4f} < {args.min_consistency:.4f}")
    if metrics["reversed_schema_ok_rate"] < 0.80:
        failures.append(f"reversed_schema_ok_rate {metrics['reversed_schema_ok_rate']:.4f} < 0.8000")
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.contexts, args.rationales, args.base_judge],
        outputs_created=[args.output, args.metrics, args.manifest, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=(status == "PASS")
    )
    logging.info(f"Done. Consistency: {argmax_consistency:.2f}, L1 Delta: {mean_l1_delta:.2f}")

if __name__ == "__main__":
    main()
