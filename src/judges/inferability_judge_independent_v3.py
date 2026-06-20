import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.parse_and_validate_rationale import parse_llm_json_strict, validate_rationale_schema_strict
from src.utils.artifacts import write_manifest, write_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STEP = "07_INDEPENDENT_INFERABILITY_JUDGE"

PROMPT_TEMPLATE = """You are an independent, objective financial judge. Your task is to evaluate a set of facts and infer the probability distribution of the stock's return for the NEXT TRADING DAY.
DO NOT use any external or future information. You must rely ONLY on the provided context and the extracted rationale.

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

Based on the above context and rationale, predict the probability distribution across 5 outcomes: strong_down (<-3%), mild_down (-3% to -0.75%), neutral, mild_up (0.75% to 3%), strong_up (>3%). The sum MUST EQUAL EXACTLY 1.0.

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

def build_prompt(context_row: pd.Series, rationale_row: pd.Series) -> str:
    ctx = {
        "headline": clip_text(context_row.get("headline", ""), 500),
        "body": clip_text(context_row.get("body", ""), 1500),
        "regime_label": clip_text(context_row.get("regime_label", "normal_vol"), 120),
        "technical_event_tokens": clip_text(context_row.get("technical_event_tokens", ""), 800)
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
    
    prompt = PROMPT_TEMPLATE.format(**ctx)
    return prompt

def generate_judgments(model_path: str, items: list[dict], batch_size: int = 4) -> Iterable[dict]:
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
        batch = items[start:start+batch_size]
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
                max_new_tokens=128,
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
        "judge_parse_ok": False,
        "judge_schema_ok": False,
        "p_strong_down": 0.0,
        "p_mild_down": 0.0,
        "p_neutral": 0.0,
        "p_mild_up": 0.0,
        "p_strong_up": 0.0
    }
    parsed = parse_llm_json_strict(raw_text)
    if not parsed:
        return res
    res["judge_parse_ok"] = True
    
    dist = parsed.get("forecast_distribution", {})
    if not isinstance(dist, dict):
        return res
        
    try:
        keys = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
        probs = [float(dist.get(k, 0.0)) for k in keys]
        total = sum(probs)
        if 0.95 <= total <= 1.05:
            res["judge_schema_ok"] = True
            probs = [p / total for p in probs]
            for i, k in enumerate(keys):
                res[f"p_{k}"] = probs[i]
    except Exception:
        pass
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    # Load data
    logging.info("Loading contexts and rationales")
    contexts_df = pd.read_parquet(args.contexts)
    rationales_df = pd.read_parquet(args.rationales)
    
    merged = pd.merge(rationales_df, contexts_df, on="sample_id", how="inner", suffixes=("", "_ctx"))
    if args.limit > 0:
        merged = merged.head(args.limit)
        
    if merged.empty:
        logging.error("No data to process")
        sys.exit(1)

    # Determine model path
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
            "prompt": build_prompt(row, row)
        })

    logging.info(f"Generating judgments for {len(items)} samples")
    results = []
    for res in generate_judgments(model_path, items, batch_size=4):
        parsed = parse_judge_output(res["raw_judge_output"])
        
        predicted_label = "neutral"
        if parsed["judge_schema_ok"]:
            keys = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
            probs = [parsed[f"p_{k}"] for k in keys]
            predicted_label = keys[probs.index(max(probs))]
            
        true_label = res["target_label_5"]
        true_label_prob = parsed.get(f"p_{true_label}", 0.0)
        
        row_data = {
            "sample_id": res["sample_id"],
            "candidate_id": res["candidate_id"],
            "judge_model": args.judge_model,
            "predicted_label": predicted_label,
            "true_label_probability": true_label_prob,
            "raw_judge_output": res["raw_judge_output"],
            **parsed
        }
        results.append(row_data)

    out_df = pd.DataFrame(results)
    if existing_df is not None and not out_df.empty:
        out_df = pd.concat([existing_df, out_df], ignore_index=True)
    elif existing_df is not None and out_df.empty:
        out_df = existing_df
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    
    schema_ok_rate = out_df["judge_schema_ok"].mean()
    metrics = {
        "judged_samples": len(out_df),
        "judge_schema_ok_rate": schema_ok_rate,
        "mean_true_label_probability": out_df["true_label_probability"].mean()
    }
    
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)
        
    status = "PASS" if schema_ok_rate >= 0.90 else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.contexts, args.rationales],
        outputs_created=[args.output, args.metrics],
        metrics=metrics,
        next_step_allowed=(status == "PASS")
    )
    logging.info(f"Done. Schema OK rate: {schema_ok_rate:.2f}")

if __name__ == "__main__":
    main()
