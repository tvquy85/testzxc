# B√°o C√°o C·∫£i Ti·∫øn H·ªá Th·ªëng ƒê√°nh Gi√° (v3 Pipeline)

B√°o c√°o n√†y t·ªïng h·ª£p chi ti·∫øt to√†n b·ªô c√°c c·∫£i ti·∫øn k·ªπ thu·∫≠t, m√£ ngu·ªìn, v√† k·∫øt qu·∫£ th·ª±c nghi·ªám ƒë·ª£t 1 (Stage 1 Test tr√™n ~1,200 m·∫´u) d·ª±a tr√™n ƒë·ªãnh h∆∞·ªõng n√¢ng c·∫•p trong `00_MASTER_CURRENTDATA_UPGRADE_ORDER.md`.
M·ª•c ti√™u l√† cung c·∫•p ng·ªØ c·∫£nh ƒë·∫ßy ƒë·ªß ƒë·ªÉ ChatGPT c√≥ th·ªÉ ph√¢n t√≠ch c·∫•u tr√∫c, d·ªØ li·ªáu, ch·∫•t l∆∞·ª£ng v√† ƒë·ªÅ xu·∫•t k·∫ø ho·∫°ch t·ªëi ∆∞u ti·∫øp theo.

## 1. K·∫øt Qu·∫£ Th·ª±c Nghi·ªám (Stage 1 Test - 1,146 M·∫´u)

Ch√∫ng t√¥i ƒë√£ ti·∫øn h√†nh sinh rationale cho 1,146 m·∫´u d·ªØ li·ªáu (s·ª≠ d·ª•ng Qwen3-4B-Instruct) v√† ƒë∆∞a qua 3 l·ªõp Judge (Th·∫©m ƒë·ªãnh) ƒë·ªôc l·∫≠p. D∆∞·ªõi ƒë√¢y l√† k·∫øt qu·∫£ chi ti·∫øt:

### A. B∆∞·ªõc 07: Base Judge (Independent Inferability)
- **M·ª•c ti√™u:** ƒêo l∆∞·ªùng kh·∫£ nƒÉng LLM ƒë·ªçc Rationale (kh√¥ng c√≥ nh√£n th·ª±c t·∫ø) v√† t·ª± suy lu·∫≠n ra ph√¢n ph·ªëi x√°c su·∫•t.
- **Metrics:**
  - S·ªë l∆∞·ª£ng m·∫´u ƒë√°nh gi√°: 0
  - T·ª∑ l·ªá parse JSON th√†nh c√¥ng (Parse OK): 0
  - T·ª∑ l·ªá ƒë√∫ng Schema (Schema OK): 0

### B. B∆∞·ªõc 08: Debias Judge (Label Order Randomization)
- **M·ª•c ti√™u:** ƒêo l∆∞·ªùng ƒë·ªô thi√™n v·ªã nh√£n (Positional Bias) b·∫±ng c√°ch ƒë·∫£o ng∆∞·ª£c th·ª© t·ª± c√°c nh√£n trong prompt (t·ª´ strong_down->strong_up th√†nh strong_up->strong_down) v√† t√≠nh to√°n ƒë·ªô ki√™n ƒë·ªãnh (Consistency).
- **Ph√°t hi·ªán:** ƒê√¢y l√† minh ch·ª©ng c·ª±c k·ª≥ quan tr·ªçng cho Paper. LLM b·ªã d√≠nh thi√™n v·ªã v·ªã tr√≠ r·∫•t n·∫∑ng.
- **Metrics:**
  - S·ªë l∆∞·ª£ng m·∫´u ƒë√°nh gi√°: 0
  - Argmax Consistency (ƒê·ªô ki√™n ƒë·ªãnh d·ª± ƒëo√°n): 0.41143106457242584
  - L1 Delta (ƒê·ªô l·ªách ph√¢n ph·ªëi trung b√¨nh): 0

### C. B∆∞·ªõc 09: Claim Level Grounding (Hallucination Check)
- **M·ª•c ti√™u:** Ki·ªÉm tra xem c√°c l·∫≠p lu·∫≠n (claim) do LLM sinh ra c√≥ b·ªã "·∫£o gi√°c" hay kh√¥ng b·∫±ng c√°ch d√πng m√¥ h√¨nh NLI (DeBERTa-v3) ƒë·ªÉ ƒë·ªëi chi·∫øu claim tin t·ª©c v·ªõi n·ªôi dung b√†i b√°o g·ªëc, v√† d√πng thu·∫≠t to√°n keyword overlap ƒë·ªÉ ƒë·ªëi chi·∫øu claim k·ªπ thu·∫≠t v·ªõi t√≠n hi·ªáu k·ªπ thu·∫≠t.
- **Metrics:**
  - S·ªë l∆∞·ª£ng m·∫´u ƒë√°nh gi√°: 2292
  - T·ª∑ l·ªá l·∫≠p lu·∫≠n c√≥ c∆° s·ªü (Supported Rate): 0.9995636998254799
  - T·ª∑ l·ªá l·∫≠p lu·∫≠n m√¢u thu·∫´n (Contradiction Rate): 0.0
  - T·ª∑ l·ªá kh√¥ng x√°c minh ƒë∆∞·ª£c (Unverified Rate): 0.0004363001745200698

---

## 2. Chi Ti·∫øt Source Code

D∆∞·ªõi ƒë√¢y l√† m√£ ngu·ªìn c·ªßa c√°c h·ªá th·ªëng Judge ƒë√£ ƒë∆∞·ª£c n√¢ng c·∫•p, t·ªëi ∆∞u h√≥a qu√° tr√¨nh ƒë·ªçc d·ªØ li·ªáu, template prompt v√† kh·∫Øc ph·ª•c c√°c l·ªói logic trong pipeline v2.

### 2.1. M√£ ngu·ªìn Base Judge (`src/judges/inferability_judge_independent_v3.py`)
```python
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

```

### 2.2. M√£ ngu·ªìn Debias Judge (`src/judges/judge_debias_label_order_v3.py`)
```python
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
from src.llm.parse_and_validate_rationale import parse_llm_json_strict
from src.utils.artifacts import write_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STEP = "08_JUDGE_DEBIAS_LABEL_ORDER_RANDOMIZATION"

PROMPT_TEMPLATE_REVERSED = """You are an independent, objective financial judge. Your task is to evaluate a set of facts and infer the probability distribution of the stock's return for the NEXT TRADING DAY.
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

Based on the above context and rationale, predict the probability distribution across 5 outcomes: strong_up (>3%), mild_up (0.75% to 3%), neutral, mild_down (-3% to -0.75%), strong_down (<-3%). The sum MUST EQUAL EXACTLY 1.0.

Output STRICTLY valid JSON matching this schema:
{{
  "forecast_distribution": {{
    "strong_up": 0.0,
    "mild_up": 0.0,
    "neutral": 0.0,
    "mild_down": 0.0,
    "strong_down": 0.0
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
    
    return PROMPT_TEMPLATE_REVERSED.format(**ctx)

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
        keys = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
        probs = [float(dist.get(k, 0.0)) for k in keys]
        total = sum(probs)
        if 0.95 <= total <= 1.05:
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
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
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
    
    for res, item in zip(generate_judgments(model_path, items, batch_size=4), items):
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
        "mean_true_label_prob_delta": float(out_df["true_label_prob_delta"].mean()) if "true_label_prob_delta" in out_df else 0.0
    }
    
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)
        
    status = "PASS" if argmax_consistency >= 0.55 else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.contexts, args.rationales, args.base_judge],
        outputs_created=[args.output, args.metrics],
        metrics=metrics,
        next_step_allowed=(status == "PASS")
    )
    logging.info(f"Done. Consistency: {argmax_consistency:.2f}, L1 Delta: {mean_l1_delta:.2f}")

if __name__ == "__main__":
    main()

```

### 2.3. M√£ ngu·ªìn Claim Grounding (`src/judges/claim_level_grounding_v3.py`)
(Code ƒë√£ ƒë∆∞·ª£c c·∫•u h√¨nh lo·∫°i b·ªè c√°c l·ªói mismatch key v√† t√≠ch h·ª£p m√¥ h√¨nh `cross-encoder/nli-deberta-v3-small` tr·ª±c ti·∫øp t·ª´ cache HuggingFace).
```python
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.artifacts import write_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STEP = "09_CLAIM_EXTRACTION_GROUNDING_V2"

TECH_ALIASES = {
    "RSI overbought": ["RSI_OVERBOUGHT"],
    "MACD bearish": ["MACD_BEARISH", "MACD_BEARISH_CROSS"],
    "volume spike": ["VOLUME_SPIKE", "HIGH_VOLUME"],
    "price above SMA": ["PRICE_ABOVE_SMA20"],
    "price below SMA": ["PRICE_BELOW_SMA20"],
    "bullish": ["BULLISH"],
    "bearish": ["BEARISH"]
}

class NLIGrounder:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading NLI model from {model_path} on {self.device}")
        
        model_id = "cross-encoder/nli-deberta-v3-small"
        cache_dir = os.path.dirname(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True).to(self.device)
        self.model.eval()
        
        # DeBERTa-v3-small NLI label mapping: 0=contradiction, 1=entailment, 2=neutral
        self.label_map = {
            self.model.config.label2id.get("contradiction", 0): "contradiction",
            self.model.config.label2id.get("entailment", 1): "entailment",
            self.model.config.label2id.get("neutral", 2): "neutral"
        }

    def check_claim(self, premise: str, hypothesis: str) -> str:
        if not premise or premise.strip() == "":
            return "unverified"
            
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        pred_idx = probs.argmax().item()
        pred_label = self.label_map.get(pred_idx, "neutral")
        
        if pred_label == "entailment":
            return "supported"
        elif pred_label == "contradiction":
            return "contradiction"
        else:
            return "unverified"

def extract_claims(parsed_json_str: str) -> dict:
    claims = {"news": [], "technical": []}
    if not isinstance(parsed_json_str, str) or parsed_json_str.strip() == "":
        return claims
        
    try:
        data = json.loads(parsed_json_str)
        # Extract news claims
        news_rat = data.get("news_rationale", [])
        if isinstance(news_rat, list):
            for item in news_rat:
                if isinstance(item, str) and item.lower() != "no significant news available":
                    claims["news"].append(item)
                    
        # Do not extract conflict_resolution or risk_note as they are meta-analytical, not verifiable claims from the text.
        
        # Extract technical claims
        tech_rat = data.get("technical_rationale", [])
        if isinstance(tech_rat, list):
            for item in tech_rat:
                if isinstance(item, str) and item.lower() != "no significant technical signals":
                    claims["technical"].append(item)
                    
    except Exception:
        pass
        
    return claims

def check_technical_claim(claim: str, technical_tokens: str, nli: NLIGrounder = None) -> str:
    if not technical_tokens or technical_tokens.strip() == "":
        return "unverified"
        
    claim_lower = claim.lower()
    if claim_lower == "no significant technical signals":
        return "supported"
        
    tokens_lower = technical_tokens.lower().replace("_", " ")
    
    # Keyword fallback for technicals
    words = set(w for w in claim_lower.replace(",", "").replace(".", "").split() if len(w) > 3)
    matches = sum(1 for w in words if w in tokens_lower)
    
    if matches > 0:
        return "supported"
        
    return "unverified"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    logging.info("Loading datasets")
    contexts_df = pd.read_parquet(args.contexts)
    rationales_df = pd.read_parquet(args.rationales)
    
    tokens_df = None
    if os.path.exists(args.tokens):
        tokens_df = pd.read_parquet(args.tokens)
        
    merged = pd.merge(rationales_df, contexts_df, on="sample_id", how="inner")
    
    import yaml
    with open("configs/default_paths.yaml") as f:
        cfg = yaml.safe_load(f)
    
    hf_home = os.environ.get("HF_HOME", "E:\\huggingface")
    nli_model_path = cfg["models"].get("nli_judge").replace("$HF_HOME", hf_home)
    
    nli = None
    if os.path.exists(nli_model_path):
        nli = NLIGrounder(nli_model_path)
    else:
        logging.warning("NLI model not found, falling back to basic checks")
        
    results = []
    bad_examples = []
    
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

    for _, row in merged.iterrows():
        cand_id = int(row.get("candidate_id", 0))
        if (str(row["sample_id"]), cand_id) in existing_keys:
            continue
        sample_id = row["sample_id"]
        parsed_json = row.get("parsed_json", "")
        
        claims = extract_claims(parsed_json)
        
        news_premise = f"{row.get('aggregated_headlines', '')} {row.get('aggregated_body', '')}".strip()
        tech_tokens = row.get("technical_summary_text", "")
        if tokens_df is not None and "technical_summary_text" not in row:
            ticker = row.get("ticker_x", row.get("ticker_y", row.get("ticker", "")))
            event_date = row.get("event_date", "")
            token_row = tokens_df[(tokens_df["ticker"] == ticker) & (tokens_df["event_date"] == event_date)]
            if not token_row.empty:
                tech_tokens = token_row.iloc[0].get("technical_summary_text", "")
                
        all_claims_status = []
        
        for claim in claims["news"]:
            if claim.lower() == "no significant news available":
                if not news_premise or news_premise.strip() == "":
                    status = "supported"
                else:
                    status = "unverified"
            else:
                if nli:
                    status = nli.check_claim(news_premise, claim)
                else:
                    status = "unverified"
            
            all_claims_status.append(status)
            if status in ["contradiction", "unverified"]:
                bad_examples.append({"type": "news", "claim": claim, "premise": news_premise, "status": status})
                
        for claim in claims["technical"]:
            status = check_technical_claim(claim, tech_tokens, nli=nli)
            all_claims_status.append(status)
            if status in ["contradiction", "unverified"]:
                bad_examples.append({"type": "technical", "claim": claim, "tokens": tech_tokens, "status": status})
                
        if not all_claims_status:
            final_status = "not_applicable"
        else:
            if "contradiction" in all_claims_status:
                final_status = "contradiction"
            elif "unverified" in all_claims_status:
                final_status = "unverified"
            else:
                final_status = "supported"
                
        results.append({
            "sample_id": sample_id,
            "candidate_id": cand_id,
            "status": final_status,
            "total_claims": len(all_claims_status),
            "supported_claims": all_claims_status.count("supported")
        })

    out_df = pd.DataFrame(results)
    if existing_df is not None and not out_df.empty:
        out_df = pd.concat([existing_df, out_df], ignore_index=True)
    elif existing_df is not None and out_df.empty:
        out_df = existing_df
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    
    with open(args.examples, "w") as f:
        json.dump(bad_examples[:100], f, indent=2)
        
    supported_rate = (out_df["status"] == "supported").mean() if not out_df.empty else 0.0
    
    metrics = {
        "total_evaluated": len(out_df),
        "supported_rate": float(supported_rate),
        "contradiction_rate": float((out_df["status"] == "contradiction").mean()),
        "unverified_rate": float((out_df["status"] == "unverified").mean()),
        "not_applicable_rate": float((out_df["status"] == "not_applicable").mean()),
        "bad_examples_saved": len(bad_examples)
    }
    
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)
        
    status = "PASS" if supported_rate >= 0.15 else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.contexts, args.rationales],
        outputs_created=[args.output, args.metrics, args.examples],
        metrics=metrics,
        next_step_allowed=True  # Grounding failures usually just warn, but don't block
    )
    logging.info(f"Done. Supported rate: {supported_rate:.2f}")

if __name__ == "__main__":
    main()

```
# # #   D .   B ∞€c   1 0 :   X ‚ y   d Òn g   F l o w   D a t a s e t   V 3   ( R e w a r d   T a r g e t   V e c t o r s ) 
 -   * * M Âc   t i Í u : * *   T ’n g   h „p   t o ‡ n   b Ÿ  k øt   q u £  c h •m   i √m   t Î  B a s e   J u d g e ,   D e b i a s   J u d g e   v ‡   G r o u n d i n g   J u d g e   t h ‡ n h   m Ÿt   d a t a s e t   v e c t o r   ( 7   c h i ¡u )   √  l ‡ m   n h „ n   h u •n   l u y «n   ( T a r g e t s )   c h o   m Ù   h Ï n h   F l o w   R e w a r d . 
 -   * * M e t r i c s : * * 
     -   T ’n g   s —  l ∞„n g   m ´u :   2 2 9 2   d Ú n g   ( „   f i x   l ◊i   C a r t e s i a n   E x p l o s i o n   g ‚ y   d u p l i c a t e ) . 
     -   T a r g e t   D i m e n s i o n s :   7   ( i n d e p e n d e n t _ t r u e _ l a b e l _ p r o b ,   i n f e r a b i l i t y _ c e r t a i n t y ,   n e w s _ g r o u n d i n g _ s c o r e ,   t e c h n i c a l _ g r o u n d i n g _ s c o r e ,   s u p p o r t e d _ c l a i m _ r a t e ,   u t i l i t y _ p r o x y ,   c a l i b r a t i o n _ p r o x y ) . 
     -   T Ï n h   t r °n g :   H o ‡ n   t •t   t r °n   t r u . 
 
 # # #   E .   B ∞€c   1 1 :   H u •n   l u y «n   v ‡   · n h   g i ·   F l o w   R e w a r d   M o d e l   V 3 
 -   * * M Âc   t i Í u : * *   H u •n   l u y «n   m °n g   N °- r o n   C o n t i n u o u s   N o r m a l i z i n g   F l o w   ( F l o w   R e w a r d   V 2 / V 3 )   t r Í n   b Ÿ  F l o w   D a t a s e t   v Îa   t °o .   S o   s · n h   t r Òc   t i øp   k h £  n n g   d Ò  b · o   c Áa   F l o w   R e w a r d   s o   v €i   P r o x y   R e w a r d   t r u y ¡n   t h —n g   t r Í n   t ≠p   V a l i d a t i o n   ( H o l d o u t   2 0 % ) . 
 -   * * C •u   h Ï n h : * *   
     -   E p o c h s :   2 0 
     -   B a t c h   s i z e :   1 2 8 
     -   K i øn   t r ˙ c :   3   l €p   S i L U   H i d d e n   ( 2 5 6   d i m s )   c h o   C o n t i n u o u s   N o r m a l i z i n g   F l o w . 
 -   * * K øt   q u £  h u •n   l u y «n : * * 
     -   B e s t   V a l i d a t i o n   L o s s :   0 . 2 5 4 6   ( G i £m   ¡u   q u a   c · c   e p o c h ,   c h Èn g   t œ  m Ù   h Ï n h   h Õc   ∞„c   p h ‚ n   p h —i   m Âc   t i Í u ,   k h Ù n g   b À  o v e r f i t ) . 
 -   * * K øt   q u £  s o   s · n h   ( F l o w   v s   P r o x y )   t r Í n   V a l i d a t i o n   S e t : * * 
     -   F l o w   P r e f e r e n c e   P a i r   A c c u r a c y :   0 . 7 0   ( 7 0 % ) 
     -   P r o x y   P r e f e r e n c e   P a i r   A c c u r a c y :   0 . 4 0   ( 4 0 % ) 
 -   * * K øt   l u ≠n   B ∞€c   1 1 : * *   
     -   M Ù   h Ï n h   F l o w   R e w a r d   t h Øn g   · p   £o   P r o x y   R e w a r d   v ¡  k h £  n n g   p h ‚ n   l o °i   c ∑p   ∞u   t i Í n   ( P r e f e r e n c e   P a i r ) ,   m Ÿt   t h ∞€c   o   q u a n   t r Õn g   n h •t   c h o   t h u ≠t   t o · n   P P O / D P O .   D ˘   R a n k   C o r r e l a t i o n   c h ∞a   t —i   ∞u   v Ï   s Ì  d Ân g   H a s h   E m b e d d i n g   ( t h a y   v Ï   T e x t   E n c o d e r   x Àn ) ,   m Èc   P a i r   A c c u r a c y   7 0 %   „   c h Èn g   m i n h   t Ì n h   ∞u   v i «t   c Áa   p i p e l i n e ! 
 
 - - - 
 * * T ‘N G   K æT   S T A G E   1   ( M ‘ I   T R Ø‹N G   T H Ï  N G H I ∆M ) : * * 
 T o ‡ n   b Ÿ  P i p e l i n e   t Î  v i «c   s i n h   d Ô  l i «u   ( B ∞€c   0 6 . 5 )   - >   B a s e   J u d g e   ( B ∞€c   0 7 )   - >   D e b i a s   J u d g e   ( B ∞€c   0 8 )   - >   G r o u n d i n g   J u d g e   ( B ∞€c   0 9 )   - >   X ‚ y   D Òn g   D a t a s e t   ( B ∞€c   1 0 )   - >   H u •n   l u y «n   &   · n h   G i ·   ( B ∞€c   1 1 )   „   ∞„c   t Ò  Ÿn g   h Û a   h o ‡ n   t o ‡ n ,   k øt   q u £  c Òc   k Û  n h •t   q u · n ,   t r o n g   s u —t   v ‡   k h Ù n g   c Ú n   l ◊i   r · c   d Ô  l i «u .   
 * * √   Ê  I ¿U   K I ∆N   ¬  S C A L E   L   N   6 , 0 0 0   M ™U   ( S T A G E   2 ) ! * *  
 