import json
import argparse
import time
import os
import uuid
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.generate_rationales import (
    read_jsonl,
    parsed_records_from_raw,
    generate_transformers,
    raw_record,
    estimate_tokens_from_text,
    resolve_model_path,
    generation_config
)
from src.utils.artifacts import write_status, write_manifest

def derive_action(forecast: dict) -> str:
    try:
        score = (
            float(forecast.get("Strong Up", 0)) * 2 +
            float(forecast.get("Mild Up", 0)) * 1 +
            float(forecast.get("Neutral", 0)) * 0 +
            float(forecast.get("Mild Down", 0)) * -1 +
            float(forecast.get("Strong Down", 0)) * -2
        )
        if score >= 0.2:
            return "long"
        elif score <= -0.2:
            return "short"
        return "hold"
    except Exception:
        return "hold"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", default="data/rationales/raw/current_v6_train_qwen3_1000x3.jsonl")
    parser.add_argument("--contexts", default="data/processed/ticker_date_evidence_contexts_h1_v6_repaired.parquet")
    parser.add_argument("--prompt", default="prompts/rationale_generation_prompt_evidence_v6.txt")
    parser.add_argument("--config", default="configs/local_paths.yaml")
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--parsed-output", default="data/rationales/parsed/current_v6_train_qwen3_1000x3_repaired.parquet")
    parser.add_argument("--raw-output-repaired", default="data/rationales/raw/current_v6_train_qwen3_1000x3_repaired.jsonl")
    parser.add_argument("--status", default="outputs/status/06_5_POST_PROCESSING_AND_REPAIR_V6.status.json")
    
    # Generation args for repair
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-p", type=float, default=0.88)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--backend", default="transformers")
    parser.add_argument("--candidate-id-offset", type=int, default=0)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--stage", default="repair_v6")
    args = parser.parse_args()

    started = time.time()
    run_id = f"repair_v6_{uuid.uuid4().hex[:8]}"
    
    print("Reading raw input...")
    raw_rows = read_jsonl(args.raw_input)
    print(f"Total raw rows: {len(raw_rows)}")
    
    # Parse to check validity
    print("Parsing records to find failures...")
    parsed_records = parsed_records_from_raw(raw_rows, schema_version="v4")
    
    good_raw_records = []
    failed_items_info = [] # store dict with sample_id, candidate_id, prompt
    
    for raw_rec, parsed_rec in zip(raw_rows, parsed_records):
        failed = False
        if not parsed_rec["parse_ok"] or not parsed_rec["schema_ok"]:
            failed = True
        else:
            # Check empty news rationale despite N evidence
            context_meta = json.loads(parsed_rec.get("context_meta_json", "{}"))
            company_ev = context_meta.get("company_evidence_ids", [])
            has_N = any(eid.startswith("N") for eid in company_ev)
            parsed_json = json.loads(parsed_rec["parsed_json"])
            if has_N and len(parsed_json.get("news_rationale", [])) == 0:
                failed = True
                
        if failed:
            failed_items_info.append({
                "sample_id": parsed_rec["sample_id"],
                "candidate_id": parsed_rec["candidate_id"],
                "prompt": parsed_rec["prompt"],
            })
        else:
            good_raw_records.append(raw_rec)

    print(f"Found {len(failed_items_info)} failed candidates out of {len(raw_rows)}.")
    
    repaired_raw_records = []
    
    if failed_items_info:
        print("Loading contexts to rebuild generation items...")
        contexts_df = pd.read_parquet(args.contexts)
        
        # Build items for generation
        items_to_generate = []
        for info in failed_items_info:
            row_df = contexts_df[contexts_df["sample_id"] == info["sample_id"]]
            if len(row_df) == 0:
                print(f"WARNING: Sample ID {info['sample_id']} not found in contexts.")
                continue
            row = row_df.iloc[0].to_dict()
            items_to_generate.append({
                "sample_id": info["sample_id"],
                "candidate_id": info["candidate_id"],
                "ticker": row.get("ticker"),
                "timestamp": str(row.get("timestamp_utc", row.get("event_date"))),
                "horizon": "h1",
                "split": "train",
                "prompt": info["prompt"],
                "row": row,
                "prompt_tokens_est": estimate_tokens_from_text(info["prompt"])
            })
            
        print(f"Starting top-up generation for {len(items_to_generate)} items...")
        model_path, model_name = resolve_model_path(args)
        prompt_template = Path(args.prompt).read_text(encoding="utf-8")
        import hashlib
        prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
        
        generator = generate_transformers(model_path, items_to_generate, args)
        
        Path(args.raw_output_repaired).parent.mkdir(parents=True, exist_ok=True)
        with open(args.raw_output_repaired, "w", encoding="utf-8") as f:
            for item in generator:
                record = raw_record(item, args, str(model_path), model_name, prompt_hash, run_id)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                repaired_raw_records.append(record)
                
    else:
        print("No repairs needed.")
        
    print("Combining good and repaired records...")
    final_raw_records = good_raw_records + repaired_raw_records
    print(f"Total final raw records: {len(final_raw_records)}")
    
    print("Re-parsing final dataset and deriving action...")
    final_parsed_records = parsed_records_from_raw(final_raw_records, schema_version="v4")
    
    # Derive action
    for rec in final_parsed_records:
        if rec["parse_ok"] and rec["schema_ok"] and rec.get("parsed_json"):
            parsed_json = json.loads(rec["parsed_json"])
            forecast = parsed_json.get("forecast_distribution", {})
            derived_act = derive_action(forecast)
            parsed_json["action"] = derived_act
            # Update parsed_json string
            rec["parsed_json"] = json.dumps(parsed_json, ensure_ascii=False)
            
    Path(args.parsed_output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_parsed_records).to_parquet(args.parsed_output, index=False)
    
    parse_ok_rate = sum(1 for r in final_parsed_records if r["parse_ok"]) / max(1, len(final_parsed_records))
    schema_ok_rate = sum(1 for r in final_parsed_records if r["schema_ok"]) / max(1, len(final_parsed_records))
    
    metrics = {
        "total_rows": len(final_parsed_records),
        "repaired_rows": len(repaired_raw_records),
        "parse_ok_rate": parse_ok_rate,
        "schema_ok_rate": schema_ok_rate,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    
    status = "PASS" if parse_ok_rate >= 0.95 and schema_ok_rate >= 0.95 else "FAIL"
    write_status(
        args.status,
        "06_5_POST_PROCESSING_AND_REPAIR_V6",
        status,
        inputs_checked=[args.raw_input, args.contexts],
        outputs_created=[args.parsed_output, args.raw_output_repaired],
        metrics=metrics,
        failures=[],
        next_step_allowed=status == "PASS"
    )
    print("Done!")

if __name__ == "__main__":
    main()
