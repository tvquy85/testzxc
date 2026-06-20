import argparse
import logging
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.info(f"Loading base model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    
    # Load model in bfloat16 to merge
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Merge on CPU or GPU if enough RAM
        trust_remote_code=True,
        local_files_only=True
    )

    logging.info(f"Loading adapter {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    
    logging.info("Merging and unloading...")
    model = model.merge_and_unload()
    
    logging.info(f"Saving merged model to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logging.info("Done.")

if __name__ == "__main__":
    main()
