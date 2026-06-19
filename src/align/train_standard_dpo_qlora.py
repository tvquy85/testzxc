import argparse
import json
import yaml
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        paths = yaml.safe_load(f)
    model_name = paths.get('models', {}).get('main_explanation_llm')

    # Load dataset
    df = pd.read_json(args.train, lines=True)
    if args.limit:
        df = df.head(args.limit)
        
    dataset = Dataset.from_pandas(df[['prompt', 'chosen', 'rejected']])

    print("Loading model for DPO...")
    try:
        # Load model in bfloat16 (1.5B model fits easily in 24GB VRAM without 4-bit)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # The reference model for DPO
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    dpo_config = DPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        num_train_epochs=1,
        fp16=True,
        gradient_checkpointing=True,
        max_prompt_length=1024,
        max_length=2048,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False
    )
    
    try:
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
        print("Starting DPO training...")
        trainer.train()
        print("Saving adapter...")
        trainer.model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print("DPO complete!")
    except torch.cuda.OutOfMemoryError as e:
        print("OOM Error during DPO training. Deferring DPO.")
        with open("outputs/status/dpo_deferred.txt", "w") as f:
            f.write("DPO Deferred due to OOM.")
        sys.exit(0)

if __name__ == "__main__":
    main()
