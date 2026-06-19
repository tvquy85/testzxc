import argparse
import json
import yaml
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os

class WeightedDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        weights = [ex.pop("weight") for ex in examples if "weight" in ex]
        for ex in examples:
            ex.pop("text", None) # Remove raw string which causes tensor conversion error
        batch = super().__call__(examples)
        if weights:
            batch["weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch

class WeightedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("weight", None)
        
        # Super compute_loss will call model(**inputs) and return the scalar loss
        if num_items_in_batch is not None:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        else:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            
        if weights is not None:
            # Since batch_size is usually 1, we can just multiply by the weight.
            # If batch_size > 1, the model returns the average loss over the batch.
            # This is an approximation if batch_size > 1, but exact for batch_size = 1.
            loss = loss * weights.mean()
            
        return (loss, outputs) if return_outputs else loss

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
    
    # We need to apply chat template to messages
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    def format_fn(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)
        
    df['text'] = df.apply(format_fn, axis=1)
    dataset = Dataset.from_pandas(df[['text', 'weight']])

    # Load model with standard bfloat16 instead of 4-bit due to Windows BitsAndBytes compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    sft_config = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        num_train_epochs=1,
        fp16=True,
        gradient_checkpointing=True,
        max_seq_length=2048,
        dataset_text_field="text",
        remove_unused_columns=False, # Keep 'weight'
        logging_steps=5,
        save_strategy="no"
    )
    
    trainer = WeightedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=WeightedDataCollator(tokenizer, mlm=False)
    )

    print("Starting RWSFT training...")
    trainer.train()
    
    print("Saving adapter...")
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("RWSFT complete!")

if __name__ == "__main__":
    main()
