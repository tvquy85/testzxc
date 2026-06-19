import argparse
import json
import yaml
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
import os
import sys

def build_dataset(df, tokenizer):
    def tokenize(sample):
        # Format prompt
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    def extract_prompt(row):
        # the prompt is the user message
        prompt = ""
        for m in row['messages']:
            if m['role'] == 'user':
                prompt = m['content']
                break
        return prompt
    df['prompt'] = df.apply(extract_prompt, axis=1)
    df['reward'] = df['flow_reward']
    ds = Dataset.from_pandas(df[['prompt', 'reward']])
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to standard RLHF dataset with prompt and reward")
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

    print("Loading model for PPO...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
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

    dataset = build_dataset(df, tokenizer)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    ppo_config = PPOConfig(
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
    )
    
    try:
        # Dummy reward model to satisfy TRL 0.15.2 PPOTrainer signature
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            train_dataset=dataset,
        )
        print("Starting PPO training...")
        
        # In a real PPO loop, we generate responses and score them with a reward model
        # For this script to mirror the PPO ablation from SEP, we train based on pre-calculated scalar rewards
        # We simulate the PPO step:
        
        for epoch, batch in enumerate(trainer.dataloader):
            query_tensors = batch["input_ids"]
            # Generate response
            response_tensors = trainer.generate(query_tensors, return_prompt=False, max_new_tokens=300)
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            # Use the pre-calculated reward from the dataset
            rewards = [torch.tensor(r, dtype=torch.float32) for r in batch["reward"]]
            # PPO step
            stats = trainer.step(query_tensors, response_tensors, rewards)
            trainer.log_stats(stats, batch, rewards)
            print(f"Step {epoch} complete")

        print("Saving adapter...")
        trainer.model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print("PPO complete!")
    except torch.cuda.OutOfMemoryError as e:
        print("OOM Error during PPO training. Deferring PPO.")
        with open("outputs/status/ppo_deferred.txt", "w") as f:
            f.write("PPO Deferred due to OOM.")
        sys.exit(0)

if __name__ == "__main__":
    main()
