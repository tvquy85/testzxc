from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment.train_rwsft_v2 import (
    QWEN3_TARGET_MODULES,
    append_extra_site_packages,
    import_training_stack,
    resolve_attn_implementation,
    resolve_model_path,
)
from src.utils.artifacts import sha256_file, write_json, write_manifest, write_status


STEP = "15_ALIGNMENT_TRAINING_REPRODUCIBLE"


def load_dpo_records(path: str, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            if row.get("prompt") and row.get("chosen") and row.get("rejected"):
                records.append(row)
            if len(records) >= limit:
                break
    return records


def trim_prompt(prompt: str, max_chars: int) -> str:
    prompt = str(prompt)
    if len(prompt) <= max_chars:
        return prompt
    return prompt[-max_chars:]


def render_pair(tokenizer: Any, prompt: str, answer: str) -> tuple[str, str]:
    user = {"role": "user", "content": prompt}
    assistant = {"role": "assistant", "content": str(answer)}
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
    else:
        prompt_text = f"user: {prompt}\nassistant:"
        full_text = f"{prompt_text} {answer}"
    return prompt_text, full_text


def sequence_logps(model: Any, tokenizer: Any, prompt_texts: list[str], full_texts: list[str], max_length: int) -> Any:
    import torch

    encoded = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    for row_idx, prompt_text in enumerate(prompt_texts):
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[row_idx, :prompt_len] = -100
    outputs = model(**encoded)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def run_dpo_smoke(args: argparse.Namespace, model_path: str) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    records = load_dpo_records(args.train, limit=max(args.max_steps * args.batch_size * 2, 8))
    if not records:
        raise RuntimeError("no train split DPO records with prompt/chosen/rejected found")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for QLoRA DPO smoke training")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_attn_implementation(args.attn_implementation),
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=QWEN3_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)

    losses: list[float] = []
    for step in range(args.max_steps):
        batch = [records[(step * args.batch_size + i) % len(records)] for i in range(args.batch_size)]
        chosen_prompts: list[str] = []
        chosen_texts: list[str] = []
        rejected_prompts: list[str] = []
        rejected_texts: list[str] = []
        for row in batch:
            prompt = trim_prompt(str(row["prompt"]), args.max_user_chars)
            chosen_prompt, chosen_text = render_pair(tokenizer, prompt, str(row["chosen"]))
            rejected_prompt, rejected_text = render_pair(tokenizer, prompt, str(row["rejected"]))
            chosen_prompts.append(chosen_prompt)
            chosen_texts.append(chosen_text)
            rejected_prompts.append(rejected_prompt)
            rejected_texts.append(rejected_text)
        chosen_logps = sequence_logps(model, tokenizer, chosen_prompts, chosen_texts, args.max_seq_length)
        rejected_logps = sequence_logps(model, tokenizer, rejected_prompts, rejected_texts, args.max_seq_length)
        loss = -torch.nn.functional.logsigmoid(args.beta * (chosen_logps - rejected_logps)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    adapter_path = Path(args.output_dir) / "adapter_config.json"
    adapter_model = Path(args.output_dir) / "adapter_model.safetensors"
    if not adapter_path.exists() or not adapter_model.exists():
        raise RuntimeError("DPO adapter artifacts were not produced")

    finite_losses = [value for value in losses if math.isfinite(value)]
    if len(finite_losses) != len(losses):
        raise RuntimeError("DPO smoke produced non-finite loss")
    return {
        "dpo_losses": losses,
        "dpo_loss_first": losses[0] if losses else None,
        "dpo_loss_last": losses[-1] if losses else None,
        "dpo_train_records_loaded": len(records),
        "dpo_adapter_config": str(adapter_path),
        "dpo_cuda_device": torch.cuda.get_device_name(0),
        "dpo_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "dpo_attn_implementation": getattr(model.config, "_attn_implementation", None),
    }


def read_previous_rwsft_status(path: str) -> dict[str, Any]:
    status_path = Path(path)
    if not status_path.exists():
        return {}
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        return {
            "rwsft_status_before_dpo": data.get("status"),
            "rwsft_adapter_config": metrics.get("adapter_config"),
            "rwsft_smoke_train_executed": metrics.get("smoke_train_executed"),
            "rwsft_loss_first": metrics.get("loss_first"),
            "rwsft_loss_last": metrics.get("loss_last"),
            "rwsft_max_memory_allocated_bytes": metrics.get("max_memory_allocated_bytes"),
        }
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/alignment/dpo_pairs_train_v2.jsonl")
    parser.add_argument("--rwsft-checkpoint", default="checkpoints/aligned/qwen3_4b/rwsft_v2")
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--extra-site-packages", action="append", default=[])
    parser.add_argument("--output-dir", default="checkpoints/aligned/qwen3_4b/dpo_v2")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--max-user-chars", type=int, default=3500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--attn-implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager", "none", "default"])
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    append_extra_site_packages(args.extra_site_packages)

    failures: list[str] = []
    module_inventory, import_failures = import_training_stack()
    failures.extend(f"training module import failed: {item}" for item in import_failures)
    if not Path(args.train).exists() or Path(args.train).stat().st_size == 0:
        failures.append(f"DPO training dataset missing or empty: {args.train}")
    rwsft_checkpoint = Path(args.rwsft_checkpoint)
    if not (rwsft_checkpoint / "adapter_model.safetensors").exists():
        failures.append(f"RWSFT adapter missing: {rwsft_checkpoint / 'adapter_model.safetensors'}")

    model_path = None
    if not failures:
        try:
            model_path = resolve_model_path(args.model, args.model_path, args.config)
            if not Path(model_path).exists():
                failures.append(f"model path missing: {model_path}")
        except Exception as exc:
            failures.append(f"model path resolution failed: {type(exc).__name__}: {str(exc)[:200]}")

    dpo_metrics: dict[str, Any] = {}
    if not failures:
        try:
            dpo_metrics = run_dpo_smoke(args, str(model_path))
        except Exception as exc:
            failures.append(f"DPO smoke training failed: {type(exc).__name__}: {str(exc)[:500]}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model,
        "model_path": model_path,
        "train": args.train,
        "dataset_sha256": sha256_file(args.train) if Path(args.train).exists() else None,
        "rwsft_checkpoint": args.rwsft_checkpoint,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "beta": args.beta,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "load_in_4bit": True,
        "attn_implementation": resolve_attn_implementation(args.attn_implementation),
        "extra_site_packages": args.extra_site_packages,
    }
    config_path = str(Path(args.output_dir) / "trainer_config.json")
    write_json(config_path, config)
    artifact_paths = [config_path]
    for name in ["adapter_config.json", "adapter_model.safetensors"]:
        path = Path(args.output_dir) / name
        if path.exists():
            artifact_paths.append(str(path))
    rwsft_artifacts = [
        str(rwsft_checkpoint / "adapter_config.json"),
        str(rwsft_checkpoint / "adapter_model.safetensors"),
    ]
    artifact_paths.extend([p for p in rwsft_artifacts if Path(p).exists()])
    write_manifest(args.manifest, artifact_paths, STEP)

    rwsft_status = read_previous_rwsft_status(args.status)
    metrics = {
        "module_inventory": module_inventory,
        "missing_modules": [k for k, v in module_inventory.items() if v.get("status") != "available"],
        "rwsft_smoke_pass": (rwsft_checkpoint / "adapter_model.safetensors").exists(),
        "dpo_smoke_pass": bool(dpo_metrics) and (Path(args.output_dir) / "adapter_model.safetensors").exists(),
        "max_steps": args.max_steps,
        "dpo_smoke_train_executed": bool(dpo_metrics),
        **rwsft_status,
        **dpo_metrics,
    }
    if not metrics["rwsft_smoke_pass"]:
        failures.append("RWSFT smoke adapter missing")
    if not metrics["dpo_smoke_pass"]:
        failures.append("DPO smoke adapter missing")
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.train, args.rwsft_checkpoint, args.config],
        artifact_paths + [args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
