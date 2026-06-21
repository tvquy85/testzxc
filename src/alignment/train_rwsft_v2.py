from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import sha256_file, write_json, write_manifest, write_status
from src.utils.config import load_config


STEP = "15_ALIGNMENT_TRAINING_REPRODUCIBLE"
REQUIRED_MODULES = ["torch", "transformers", "peft", "trl", "bitsandbytes"]
QWEN3_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def append_extra_site_packages(paths: list[str]) -> None:
    for raw_path in paths:
        if not raw_path:
            continue
        path = str(Path(raw_path)) if not (":" in raw_path[:3]) else raw_path
        if path not in sys.path:
            sys.path.append(path)


def import_training_stack() -> tuple[dict[str, dict[str, str | None]], list[str]]:
    inventory: dict[str, dict[str, str | None]] = {}
    failures: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            inventory[name] = {
                "status": "available",
                "version": str(getattr(module, "__version__", "UNKNOWN")),
                "path": str(getattr(module, "__file__", None)),
            }
        except Exception as exc:
            inventory[name] = {
                "status": "missing_or_import_error",
                "version": None,
                "path": None,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
            failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
    return inventory, failures


def resolve_attn_implementation(value: str | None) -> str | None:
    if not value or value == "auto":
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except Exception:
            return None
    if value in {"none", "default"}:
        return None
    return value


def resolve_model_path(model: str, model_path: str | None, config_path: str) -> str:
    if model_path:
        return model_path
    if any(sep in model for sep in ("/", "\\")) or Path(model).exists():
        return model
    cfg = load_config(config_path)
    models = cfg.get("models", {})
    alias_map = {
        "qwen3_4b": "main_explanation_llm",
        "main_explanation_llm": "main_explanation_llm",
        "qwen3_judge": "qwen3_judge",
    }
    key = alias_map.get(model, model)
    value = models.get(key)
    if not value:
        raise RuntimeError(f"Cannot resolve model alias: {model}")
    hf_home = os.environ.get("HF_HOME")
    resolved = str(value)
    if hf_home:
        resolved = resolved.replace("$HF_HOME", hf_home)
    return resolved


def load_jsonl_records(path: str, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            if isinstance(row.get("messages"), list) and len(row["messages"]) >= 2:
                records.append(row)
            if len(records) >= limit:
                break
    return records


def compact_messages(messages: list[dict[str, str]], max_user_chars: int) -> list[dict[str, str]]:
    user = dict(messages[0])
    assistant = dict(messages[-1])
    content = str(user.get("content", ""))
    if len(content) > max_user_chars:
        user["content"] = content[-max_user_chars:]
    return [user, assistant]


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def run_smoke_train(args: argparse.Namespace, model_path: str) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    records = load_jsonl_records(args.train, limit=max(args.max_steps * args.batch_size * 2, 8))
    if not records:
        raise RuntimeError("no train split records with messages found")

    device_available = torch.cuda.is_available()
    if not device_available:
        raise RuntimeError("CUDA is not available for QLoRA smoke training")

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
        attn_implementation=resolve_attn_implementation(getattr(args, "attn_implementation", "auto")),
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

    texts = [
        render_chat(tokenizer, compact_messages(row["messages"], args.max_user_chars))
        for row in records
    ]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    losses: list[float] = []
    for step in range(args.max_steps):
        batch_texts = [texts[(step * args.batch_size + i) % len(texts)] for i in range(args.batch_size)]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        output = model(**encoded, labels=labels)
        loss = output.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    adapter_path = Path(args.output_dir) / "adapter_config.json"
    if not adapter_path.exists():
        raise RuntimeError("adapter_config.json was not produced")

    return {
        "losses": losses,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "train_records_loaded": len(records),
        "adapter_config": str(adapter_path),
        "cuda_device": torch.cuda.get_device_name(0),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/alignment/rwsft_train_v2.jsonl")
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--extra-site-packages", action="append", default=[])
    parser.add_argument("--output-dir", default="checkpoints/aligned/qwen3_4b/rwsft_v2")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--max-user-chars", type=int, default=3500)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--attn-implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager", "none", "default"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--metrics", default="outputs/metrics/14_rwsft_train_medium.json")
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
        failures.append(f"training dataset missing or empty: {args.train}")
    if args.dry_run:
        failures.append("dry-run cannot satisfy adapter output acceptance")

    model_path = None
    smoke_metrics: dict[str, Any] = {}
    if not failures:
        try:
            model_path = resolve_model_path(args.model, args.model_path, args.config)
            if not Path(model_path).exists():
                failures.append(f"model path missing: {model_path}")
        except Exception as exc:
            failures.append(f"model path resolution failed: {type(exc).__name__}: {str(exc)[:200]}")

    if not failures:
        try:
            smoke_metrics = run_smoke_train(args, str(model_path))
        except Exception as exc:
            failures.append(f"smoke training failed: {type(exc).__name__}: {str(exc)[:500]}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model,
        "model_path": model_path,
        "train": args.train,
        "dataset_sha256": sha256_file(args.train) if Path(args.train).exists() else None,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "load_in_4bit": True,
        "attn_implementation": resolve_attn_implementation(args.attn_implementation),
        "extra_site_packages": args.extra_site_packages,
    }
    config_path = str(Path(args.output_dir) / "trainer_config.json")
    write_json(config_path, config)
    artifact_paths = [config_path]
    adapter_config = Path(args.output_dir) / "adapter_config.json"
    adapter_model = Path(args.output_dir) / "adapter_model.safetensors"
    if adapter_config.exists():
        artifact_paths.append(str(adapter_config))
    if adapter_model.exists():
        artifact_paths.append(str(adapter_model))
    metrics = {
        "module_inventory": module_inventory,
        "missing_modules": [k for k, v in module_inventory.items() if v.get("status") != "available"],
        "max_steps": args.max_steps,
        "smoke_train_executed": bool(smoke_metrics),
        **smoke_metrics,
    }
    write_json(args.metrics, metrics)
    artifact_paths.append(args.metrics)
    write_manifest(args.manifest, artifact_paths, STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.train, args.config],
        artifact_paths + [args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
