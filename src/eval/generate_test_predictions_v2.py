from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment.train_rwsft_v2 import append_extra_site_packages, resolve_attn_implementation, resolve_model_path
from src.eval.forecast_prediction import (
    FORECAST_KEYS,
    format_forecast_context,
    parse_forecast_prediction,
    render_forecast_prompt,
)
from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "PREDICTION_SMOKE_V2"
def choose_checkpoint(primary: str, fallback: str | None, allow_fallback: bool = False) -> tuple[str, str]:
    primary_path = Path(primary)
    if (primary_path / "adapter_model.safetensors").exists() and (primary_path / "adapter_config.json").exists():
        return str(primary_path), "primary"
    if fallback and allow_fallback:
        fallback_path = Path(fallback)
        if (fallback_path / "adapter_model.safetensors").exists() and (fallback_path / "adapter_config.json").exists():
            return str(fallback_path), "fallback"
    raise RuntimeError(f"no adapter checkpoint found at primary={primary} fallback={fallback}")


def load_model(args: argparse.Namespace, model_path: str, checkpoint: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_attn_implementation(args.attn_implementation),
    )
    model = PeftModel.from_pretrained(base, checkpoint, local_files_only=True)
    model.eval()
    return tokenizer, model


def parse_prediction(raw_text: str) -> tuple[dict[str, float], str, str, bool, bool, list[str]]:
    return parse_forecast_prediction(raw_text)


def model_device(model: Any) -> Any:
    return getattr(model, "device", None) or next(model.parameters()).device


def generate_raw_outputs(tokenizer: Any, model: Any, prompts: list[str], args: argparse.Namespace, progress_event: str = "prediction_progress") -> list[str]:
    import json as json_module
    import torch

    raw_outputs: list[str] = []
    device = model_device(model)
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        messages = [
            [
                {"role": "system", "content": "You output compact valid JSON only."},
                {"role": "user", "content": prompt},
            ]
            for prompt in batch_prompts
        ]
        texts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            if getattr(tokenizer, "chat_template", None)
            else msg[-1]["content"]
            for msg in messages
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_width = int(encoded["input_ids"].shape[1])
        for output_ids in outputs:
            raw_outputs.append(tokenizer.decode(output_ids[input_width:], skip_special_tokens=True).strip())
        batch_index = start // args.batch_size + 1
        if args.progress_every > 0 and (batch_index % args.progress_every == 0 or start + args.batch_size >= len(prompts)):
            print(
                json_module.dumps(
                    {"event": progress_event, "completed": min(start + args.batch_size, len(prompts)), "total": len(prompts)}
                ),
                flush=True,
            )
    return raw_outputs


def prediction_quality_failures(out: Any, args: argparse.Namespace, checkpoint_used: str | None, checkpoint_source: str | None) -> list[str]:
    import pandas as pd

    failures: list[str] = []
    if len(out) == 0:
        failures.append("prediction output is empty")
        return failures
    if set(out["split"].dropna()) != {"test"}:
        failures.append("prediction output contains non-test rows")
    parse_ok_rate = float(out["parse_ok"].mean()) if "parse_ok" in out.columns else 0.0
    schema_ok_rate = float(out["schema_ok"].mean()) if "schema_ok" in out.columns else 0.0
    if parse_ok_rate < args.min_parse_ok_rate:
        failures.append(f"parse_ok_rate {parse_ok_rate:.4f} < {args.min_parse_ok_rate:.4f}")
    if schema_ok_rate < args.min_schema_ok_rate:
        failures.append(f"schema_ok_rate {schema_ok_rate:.4f} < {args.min_schema_ok_rate:.4f}")
    valid = out[out["schema_ok"].astype(bool)] if "schema_ok" in out.columns else out.iloc[0:0]
    valid_actions = set(valid["action"].dropna()) if "action" in valid.columns else set()
    if not valid_actions:
        failures.append("no schema-valid actions")
    elif valid_actions <= {"hold"}:
        failures.append("all schema-valid actions are hold")
    if checkpoint_source != "primary" and not args.allow_fallback:
        failures.append(f"prediction checkpoint source is not primary: {checkpoint_source}")
    normalized_checkpoint = str(checkpoint_used).replace("\\", "/").lower() if checkpoint_used else ""
    if checkpoint_used and "dpo" not in normalized_checkpoint and not args.allow_non_dpo_checkpoint:
        failures.append(f"prediction checkpoint is not dpo_v2: {checkpoint_used}")
    if "event_date" in out.columns and getattr(args, "min_trading_days", 0):
        trading_days = int(pd.to_datetime(out["event_date"]).dt.date.nunique())
        if trading_days < args.min_trading_days:
            failures.append(f"selected trading days {trading_days} < {args.min_trading_days}")
    return failures


def select_prediction_rows(samples: Any, tokens: Any, args: argparse.Namespace) -> Any:
    import pandas as pd

    if "split" not in samples.columns:
        raise ValueError("samples must contain split")
    df = samples[samples["split"] == args.split].copy()
    if not tokens.empty:
        keep = [col for col in ["sample_id", "regime_label", "technical_event_tokens_json"] if col in tokens.columns and (col == "sample_id" or col not in df.columns)]
        if keep:
            df = df.merge(tokens[keep], on="sample_id", how="left")
    df["event_date"] = pd.to_datetime(df["event_date"])
    if args.start_date:
        df = df[df["event_date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        df = df[df["event_date"] <= pd.Timestamp(args.end_date)].copy()
    df = df.sort_values(["event_date", "sample_id"]).copy()
    df["_event_day"] = df["event_date"].dt.date
    if args.max_days and args.max_days > 0:
        allowed_days = sorted(df["_event_day"].dropna().unique())[: args.max_days]
        df = df[df["_event_day"].isin(allowed_days)].copy()
    if args.max_rows_per_day and args.max_rows_per_day > 0:
        df = df.groupby("_event_day", sort=True, group_keys=False).head(args.max_rows_per_day).copy()
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    return df.drop(columns=["_event_day"], errors="ignore")


def apply_ablation_to_row(row: Any, mode: str) -> Any:
    if mode == "full":
        return row
    row = row.copy()
    if "clean_context_text" in row:
        text = str(row.get("clean_context_text") or "")
        if mode in {"no_news_body", "no_news"}:
            marker = "\nTechnical signals:"
            technical = text[text.find(marker) :] if marker in text else ""
            row["clean_context_text"] = (
                f"Ticker: {row.get('ticker', '')}\n"
                f"Date: {row.get('event_date', '')}\n\n"
                "Company-specific evidence:\nNone\n\n"
                "Context-only evidence:\nNone"
                f"{technical}"
            )
        if mode in {"no_technical_tokens", "no_technical"}:
            marker = "\nTechnical signals:"
            row["clean_context_text"] = (text[: text.find(marker)] if marker in text else text) + "\nTechnical signals:\nNone"
    if mode in {"no_news_body", "no_news"}:
        for col in ["headline", "body", "aggregated_headlines", "aggregated_body"]:
            if col in row:
                row[col] = ""
    if mode in {"no_technical_tokens", "no_technical"}:
        for col in ["technical_event_tokens_json", "technical_event_tokens"]:
            if col in row:
                row[col] = "[]"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--tokens", default="data/indicators/technical_event_tokens_h1_v2.parquet")
    parser.add_argument("--prompt", default="prompts/forecast_prediction_prompt_qwen3_json.txt")
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/aligned/qwen3_4b/dpo_v2")
    parser.add_argument("--fallback-checkpoint", default="checkpoints/aligned/qwen3_4b/rwsft_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--max-rows-per-day", type=int, default=None)
    parser.add_argument("--min-trading-days", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--body-chars", type=int, default=1200)
    parser.add_argument("--token-chars", type=int, default=1200)
    parser.add_argument("--ablation-mode", default="full", choices=["full", "no_news_body", "no_news", "no_technical_tokens", "no_technical"])
    parser.add_argument("--min-parse-ok-rate", type=float, default=0.80)
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.80)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--allow-non-dpo-checkpoint", action="store_true")
    parser.add_argument("--attn-implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager", "none", "default"])
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--extra-site-packages", action="append", default=[])
    parser.add_argument("--output", default="outputs/predictions/test_predictions_qwen3_dpo_smoke.parquet")
    parser.add_argument("--metrics", default="outputs/metrics/test_predictions_qwen3_dpo_smoke.json")
    parser.add_argument("--status", default="outputs/status/PREDICTION_SMOKE_V2.status.json")
    parser.add_argument("--manifest", default="outputs/manifests/PREDICTION_SMOKE_V2.manifest.json")
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    append_extra_site_packages(args.extra_site_packages)

    import pandas as pd
    import torch

    failures: list[str] = []
    samples = pd.read_parquet(args.samples)
    tokens = pd.read_parquet(args.tokens) if Path(args.tokens).exists() else pd.DataFrame()
    if args.split != "test":
        failures.append("prediction smoke must use test split")
    df = select_prediction_rows(samples, tokens, args)
    if df.empty:
        failures.append("no prediction samples selected")
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")
    model_path = resolve_model_path(args.model, args.model_path, args.config)
    checkpoint_used = None
    checkpoint_source = None
    if not failures:
        try:
            checkpoint_used, checkpoint_source = choose_checkpoint(args.checkpoint, args.fallback_checkpoint, args.allow_fallback)
        except Exception as exc:
            failures.append(str(exc))

    rows: list[dict[str, Any]] = []
    attn_implementation_used = None
    if not failures:
        tokenizer, model = load_model(args, model_path, str(checkpoint_used))
        attn_implementation_used = getattr(model.config, "_attn_implementation", None)
        prompts = [
            render_forecast_prompt(
                prompt_template,
                format_forecast_context(apply_ablation_to_row(row, args.ablation_mode), body_chars=args.body_chars, token_chars=args.token_chars),
            )
            for _, row in df.iterrows()
        ]
        run_id = f"test_prediction_smoke_{checkpoint_source}_{args.ablation_mode}_{len(df)}"
        raw_outputs = generate_raw_outputs(tokenizer, model, prompts, args)
        for offset, raw_text in enumerate(raw_outputs):
            dist, action, pred_label, parse_ok, schema_ok, parse_errors = parse_prediction(raw_text)
            source_row = df.iloc[offset]
            rows.append(
                {
                    "sample_id": source_row["sample_id"],
                    "split": source_row["split"],
                    "ticker": source_row.get("ticker"),
                    "event_date": str(source_row.get("event_date")),
                    "pred_label": pred_label,
                    "action": action,
                    "p_strong_down": dist["strong_down"],
                    "p_mild_down": dist["mild_down"],
                    "p_neutral": dist["neutral"],
                    "p_mild_up": dist["mild_up"],
                    "p_strong_up": dist["strong_up"],
                    "model_checkpoint": str(checkpoint_used),
                    "checkpoint_source": checkpoint_source,
                    "run_id": run_id,
                    "parse_ok": parse_ok,
                    "schema_ok": schema_ok,
                    "parse_errors": json.dumps(parse_errors),
                    "raw_output": raw_text,
                }
            )
        del model
        torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    metrics = {
        "rows": int(len(out)),
        "split": args.split,
        "checkpoint_used": checkpoint_used,
        "checkpoint_source": checkpoint_source,
        "parse_ok_rate": float(out["parse_ok"].mean()) if len(out) else 0.0,
        "schema_ok_rate": float(out["schema_ok"].mean()) if len(out) else 0.0,
        "fallback_used": checkpoint_source == "fallback",
        "temperature": 0.0,
        "do_sample": False,
        "attn_implementation": attn_implementation_used,
        "schema_ok_rows": int(out["schema_ok"].sum()) if len(out) and "schema_ok" in out.columns else 0,
        "parse_ok_rows": int(out["parse_ok"].sum()) if len(out) and "parse_ok" in out.columns else 0,
        "action_distribution": out["action"].value_counts(dropna=False).to_dict() if len(out) and "action" in out.columns else {},
        "selected_rows": int(len(df)),
        "selected_trading_days": int(df["event_date"].dt.date.nunique()) if len(df) and "event_date" in df.columns else 0,
        "selected_start_date": str(df["event_date"].min().date()) if len(df) and "event_date" in df.columns else None,
        "selected_end_date": str(df["event_date"].max().date()) if len(df) and "event_date" in df.columns else None,
        "max_days": args.max_days,
        "max_rows_per_day": args.max_rows_per_day,
        "ablation_mode": args.ablation_mode,
    }
    failures.extend(prediction_quality_failures(out, args, checkpoint_used, checkpoint_source))
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.samples, args.tokens, args.prompt, args.checkpoint],
        [args.output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
