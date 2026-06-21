from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.parse_and_validate_rationale import parse_llm_json_strict, validate_rationale_schema_strict
from src.llm.render_context import render_context
from src.utils.artifacts import write_manifest, write_status
from src.utils.config import load_config


STEP = "09_RATIONALE_GENERATION_SCALEUP_TRAIN_ONLY"
MODEL_ALIASES = {
    "qwen3_4b": "main_explanation_llm",
    "qwen25_3b": "qwen25_judge",
    "deepseek_1_5b": "deepseek_reasoning_judge",
    "llama3_8b": "llama3_judge",
}


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


def resolve_snapshot(path_text: str | None) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return path_text
    refs_main = path / "refs" / "main"
    if refs_main.exists():
        snapshot = path / "snapshots" / refs_main.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return str(snapshot)
    snapshots = path / "snapshots"
    if snapshots.exists():
        candidates = sorted([p for p in snapshots.iterdir() if p.is_dir()])
        if candidates:
            return str(candidates[-1])
    return str(path)


def clip_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [TRUNCATED]"


def normalize_direction(value: Any) -> str:
    text = str(value or "").lower()
    if any(word in text for word in ["bearish", "negative", "down"]):
        return "negative"
    if any(word in text for word in ["bullish", "positive", "up"]):
        return "positive"
    return "neutral"


def normalize_strength(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"high", "strong"}:
        return "strong"
    if text in {"low", "weak"}:
        return "weak"
    return "medium"


def normalized_technical_context(row: Any, fallback_text: str, max_chars: int) -> str:
    tokens = row.get("technical_event_tokens_json", row.get("technical_event_tokens", None))
    parsed = None
    if isinstance(tokens, str) and tokens.strip():
        try:
            parsed = json.loads(tokens)
        except Exception:
            parsed = None
    elif isinstance(tokens, list):
        parsed = tokens
    if not isinstance(parsed, list):
        return clip_text(fallback_text, max_chars).replace("strength=high", "strength=strong")
    lines: list[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        token = str(item.get("token", "")).strip()
        if not token:
            continue
        direction = normalize_direction(item.get("direction_prior", item.get("direction")))
        strength = normalize_strength(item.get("strength"))
        value = item.get("value")
        value_text = f", value={float(value):.4g}" if isinstance(value, (int, float)) else ""
        rule = str(item.get("rule", "")).strip()
        rule_text = f", rule={rule}" if rule else ""
        lines.append(f"[{token}: direction={direction}, strength={strength}{value_text}{rule_text}]")
    return clip_text("\n".join(lines), max_chars) if lines else clip_text(fallback_text, max_chars)


def context_text(row: Any, max_input_tokens: int) -> str:
    if row.get("evidence_pack_json", None):
        from src.llm.render_context_evidence_v4 import render_evidence_context

        return clip_text(render_evidence_context(row), max_input_tokens * 4)
    ctx = render_context(row)
    # Keep the prompt small before tokenization. Body is least reliable and most verbose.
    body_chars = min(6000, max(1200, max_input_tokens * 2))
    token_chars = min(2400, max(800, max_input_tokens))
    technical_tokens = normalized_technical_context(row, ctx.get("technical_event_tokens", ""), token_chars)
    return "\n".join(
        [
            f"News Headline: {clip_text(ctx.get('headline', ''), 500)}",
            f"News Body: {clip_text(ctx.get('body', ''), body_chars)}",
            f"Market Regime: {clip_text(ctx.get('regime_label', 'normal_vol'), 120)}",
            "Technical Indicator Tokens:",
            technical_tokens,
        ]
    )


def render_prompt(row: Any, template: str, max_input_tokens: int) -> str:
    ctx = render_context(row)
    ctx["context"] = context_text(row, max_input_tokens)
    prompt = template
    for key, value in ctx.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt


def load_rows(input_path: str, splits_path: str | None, tokens_path: str | None, split: str, num_samples: int, seed: int):
    import pandas as pd

    samples = pd.read_parquet(input_path)
    if splits_path and Path(splits_path).exists():
        splits = pd.read_parquet(splits_path)
        if "split" in samples.columns:
            samples = samples.drop(columns=["split"])
        samples = samples.merge(splits[["sample_id", "split"]], on="sample_id", how="inner", validate="one_to_one")
    elif "split" not in samples.columns:
        samples["split"] = split
    df = samples[samples["split"] == split].copy()
    if tokens_path and Path(tokens_path).exists():
        tokens = pd.read_parquet(tokens_path)
        keep = [c for c in ["sample_id", "technical_event_tokens_json", "technical_event_tokens", "regime_label"] if c in tokens.columns]
        df = df.merge(tokens[keep], on="sample_id", how="left", suffixes=("", "_token"))
        if "regime_label_token" in df.columns:
            df["regime_label"] = df["regime_label"].fillna(df["regime_label_token"]) if "regime_label" in df.columns else df["regime_label_token"]
    if num_samples:
        df = df.sample(n=min(len(df), num_samples), random_state=seed)
    return df


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record.get("sample_id")), int(record.get("candidate_id", 0))


def estimate_tokens_from_text(text: str) -> int:
    return max(1, len(str(text)) // 4)


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "backend": args.backend,
        "max_input_tokens": args.max_input_tokens,
        "candidate_id_offset": args.candidate_id_offset,
        "attn_implementation": resolve_attn_implementation(getattr(args, "attn_implementation", "auto")),
    }


def build_items(rows: Any, prompt_template: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        prompt = render_prompt(row, prompt_template, args.max_input_tokens)
        for candidate_id in range(args.candidate_id_offset, args.candidate_id_offset + args.num_candidates):
            items.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "candidate_id": int(candidate_id),
                    "ticker": row.get("ticker"),
                    "timestamp": str(row.get("timestamp_utc", row.get("event_date"))),
                    "horizon": str(row.get("horizon", "h1")),
                    "split": args.split,
                    "prompt": prompt,
                    "row": row,
                    "prompt_tokens_est": estimate_tokens_from_text(prompt),
                }
            )
    if args.sort_by_length:
        items.sort(key=lambda item: item["prompt_tokens_est"])
    return items


def batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start : start + max(1, batch_size)]


def generate_transformers(model_path: str, items: list[dict[str, Any]], args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        attn_implementation=resolve_attn_implementation(args.attn_implementation),
    )
    model.eval()
    for batch in batched(items, args.batch_size):
        chat_texts = []
        for item in batch:
            messages = [
                {"role": "system", "content": "You generate concise financial JSON. Return JSON only."},
                {"role": "user", "content": item["prompt"]},
            ]
            chat_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_tokens)
        prompt_token_counts = inputs["attention_mask"].sum(dim=1).tolist()
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "repetition_penalty": args.repetition_penalty,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
            if args.top_k and args.top_k > 0:
                gen_kwargs["top_k"] = args.top_k
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        input_width = inputs["input_ids"].shape[-1]
        for item, prompt_tokens, generated_ids in zip(batch, prompt_token_counts, generated):
            output_ids = generated_ids[input_width:]
            raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            yield {
                **item,
            "raw_output": raw_output,
            "prompt_tokens_est": int(prompt_tokens),
            "output_tokens_est": len(tokenizer.encode(raw_output, add_special_tokens=False)),
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
        }


def generate_vllm(model_path: str, items: list[dict[str, Any]], args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if not args.vllm_base_url:
        raise RuntimeError("--vllm-base-url is required when --backend vllm")
    url = args.vllm_base_url.rstrip("/") + "/chat/completions"
    for item in items:
        payload = {
            "model": model_path,
            "messages": [
                {"role": "system", "content": "You generate concise financial JSON. Return JSON only."},
                {"role": "user", "content": item["prompt"]},
            ],
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        yield {**item, "raw_output": content.strip(), "output_tokens_est": estimate_tokens_from_text(content)}


def raw_record(item: dict[str, Any], args: argparse.Namespace, model_path: str, model_name: str, prompt_hash: str, run_id: str) -> dict[str, Any]:
    context_meta = {}
    if item.get("row") is not None and item["row"].get("evidence_pack_json", None):
        from src.llm.render_context_evidence_v4 import context_meta_from_row

        context_meta = context_meta_from_row(item["row"])
    return {
        "sample_id": item["sample_id"],
        "ticker": item.get("ticker"),
        "timestamp": item.get("timestamp"),
        "event_timestamp": item.get("timestamp"),
        "horizon": item.get("horizon", "h1"),
        "split": item.get("split"),
        "generator_model": model_name,
        "model_name": args.model,
        "model_path": model_path,
        "candidate_id": item["candidate_id"],
        "prompt_hash": prompt_hash,
        "prompt_tokens_est": item.get("prompt_tokens_est", 0),
        "output_tokens_est": item.get("output_tokens_est", 0),
        "prompt": item["prompt"],
        "raw_output": item["raw_output"],
        "raw_text": item["raw_output"],
        "context_meta_json": json.dumps(context_meta, ensure_ascii=False),
        "generation_config": generation_config(args),
        "run_id": run_id,
        "stage": args.stage,
    }


def parsed_records_from_raw(raw_rows: list[dict[str, Any]], schema_version: str = "auto") -> list[dict[str, Any]]:
    parsed_records: list[dict[str, Any]] = []
    for record in raw_rows:
        raw_text = record.get("raw_output", record.get("raw_text", ""))
        use_v4 = schema_version == "v4" or (schema_version == "auto" and (record.get("context_meta_json") not in {None, "", "{}"} or "v4" in str(record.get("stage", ""))))
        if use_v4:
            from src.llm.parse_and_validate_rationale_v4 import parse_llm_json_strict_v4, validate_rationale_schema_evidence_v4

            parsed = parse_llm_json_strict_v4(raw_text)
            context_meta = json.loads(record.get("context_meta_json") or "{}")
            schema_ok, parse_errors = validate_rationale_schema_evidence_v4(parsed, context_meta)
        else:
            parsed = parse_llm_json_strict(raw_text)
            schema_ok, parse_errors = validate_rationale_schema_strict(parsed)
        parse_ok = parsed is not None
        parsed_records.append(
            {
                "sample_id": record.get("sample_id"),
                "split": record.get("split"),
                "ticker": record.get("ticker"),
                "event_timestamp": record.get("event_timestamp", record.get("timestamp")),
                "candidate_id": int(record.get("candidate_id", 0)),
                "model_name": record.get("model_name", record.get("generator_model")),
                "model_path": record.get("model_path"),
                "prompt_hash": record.get("prompt_hash"),
                "prompt": record.get("prompt"),
                "generator_model": record.get("generator_model"),
                "horizon": record.get("horizon", "h1"),
                "prompt_tokens_est": record.get("prompt_tokens_est"),
                "output_tokens_est": record.get("output_tokens_est"),
                "generation_config": json.dumps(record.get("generation_config", {}), ensure_ascii=False),
                "run_id": record.get("run_id"),
                "stage": record.get("stage"),
                "raw_text": raw_text,
                "raw_output": raw_text,
                "schema_version": "v4" if use_v4 else "v3",
                "context_meta_json": record.get("context_meta_json", "{}"),
                "parse_ok": parse_ok,
                "schema_ok": bool(schema_ok),
                "parse_errors": parse_errors,
                "parsed_json": json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
            }
        )
    return parsed_records


def resolve_model_path(args: argparse.Namespace) -> tuple[str | None, str]:
    cfg = load_config(args.config)
    model_key = args.model_key or MODEL_ALIASES.get(args.model, args.model)
    model_path = resolve_snapshot(cfg.get("models", {}).get(model_key, args.model))
    model_name = Path(str(model_path)).name if model_path else args.model
    return model_path, model_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--samples", dest="input", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--splits", default="data/processed/split_membership.parquet")
    parser.add_argument("--tokens", default="data/indicators/technical_event_tokens_h1_v2.parquet")
    parser.add_argument("--prompt", default="prompts/rationale_generation_prompt.txt")
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", "--limit", dest="num_samples", type=int, default=1000)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--candidate-id-offset", type=int, default=0)
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--attn-implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager", "none", "default"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-raw-output", action="store_true")
    parser.add_argument("--raw-output", "--output", dest="raw_output", default="data/rationales/raw/train_candidates.jsonl")
    parser.add_argument("--parsed-output", default="data/rationales/parsed/train_candidates_strict.parquet")
    parser.add_argument("--stage", default="stage_0_sanity_check")
    parser.add_argument("--schema-version", choices=["auto", "v3", "v4"], default="auto")
    parser.add_argument("--min-parse-ok-rate", type=float, default=0.0)
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.0)
    parser.add_argument("--max-avg-output-tokens", type=float, default=100000.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    run_id = args.run_id or f"{args.stage}_{uuid.uuid4().hex[:8]}"
    started = time.time()
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    model_path, model_name = resolve_model_path(args)
    rows = load_rows(args.input, args.splits, args.tokens, args.split, args.num_samples, seed=42)
    items = build_items(rows, prompt_template, args)
    expected_keys = {record_key(item) for item in items}
    failures: list[str] = []
    raw_path = Path(args.raw_output)

    if rows.empty:
        failures.append(f"no {args.split} rows selected for generation")
    if not model_path or (args.backend == "transformers" and not Path(model_path).exists()):
        failures.append(f"model path missing for {args.model_key or args.model}: {model_path}")
    if raw_path.exists() and raw_path.stat().st_size > 0 and not args.resume and not args.overwrite:
        failures.append(f"raw output exists and --overwrite/--resume was not set: {raw_path}")

    if args.overwrite and raw_path.exists():
        raw_path.unlink()
    existing_raw = read_jsonl(raw_path)
    existing_keys = {record_key(record) for record in existing_raw} if args.resume else set()
    pending = [item for item in items if record_key(item) not in existing_keys]
    generated_count = 0

    if args.dry_run:
        failures.append("dry-run does not generate rationale outputs and cannot PASS Step 09")
    elif not failures and pending:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume and raw_path.exists() else "w"
        generator = generate_transformers(model_path, pending, args) if args.backend == "transformers" else generate_vllm(model_path, pending, args)
        with open(raw_path, mode, encoding="utf-8") as f:
            for item in generator:
                record = raw_record(item, args, str(model_path), model_name, prompt_hash, run_id)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated_count += 1
                if args.save_every and generated_count % args.save_every == 0:
                    f.flush()
                    os.fsync(f.fileno())

    raw_rows_all = read_jsonl(raw_path)
    current_raw_rows = [record for record in raw_rows_all if record_key(record) in expected_keys]
    parsed_records = parsed_records_from_raw(current_raw_rows, args.schema_version)
    Path(args.parsed_output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(parsed_records).to_parquet(args.parsed_output, index=False)

    if len(current_raw_rows) != len(parsed_records):
        failures.append("raw and parsed output row counts differ")
    if len(parsed_records) != len(items):
        failures.append(f"generated row count {len(parsed_records)} does not match expected {len(items)}")
    if parsed_records and {r["split"] for r in parsed_records} != {args.split}:
        failures.append(f"parsed output contains non-{args.split} split")
    parse_ok_rate = sum(1 for r in parsed_records if r["parse_ok"]) / max(1, len(parsed_records))
    schema_ok_rate = sum(1 for r in parsed_records if r["schema_ok"]) / max(1, len(parsed_records))
    avg_output_tokens = sum(int(r.get("output_tokens_est", 0) or 0) for r in current_raw_rows) / max(1, len(current_raw_rows))
    if not parsed_records:
        failures.append("no generated rows")
    if parse_ok_rate < args.min_parse_ok_rate:
        failures.append(f"parse_ok_rate {parse_ok_rate:.4f} < {args.min_parse_ok_rate:.4f}")
    if schema_ok_rate < args.min_schema_ok_rate:
        failures.append(f"schema_ok_rate {schema_ok_rate:.4f} < {args.min_schema_ok_rate:.4f}")
    if avg_output_tokens > args.max_avg_output_tokens:
        failures.append(f"avg_output_tokens_est {avg_output_tokens:.2f} > {args.max_avg_output_tokens:.2f}")

    status = "PASS" if not failures else "FAIL"
    write_manifest(args.manifest, [args.raw_output, args.parsed_output], STEP, run_id=run_id)
    metrics = {
        "row_count": len(parsed_records),
        "expected_row_count": len(items),
        "generated_this_run": generated_count,
        "unique_sample_count": int(rows["sample_id"].nunique()) if len(rows) else 0,
        "parse_ok_rate": parse_ok_rate,
        "schema_ok_rate": schema_ok_rate,
        "avg_output_tokens_est": avg_output_tokens,
        "run_id": run_id,
        "stage": args.stage,
        "backend": args.backend,
        "attn_implementation": resolve_attn_implementation(args.attn_implementation),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.input, args.splits, args.tokens, args.prompt, args.config],
        outputs_created=[args.raw_output, args.parsed_output, args.manifest, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
