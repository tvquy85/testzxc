from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.parse_and_validate_rationale_v4 import FORECAST_CANONICAL, TITLE_TO_CANONICAL, forecast_distribution_errors, parse_llm_json_strict_v4
from src.llm.render_context_evidence_v4 import render_evidence_context
from src.utils.artifacts import write_json, write_manifest, write_status
from src.utils.config import load_config

STEP = "14_INDEPENDENT_JUDGE_RERUN_EVIDENCE_V4"
TITLE_LABELS = ["Strong Down", "Mild Down", "Neutral", "Mild Up", "Strong Up"]
REVERSED_TITLE_LABELS = list(reversed(TITLE_LABELS))

PROMPT_TEMPLATE = """You are an independent financial forecast judge.

Task:
Given the evidence context and an analyst rationale, infer a probability distribution for the next-trading-day stock movement.

Rules:
- Return valid JSON only.
- Do not include markdown.
- Do not include explanations outside JSON.
- Do not use any realized return, hidden label, or external information.
- Judge from the context and rationale content only.
- Do not copy the analyst's forecast distribution or action.
- Probabilities must be numeric and sum exactly to 1.00.

Output schema:
{{
  "forecast_distribution": {{
{schema_lines}
  }}
}}

Evidence context:
{context}

Analyst rationale:
News: {news_rationale}
Technical: {technical_rationale}
Conflict: {conflict_resolution}
Risk: {risk_note}
"""


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


def resolve_model_path(config_path: str, model_key: str, hf_home: str | None) -> str:
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    cfg = load_config(config_path)
    raw = str(cfg.get("models", {}).get(model_key, model_key))
    if hf_home:
        raw = raw.replace("$HF_HOME", hf_home)
    else:
        raw = raw.replace("$HF_HOME", os.environ.get("HF_HOME", "E:/huggingface"))
    return raw


def clip_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [TRUNCATED]"


def parse_rationale(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def compact_claims(items: Any, keys: list[str]) -> str:
    if not isinstance(items, list) or not items:
        return "None"
    parts: list[str] = []
    for item in items[:2]:
        if isinstance(item, dict):
            bits = [f"{key}={item.get(key)}" for key in keys if item.get(key) not in {None, ""}]
            parts.append("; ".join(bits))
        else:
            parts.append(str(item))
    return " | ".join(parts)


def schema_lines_for_order(order: list[str]) -> str:
    return ",\n".join(f'    "{label}": 0.0' for label in order)


def build_prompt(row: pd.Series, label_order: str, max_context_chars: int) -> str:
    parsed = parse_rationale(row.get("parsed_json"))
    order = REVERSED_TITLE_LABELS if label_order == "reversed" else TITLE_LABELS
    context = row.get("clean_context_text")
    if not isinstance(context, str) or not context.strip():
        context = render_evidence_context(row)
    return PROMPT_TEMPLATE.format(
        schema_lines=schema_lines_for_order(order),
        context=clip_text(context, max_context_chars),
        news_rationale=compact_claims(parsed.get("news_rationale"), ["evidence_id", "factor", "direction", "strength"]),
        technical_rationale=compact_claims(parsed.get("technical_rationale"), ["signal_id", "signal", "direction", "strength"]),
        conflict_resolution=clip_text(parsed.get("conflict_resolution", "None"), 300),
        risk_note=clip_text(parsed.get("risk_note", "None"), 160),
    )


def batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start : start + max(1, batch_size)]


def generate_outputs(model_path: str, items: list[dict[str, Any]], args: argparse.Namespace) -> Iterable[dict[str, Any]]:
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
                {"role": "system", "content": "You are a deterministic financial judge. Return JSON only."},
                {"role": "user", "content": item["prompt"]},
            ]
            chat_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_tokens)
        prompt_token_counts = inputs["attention_mask"].sum(dim=1).tolist()
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        input_width = inputs["input_ids"].shape[-1]
        for item, prompt_tokens, generated_ids in zip(batch, prompt_token_counts, generated):
            output_ids = generated_ids[input_width:]
            raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            yield {
                **item,
                "raw_judge_output": raw,
                "prompt_tokens_est": int(prompt_tokens),
                "output_tokens_est": len(tokenizer.encode(raw, add_special_tokens=False)),
                "attn_implementation": getattr(model.config, "_attn_implementation", None),
            }


def parse_judge_output(raw: str) -> dict[str, Any]:
    parsed = parse_llm_json_strict_v4(raw)
    result = {
        "judge_parse_ok": parsed is not None,
        "judge_schema_ok": False,
        **{f"p_{key}": 0.0 for key in FORECAST_CANONICAL},
        "parse_errors": [],
    }
    if parsed is None:
        result["parse_errors"] = ["invalid_json"]
        return result
    errors = forecast_distribution_errors(parsed.get("forecast_distribution"))
    if errors:
        result["parse_errors"] = errors
        return result
    dist = parsed["forecast_distribution"]
    mapped: dict[str, float] = {}
    for key, value in dist.items():
        canonical = TITLE_TO_CANONICAL.get(str(key))
        if canonical:
            mapped[canonical] = float(value)
    total = sum(mapped.values()) or 1.0
    result["judge_schema_ok"] = True
    for key in FORECAST_CANONICAL:
        result[f"p_{key}"] = float(mapped[key] / total)
    return result


def canonical_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "strong_down": "strong_down",
        "mild_down": "mild_down",
        "neutral": "neutral",
        "mild_up": "mild_up",
        "strong_up": "strong_up",
    }
    return aliases.get(text, "neutral")


def aggregate_variants(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sample_id, candidate_id), group in df.groupby(["sample_id", "candidate_id"], sort=False):
        base = group.iloc[0].to_dict()
        probs = {key: pd.to_numeric(group[f"p_{key}"], errors="coerce").fillna(0.0).tolist() for key in FORECAST_CANONICAL}
        mean_probs = {key: float(sum(values) / max(1, len(values))) for key, values in probs.items()}
        argmaxes = []
        for _, row in group.iterrows():
            values = [float(row[f"p_{key}"]) for key in FORECAST_CANONICAL]
            argmaxes.append(FORECAST_CANONICAL[int(max(range(len(values)), key=lambda idx: values[idx]))])
        argmax_consistency = max(argmaxes.count(label) for label in set(argmaxes)) / max(1, len(argmaxes)) if argmaxes else 0.0
        normal = group[group["label_order"].eq("normal")]
        reversed_rows = group[group["label_order"].eq("reversed")]
        l1_delta = 0.0
        if len(normal) and len(reversed_rows):
            n = normal.iloc[0]
            r = reversed_rows.iloc[0]
            l1_delta = float(sum(abs(float(n[f"p_{key}"]) - float(r[f"p_{key}"])) for key in FORECAST_CANONICAL))
        target = canonical_label(base.get("target_label_5"))
        row_out = {
            "sample_id": sample_id,
            "candidate_id": int(candidate_id),
            "split": base.get("split"),
            "track": base.get("track"),
            "target_label_5": target,
            "judge_model": base.get("judge_model"),
            "judge_parse_ok": bool(group["judge_parse_ok"].all()),
            "judge_schema_ok": bool(group["judge_schema_ok"].all()),
            "argmax_consistency": float(argmax_consistency),
            "l1_probability_delta": float(l1_delta),
            "true_label_probability": float(mean_probs[target]),
            "true_label_probability_debiased": float(mean_probs[target]),
            "raw_judge_outputs_json": json.dumps(group[["label_order", "raw_judge_output", "parse_errors"]].to_dict("records"), ensure_ascii=False),
        }
        for key in FORECAST_CANONICAL:
            row_out[f"p_{key}"] = mean_probs[key]
            row_out[f"p_{key}_debiased"] = mean_probs[key]
        rows.append(row_out)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default="outputs/manifests/14_INDEPENDENT_JUDGE_RERUN_EVIDENCE_V4.manifest.json")
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--hf-home", default="E:/huggingface")
    parser.add_argument("--judge-model-key", default="qwen3_judge")
    parser.add_argument("--limit", "--num-samples", dest="limit", type=int, default=0)
    parser.add_argument("--label-orders", default="normal,reversed")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-chars", type=int, default=3200)
    parser.add_argument("--attn-implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager", "none", "default"])
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.95)
    args = parser.parse_args()

    started = time.time()
    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if Path(args.rationales).exists() else pd.DataFrame()
    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    if args.limit and args.limit > 0:
        rationales = rationales.head(args.limit).copy()
    if rationales.empty:
        failures.append(f"rationales missing or empty: {args.rationales}")
    if contexts.empty:
        failures.append(f"contexts missing or empty: {args.contexts}")
    context_cols = [
        col
        for col in [
            "sample_id",
            "split",
            "target_label_5",
            "target_return",
            "track",
            "evidence_pack_json",
            "technical_event_tokens_json",
            "clean_context_text",
        ]
        if col in contexts.columns
    ]
    merged = rationales.merge(contexts[context_cols], on="sample_id", how="inner", suffixes=("", "_context"))
    if merged.empty and not failures:
        failures.append("rationales and contexts do not join on sample_id")
    if "split_context" in merged.columns:
        merged["split"] = merged["split_context"].combine_first(merged.get("split"))

    label_orders = [item.strip() for item in args.label_orders.split(",") if item.strip()]
    items: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        for label_order in label_orders:
            prompt = build_prompt(row, label_order, args.max_context_chars)
            items.append(
                {
                    "sample_id": row["sample_id"],
                    "candidate_id": int(row.get("candidate_id", 0)),
                    "split": row.get("split"),
                    "track": row.get("track"),
                    "target_label_5": row.get("target_label_5"),
                    "label_order": label_order,
                    "prompt": prompt,
                    "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "judge_model": args.judge_model_key,
                }
            )

    model_path = resolve_model_path(args.config, args.judge_model_key, args.hf_home)
    if model_path and not Path(model_path).exists():
        failures.append(f"judge model path missing: {model_path}")
    variant_rows: list[dict[str, Any]] = []
    if not failures:
        for item in generate_outputs(model_path, items, args):
            parsed = parse_judge_output(item["raw_judge_output"])
            variant_rows.append({**item, **parsed})
    variant_df = pd.DataFrame(variant_rows)
    out = aggregate_variants(variant_df) if len(variant_df) else pd.DataFrame()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    schema_ok_rate = float(out["judge_schema_ok"].mean()) if len(out) else 0.0
    parse_ok_rate = float(out["judge_parse_ok"].mean()) if len(out) else 0.0
    metrics = {
        "judged_samples": int(len(out)),
        "variant_rows": int(len(variant_df)),
        "judge_parse_ok_rate": parse_ok_rate,
        "judge_schema_ok_rate": schema_ok_rate,
        "mean_true_label_probability": float(out["true_label_probability"].mean()) if len(out) else 0.0,
        "random_baseline": 0.20,
        "mean_argmax_consistency": float(out["argmax_consistency"].mean()) if len(out) else 0.0,
        "mean_l1_probability_delta": float(out["l1_probability_delta"].mean()) if len(out) else 0.0,
        "inferability_claim_allowed": bool(float(out["true_label_probability"].mean()) > 0.22) if len(out) else False,
        "label_orders": label_orders,
        "judge_model_path": model_path,
        "elapsed_seconds": round(time.time() - started, 3),
        "by_track": out.groupby("track")["true_label_probability"].mean().to_dict() if len(out) and "track" in out else {},
    }
    if len(out) == 0:
        failures.append("independent judge output is empty")
    if schema_ok_rate < args.min_schema_ok_rate:
        failures.append(f"judge_schema_ok_rate {schema_ok_rate:.4f} < {args.min_schema_ok_rate:.4f}")
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.rationales, args.contexts, args.config],
        outputs_created=[args.output, args.metrics, args.manifest, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
