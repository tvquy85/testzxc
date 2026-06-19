from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignment.train_rwsft_v2 import append_extra_site_packages, resolve_model_path
from src.eval.forecast_prediction import (
    clip_text,
    forecast_score,
    parse_forecast_prediction,
    render_forecast_prompt,
)
from src.eval.generate_test_predictions_v2 import generate_raw_outputs, load_model
from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "17_COUNTERFACTUAL_DIRECTIONAL_EVAL_V2"
POSITIVE_TERMS = {
    "beat",
    "beats",
    "raised",
    "raises",
    "growth",
    "profit",
    "profits",
    "record",
    "upgrade",
    "strong",
    "approval",
    "win",
    "surge",
    "higher",
    "outperform",
}
NEGATIVE_TERMS = {
    "miss",
    "missed",
    "lawsuit",
    "decline",
    "loss",
    "losses",
    "warning",
    "downgrade",
    "weak",
    "cut",
    "investigation",
    "bankruptcy",
    "lower",
    "fell",
    "plunge",
    "recall",
}
STRENGTH_WEIGHT = {
    "weak": 0.25,
    "low": 0.25,
    "medium": 0.5,
    "strong": 1.0,
    "high": 1.0,
}


def parse_tokens(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def count_terms(text: str, terms: set[str]) -> int:
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE)) for term in terms)


def score_tokens(tokens: list[Any]) -> float:
    score = 0.0
    for token in tokens:
        name = str(token.get("token", "")) if isinstance(token, dict) else str(token)
        upper_name = name.upper()
        prior = str(token.get("direction_prior", "")).lower() if isinstance(token, dict) else ""
        strength = str(token.get("strength", "medium")).lower() if isinstance(token, dict) else "medium"
        weight = STRENGTH_WEIGHT.get(strength, 0.5)
        if prior in {"positive", "bullish", "up"} or "BULLISH" in upper_name or "OVERSOLD" in upper_name or "BOLLINGER_LOWER" in upper_name:
            score += weight
        elif prior in {"negative", "bearish", "down"} or "BEARISH" in upper_name or "OVERBOUGHT" in upper_name or "BOLLINGER_UPPER" in upper_name:
            score -= weight
        if "HIGH_VOL" in upper_name or "VOLATILITY_SPIKE" in upper_name:
            score += 0.4
    return score


def score_context(headline: Any, body: Any, tokens_json: Any) -> float:
    text = f"{headline or ''} {body or ''}"
    news_score = 0.35 * count_terms(text, POSITIVE_TERMS) - 0.35 * count_terms(text, NEGATIVE_TERMS)
    tech_score = score_tokens(parse_tokens(tokens_json))
    return float(news_score + tech_score)


def pass_expected_direction(expected: str, original_score: float, counterfactual_score: float) -> tuple[bool, bool, float]:
    delta = counterfactual_score - original_score
    no_change = abs(delta) < 1e-9
    if expected == "less_negative":
        return delta > 0, no_change, delta
    if expected == "less_positive":
        return delta < 0, no_change, delta
    if expected in {"toward_neutral", "lower_extreme"}:
        return abs(counterfactual_score) <= abs(original_score) and not no_change, no_change, delta
    return False, no_change, delta


def evaluate_tasks(tasks: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        expected = str(task.get("expected_direction") or "")
        original_score = score_context(
            task.get("original_headline", task.get("headline")),
            task.get("original_body", task.get("body")),
            task.get("original_technical_event_tokens_json", task.get("technical_event_tokens_json")),
        )
        counterfactual_score = score_context(
            task.get("counterfactual_headline", task.get("headline")),
            task.get("counterfactual_body", task.get("body")),
            task.get("counterfactual_technical_event_tokens_json", task.get("technical_event_tokens_json")),
        )
        passed, no_change, delta = pass_expected_direction(expected, original_score, counterfactual_score)
        rows.append(
            {
                "sample_id": task.get("sample_id"),
                "split": task.get("split"),
                "counterfactual_type": task.get("counterfactual_type"),
                "expected_direction": expected,
                "original_score": original_score,
                "counterfactual_score": counterfactual_score,
                "delta": delta,
                "directional_pass": passed,
                "no_change": no_change,
            }
        )
    num_tasks = len(rows)
    pass_count = sum(1 for row in rows if row["directional_pass"])
    no_change_count = sum(1 for row in rows if row["no_change"])
    wrong_count = num_tasks - pass_count - no_change_count
    metrics = {
        "num_tasks": num_tasks,
        "mean_delta": float(sum(row["delta"] for row in rows) / num_tasks) if num_tasks else None,
        "pass_rate": float(pass_count / num_tasks) if num_tasks else None,
        "wrong_direction_rate": float(wrong_count / num_tasks) if num_tasks else None,
        "no_change_rate": float(no_change_count / num_tasks) if num_tasks else None,
        "temperature": 0.0,
        "do_sample": False,
        "evaluator": "deterministic_context_directional_smoke",
    }
    return metrics, rows


def format_task_context(task: dict[str, Any], variant: str) -> str:
    prefix = "original" if variant == "original" else "counterfactual"
    return (
        f"Ticker: {clip_text(task.get('ticker', ''), 32)}\n"
        f"Event Date: {clip_text(task.get('event_date', ''), 64)}\n"
        f"News Headline: {clip_text(task.get(f'{prefix}_headline', ''), 360)}\n"
        f"News Body: {clip_text(task.get(f'{prefix}_body', ''), 1200)}\n"
        f"Market Regime: not_provided\n"
        f"Technical Indicator Tokens:\n{clip_text(task.get(f'{prefix}_technical_event_tokens_json', '[]'), 1200)}"
    )


def evaluate_tasks_with_llm(tasks: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    prompt_template = Path(args.prompt).read_text(encoding="utf-8")
    model_path = resolve_model_path(args.model, args.model_path, args.config)
    tokenizer, model = load_model(args, model_path, args.checkpoint)
    attn_implementation = getattr(model.config, "_attn_implementation", None)
    prompts: list[str] = []
    prompt_meta: list[tuple[int, str]] = []
    for idx, task in enumerate(tasks):
        for variant in ["original", "counterfactual"]:
            prompts.append(render_forecast_prompt(prompt_template, format_task_context(task, variant)))
            prompt_meta.append((idx, variant))
    raw_outputs = generate_raw_outputs(tokenizer, model, prompts, args, progress_event="counterfactual_prediction_progress")
    predictions: dict[tuple[int, str], dict[str, Any]] = {}
    parse_ok_count = 0
    schema_ok_count = 0
    for meta, raw_text in zip(prompt_meta, raw_outputs):
        dist, action, pred_label, parse_ok, schema_ok, parse_errors = parse_forecast_prediction(raw_text)
        parse_ok_count += int(parse_ok)
        schema_ok_count += int(schema_ok)
        predictions[meta] = {
            "dist": dist,
            "action": action,
            "pred_label": pred_label,
            "parse_ok": parse_ok,
            "schema_ok": schema_ok,
            "parse_errors": parse_errors,
            "raw_output": raw_text,
            "score": forecast_score(dist),
        }
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        original = predictions[(idx, "original")]
        counterfactual = predictions[(idx, "counterfactual")]
        expected = str(task.get("expected_direction") or "")
        passed, no_change, delta = pass_expected_direction(expected, float(original["score"]), float(counterfactual["score"]))
        both_schema_ok = bool(original["schema_ok"]) and bool(counterfactual["schema_ok"])
        rows.append(
            {
                "sample_id": task.get("sample_id"),
                "split": task.get("split"),
                "counterfactual_type": task.get("counterfactual_type"),
                "expected_direction": expected,
                "original_score": original["score"],
                "counterfactual_score": counterfactual["score"],
                "delta": delta,
                "directional_pass": bool(passed and both_schema_ok),
                "no_change": no_change,
                "original_action": original["action"],
                "counterfactual_action": counterfactual["action"],
                "original_schema_ok": original["schema_ok"],
                "counterfactual_schema_ok": counterfactual["schema_ok"],
                "original_raw_output": original["raw_output"],
                "counterfactual_raw_output": counterfactual["raw_output"],
            }
        )
    num_rows = len(rows)
    pass_count = sum(1 for row in rows if row["directional_pass"])
    no_change_count = sum(1 for row in rows if row["no_change"])
    wrong_count = num_rows - pass_count - no_change_count
    total_generations = max(1, len(raw_outputs))
    metrics = {
        "num_tasks": num_rows,
        "num_generations": len(raw_outputs),
        "mean_delta": float(sum(row["delta"] for row in rows) / num_rows) if num_rows else None,
        "pass_rate": float(pass_count / num_rows) if num_rows else None,
        "wrong_direction_rate": float(wrong_count / num_rows) if num_rows else None,
        "no_change_rate": float(no_change_count / num_rows) if num_rows else None,
        "parse_ok_rate": float(parse_ok_count / total_generations),
        "schema_ok_rate": float(schema_ok_count / total_generations),
        "temperature": 0.0,
        "do_sample": False,
        "evaluator": "llm_forecast_counterfactual",
        "attn_implementation": attn_implementation,
    }
    del model
    torch.cuda.empty_cache()
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/aligned/qwen3_4b/dpo_v2")
    parser.add_argument("--input", default="data/processed/counterfactual_tasks_v2.jsonl")
    parser.add_argument("--evaluator", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--debug-allow-heuristic-pass", action="store_true")
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--prompt", default="prompts/forecast_prediction_prompt_qwen3_json.txt")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--extra-site-packages", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--attn-implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager", "none", "default"])
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.80)
    parser.add_argument("--output", default="outputs/metrics/counterfactual_directional_v2.json")
    parser.add_argument("--examples", default="outputs/tables/counterfactual_directional_examples.csv")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    if args.hf_home:
        import os

        os.environ["HF_HOME"] = args.hf_home
    append_extra_site_packages(args.extra_site_packages)

    failures: list[str] = []
    if not Path(args.input).exists():
        failures.append(f"counterfactual input missing: {args.input}")
        tasks = []
    else:
        tasks = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        failures.append(f"checkpoint missing: {args.checkpoint}")
    else:
        required_adapter_files = ["adapter_model.safetensors", "adapter_config.json"]
        if checkpoint_path.is_dir() and not all((checkpoint_path / name).exists() for name in required_adapter_files):
            failures.append(f"checkpoint has no DPO adapter artifact pair: {args.checkpoint}")
    if any(task.get("split") != "test" for task in tasks):
        failures.append("counterfactual evaluation input contains non-test rows")
    missing_expected = sum(1 for task in tasks if not task.get("expected_direction"))
    if missing_expected:
        failures.append(f"counterfactual tasks missing expected_direction: {missing_expected}")
    tasks = tasks[: args.max_tasks] if args.max_tasks else tasks
    if args.evaluator == "heuristic":
        metrics, rows = evaluate_tasks(tasks)
        if not args.debug_allow_heuristic_pass:
            failures.append("heuristic evaluator is debug-only and cannot PASS official Step17")
    elif not failures:
        try:
            metrics, rows = evaluate_tasks_with_llm(tasks, args)
            if metrics.get("schema_ok_rate", 0.0) < args.min_schema_ok_rate:
                failures.append(f"counterfactual schema_ok_rate {metrics.get('schema_ok_rate', 0.0):.4f} < {args.min_schema_ok_rate:.4f}")
        except Exception as exc:
            metrics, rows = {"num_tasks": len(tasks), "evaluator": "llm_forecast_counterfactual"}, []
            failures.append(f"counterfactual LLM evaluation failed: {type(exc).__name__}: {str(exc)[:500]}")
    else:
        metrics, rows = {"num_tasks": len(tasks), "evaluator": args.evaluator}, []
    metrics["checkpoint_used"] = args.checkpoint
    pd.DataFrame(rows[:100]).to_csv(args.examples, index=False)
    write_json(args.output, metrics)
    write_manifest(args.manifest, [args.output, args.examples], STEP)
    status = "PASS" if not failures and tasks else "FAIL"
    if not tasks:
        failures.append("no counterfactual tasks")
    write_status(args.status, STEP, status, [args.input, args.checkpoint], [args.output, args.examples, args.manifest, args.status], metrics, failures, status == "PASS")
    print(json.dumps(metrics, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
