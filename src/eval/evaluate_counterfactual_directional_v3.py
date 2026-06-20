from __future__ import annotations

import argparse
import json
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

STEP = "15_COUNTERFACTUAL_DIRECTIONAL_FIX_CURRENT_DATA"

def pass_expected_direction(expected: str, original_score: float, counterfactual_score: float, min_delta: float) -> tuple[bool, bool, float]:
    delta = counterfactual_score - original_score
    no_change = abs(delta) < min_delta
    if expected == "down_decrease":
        # If P(down) decreases, the score (E[Return]) should INCREASE.
        return delta >= min_delta, no_change, delta
    if expected == "up_decrease":
        # If P(up) decreases, the score (E[Return]) should DECREASE.
        return delta <= -min_delta, no_change, delta
    return False, no_change, delta

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
    
    for meta, raw_text in zip(prompt_meta, raw_outputs):
        dist, action, pred_label, parse_ok, schema_ok, parse_errors = parse_forecast_prediction(raw_text)
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
        
        passed, no_change, delta = pass_expected_direction(expected, float(original["score"]), float(counterfactual["score"]), args.min_delta)
        
        rows.append({
            "sample_id": task.get("sample_id"),
            "split": task.get("split"),
            "counterfactual_type": task.get("counterfactual_type"),
            "expected_direction": expected,
            "original_score": original["score"],
            "counterfactual_score": counterfactual["score"],
            "delta": delta,
            "directional_pass": passed,
            "no_change": no_change,
            "original_parse_ok": original["parse_ok"],
            "counterfactual_parse_ok": counterfactual["parse_ok"],
            "original_schema_ok": original["schema_ok"],
            "counterfactual_schema_ok": counterfactual["schema_ok"],
            "original_raw_output": original["raw_output"],
            "counterfactual_raw_output": counterfactual["raw_output"],
        })
        
    num_rows = len(rows)
    pass_count = sum(1 for row in rows if row["directional_pass"])
    no_change_count = sum(1 for row in rows if row["no_change"])
    wrong_count = num_rows - pass_count - no_change_count
    
    metrics = {
        "num_tasks": num_rows,
        "mean_delta": float(sum(row["delta"] for row in rows) / num_rows) if num_rows else None,
        "pass_rate": float(pass_count / num_rows) if num_rows else None,
        "wrong_direction_rate": float(wrong_count / num_rows) if num_rows else None,
        "no_change_rate": float(no_change_count / num_rows) if num_rows else None,
        "parse_ok_rate": float(
            sum(row["original_parse_ok"] and row["counterfactual_parse_ok"] for row in rows) / num_rows
        ) if num_rows else None,
        "schema_ok_rate": float(
            sum(row["original_schema_ok"] and row["counterfactual_schema_ok"] for row in rows) / num_rows
        ) if num_rows else None,
        "temperature": args.temperature,
        "attn_implementation": attn_implementation,
        "checkpoint_used": args.checkpoint,
    }
    
    # Calculate pass_rate by task type
    for cf_type in set(row["counterfactual_type"] for row in rows):
        type_rows = [r for r in rows if r["counterfactual_type"] == cf_type]
        metrics[f"pass_rate_{cf_type}"] = float(sum(1 for r in type_rows if r["directional_pass"]) / max(1, len(type_rows)))
        
    del model
    torch.cuda.empty_cache()
    return metrics, rows

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/counterfactual/current_cf_tasks_v3.parquet")
    parser.add_argument("--checkpoint", default="checkpoints/aligned/qwen3_4b/current_v3_dpo")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-delta", type=float, default=0.03)
    parser.add_argument("--max-tasks", type=int, default=100)
    
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--prompt", default="prompts/forecast_prediction_prompt_qwen3_json.txt")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--extra-site-packages", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.80)
    
    parser.add_argument("--metrics", default="outputs/metrics/counterfactual_directional_current_v3.json")
    parser.add_argument("--fail-examples", default="outputs/data_samples/counterfactual_fail_examples_current_v3.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd
    
    if args.hf_home:
        import os
        os.environ["HF_HOME"] = args.hf_home
    append_extra_site_packages(args.extra_site_packages)

    failures: list[str] = []
    if not (Path(args.checkpoint) / "adapter_config.json").exists():
        failures.append(f"checkpoint adapter_config.json missing: {args.checkpoint}")
    if not (Path(args.checkpoint) / "adapter_model.safetensors").exists():
        failures.append(f"checkpoint adapter_model.safetensors missing: {args.checkpoint}")
    
    if not Path(args.tasks).exists():
        failures.append(f"tasks input missing: {args.tasks}")
        tasks = []
    else:
        df = pd.read_parquet(args.tasks)
        if len(df) and set(df["split"].dropna()) != {"test"}:
            failures.append("counterfactual tasks contain non-test split")
        if args.max_tasks:
            df = df.head(args.max_tasks)
        tasks = df.to_dict(orient="records")
        
    if not failures:
        try:
            metrics, rows = evaluate_tasks_with_llm(tasks, args)
        except Exception as exc:
            metrics, rows = {"num_tasks": len(tasks)}, []
            failures.append(f"LLM evaluation failed: {type(exc).__name__}: {str(exc)[:500]}")
            import traceback
            traceback.print_exc()
    else:
        metrics, rows = {"num_tasks": len(tasks)}, []

    fail_examples = [row for row in rows if not row["directional_pass"]]
    Path(args.fail_examples).parent.mkdir(parents=True, exist_ok=True)
    with open(args.fail_examples, "w", encoding="utf-8") as f:
        json.dump(fail_examples[:50], f, indent=2, ensure_ascii=False)
        
    write_json(args.metrics, metrics)
    
    pass_rate = metrics.get("pass_rate", 0.0)
    no_change_rate = metrics.get("no_change_rate", 1.0)
    schema_ok_rate = metrics.get("schema_ok_rate", 0.0) or 0.0
    if schema_ok_rate < args.min_schema_ok_rate and rows:
        failures.append(f"schema_ok_rate {schema_ok_rate:.3f} < {args.min_schema_ok_rate:.3f}")
    
    if pass_rate <= 0.16 and no_change_rate >= 0.696:
        failures.append(f"Gate failed: pass_rate {pass_rate:.3f} <= 0.16 AND no_change_rate {no_change_rate:.3f} >= 0.696")
        
    status = "PASS" if not failures and tasks else "FAIL"
    write_manifest(args.manifest, [args.tasks, args.metrics, args.fail_examples], STEP)
    write_status(
        args.status,
        STEP,
        status,
        [args.tasks, args.checkpoint, args.prompt],
        [args.metrics, args.fail_examples, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    
    print(json.dumps(metrics, indent=2))
    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
