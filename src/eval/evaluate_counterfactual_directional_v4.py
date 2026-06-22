from __future__ import annotations

import argparse
import json
import os
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

STEP = "16_COUNTERFACTUAL_NEWS_EVIDENCE_V6"


def pass_expected_direction(expected: str, original_score: float, counterfactual_score: float, min_delta: float) -> tuple[bool, bool, float]:
    from src.eval.counterfactual_direction_rules_v6 import normalized_expected_direction

    expected = normalized_expected_direction(expected)
    delta = counterfactual_score - original_score
    no_change = abs(delta) < min_delta
    if expected == "down_decrease":
        return delta >= min_delta, no_change, delta
    if expected == "up_decrease":
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


def read_tasks(path: str) -> list[dict[str, Any]]:
    import pandas as pd

    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    return pd.read_parquet(p).to_dict(orient="records")


def write_breakdown(path: str, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    table = pd.DataFrame(rows)
    if table.empty:
        out = pd.DataFrame(columns=["counterfactual_type", "tasks", "pass_rate", "wrong_direction_rate", "no_change_rate"])
    else:
        grouped = table.groupby("counterfactual_type", dropna=False)
        out = grouped.agg(
            tasks=("sample_id", "count"),
            pass_rate=("directional_pass", "mean"),
            no_change_rate=("no_change", "mean"),
            schema_ok_rate=("both_schema_ok", "mean"),
        ).reset_index()
        out["wrong_direction_rate"] = 1.0 - out["pass_rate"] - out["no_change_rate"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def evaluate_tasks_with_llm(tasks: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    prompt_template = Path(args.prompt).read_text(encoding="utf-8")
    model_path = resolve_model_path(args.model_key, args.model_path, args.config)
    tokenizer, model = load_model(args, model_path, args.adapter)
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
        dist, action, pred_label, parse_ok, schema_ok, parse_errors = parse_forecast_prediction(
            raw_text, action_policy=args.action_policy
        )
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
        both_schema_ok = bool(original["schema_ok"] and counterfactual["schema_ok"])
        both_parse_ok = bool(original["parse_ok"] and counterfactual["parse_ok"])
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
                "no_change": bool(no_change and both_schema_ok),
                "both_parse_ok": both_parse_ok,
                "both_schema_ok": both_schema_ok,
                "original_parse_ok": original["parse_ok"],
                "counterfactual_parse_ok": counterfactual["parse_ok"],
                "original_schema_ok": original["schema_ok"],
                "counterfactual_schema_ok": counterfactual["schema_ok"],
                "original_action": original["action"],
                "counterfactual_action": counterfactual["action"],
                "original_raw_output": original["raw_output"],
                "counterfactual_raw_output": counterfactual["raw_output"],
            }
        )

    num_rows = len(rows)
    pass_count = sum(1 for row in rows if row["directional_pass"])
    no_change_count = sum(1 for row in rows if row["no_change"])
    wrong_count = max(0, num_rows - pass_count - no_change_count)
    individual_parse_count = sum(row["original_parse_ok"] for row in rows) + sum(row["counterfactual_parse_ok"] for row in rows)
    individual_schema_count = sum(row["original_schema_ok"] for row in rows) + sum(row["counterfactual_schema_ok"] for row in rows)
    metrics = {
        "num_tasks": num_rows,
        "mean_delta": float(sum(row["delta"] for row in rows) / num_rows) if num_rows else None,
        "pass_rate": float(pass_count / num_rows) if num_rows else 0.0,
        "wrong_direction_rate": float(wrong_count / num_rows) if num_rows else 0.0,
        "no_change_rate": float(no_change_count / num_rows) if num_rows else 0.0,
        "parse_ok_rate": float(individual_parse_count / (2 * num_rows)) if num_rows else 0.0,
        "schema_ok_rate": float(individual_schema_count / (2 * num_rows)) if num_rows else 0.0,
        "pair_parse_ok_rate": float(sum(row["both_parse_ok"] for row in rows) / num_rows) if num_rows else 0.0,
        "pair_schema_ok_rate": float(sum(row["both_schema_ok"] for row in rows) / num_rows) if num_rows else 0.0,
        "temperature": 0.0,
        "attn_implementation": attn_implementation,
        "checkpoint_used": args.adapter,
        "min_delta": args.min_delta,
        "action_policy": args.action_policy,
    }
    for cf_type in sorted({str(row["counterfactual_type"]) for row in rows}):
        type_rows = [row for row in rows if str(row["counterfactual_type"]) == cf_type]
        metrics[f"pass_rate_{cf_type}"] = float(sum(row["directional_pass"] for row in type_rows) / max(1, len(type_rows)))

    del model
    torch.cuda.empty_cache()
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--adapter", "--checkpoint", dest="adapter", default="checkpoints/aligned/qwen3_4b/current_v3_dpo")
    parser.add_argument("--model-key", "--model", dest="model_key", default="qwen3_4b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--prompt", default="prompts/forecast_prediction_prompt_qwen3_json.txt")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--extra-site-packages", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--action-policy", default="derive", choices=["strict", "derive"])
    parser.add_argument("--min-delta", type=float, default=0.03)
    parser.add_argument("--max-tasks", type=int, default=500)
    parser.add_argument("--min-schema-ok-rate", type=float, default=0.80)
    parser.add_argument("--min-pass-rate", type=float, default=0.50)
    parser.add_argument("--max-no-change-rate", type=float, default=0.35)
    parser.add_argument("--output", "--metrics", dest="output", default="outputs/metrics/counterfactual_directional_medium_v4.json")
    parser.add_argument("--breakdown-output", "--breakdown", dest="breakdown_output", default="outputs/tables/counterfactual_directional_medium_breakdown.csv")
    parser.add_argument("--failures-output", "--fail-examples", "--failures", dest="failures_output", default="outputs/data_samples/counterfactual_fail_examples_medium_v4.json")
    parser.add_argument("--rows-output", default=None)
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--step-name", default=STEP)
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    append_extra_site_packages(args.extra_site_packages)

    failures: list[str] = []
    if args.temperature != 0.0:
        failures.append("counterfactual official eval must use deterministic temperature 0")
    if not (Path(args.adapter) / "adapter_config.json").exists():
        failures.append(f"adapter_config.json missing: {args.adapter}")
    if not (Path(args.adapter) / "adapter_model.safetensors").exists():
        failures.append(f"adapter_model.safetensors missing: {args.adapter}")

    tasks = read_tasks(args.tasks)
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        failures.append("counterfactual tasks input is empty")
    if tasks and {str(task.get("split")) for task in tasks} != {"test"}:
        failures.append("counterfactual tasks contain non-test split")

    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"num_tasks": len(tasks)}
    if not failures:
        try:
            metrics, rows = evaluate_tasks_with_llm(tasks, args)
        except Exception as exc:
            failures.append(f"LLM evaluation failed: {type(exc).__name__}: {str(exc)[:500]}")
            import traceback

            traceback.print_exc()

    Path(args.failures_output).parent.mkdir(parents=True, exist_ok=True)
    fail_examples = [row for row in rows if not row.get("directional_pass")]
    with open(args.failures_output, "w", encoding="utf-8") as f:
        json.dump(fail_examples[:100], f, indent=2, ensure_ascii=False)
    write_breakdown(args.breakdown_output, rows)
    if args.rows_output:
        import pandas as pd

        Path(args.rows_output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.rows_output, index=False)

    schema_ok_rate = float(metrics.get("schema_ok_rate") or 0.0)
    pass_rate = float(metrics.get("pass_rate") or 0.0)
    no_change_rate = float(metrics.get("no_change_rate") or 1.0)
    metrics.update(
        {
            "pipeline_pass": not failures,
            "claim_allowed": bool(pass_rate >= args.min_pass_rate and no_change_rate <= args.max_no_change_rate),
            "general_counterfactual_claim_allowed": bool(pass_rate >= args.min_pass_rate and no_change_rate <= args.max_no_change_rate),
            "news_faithfulness_claim_allowed": bool(
                metrics.get("pass_rate_remove_positive_evidence", 0.0) >= 0.35
                and metrics.get("pass_rate_remove_negative_evidence", 0.0) >= 0.35
            ),
            "min_schema_ok_rate_required": args.min_schema_ok_rate,
            "min_pass_rate": args.min_pass_rate,
            "max_no_change_rate": args.max_no_change_rate,
        }
    )
    if rows and schema_ok_rate < args.min_schema_ok_rate:
        failures.append(f"schema_ok_rate {schema_ok_rate:.3f} < {args.min_schema_ok_rate:.3f}")
    metrics["pipeline_pass"] = not failures and bool(tasks)

    write_json(args.output, metrics)
    manifest_paths = [args.tasks, args.output, args.breakdown_output, args.failures_output]
    if args.rows_output:
        manifest_paths.append(args.rows_output)
    write_manifest(args.manifest, manifest_paths, args.step_name)
    status = "PASS" if metrics["pipeline_pass"] else "FAIL"
    write_status(
        args.status,
        args.step_name,
        status,
        [args.tasks, args.adapter, args.prompt],
        [args.output, args.breakdown_output, args.failures_output, *([args.rows_output] if args.rows_output else []), args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
