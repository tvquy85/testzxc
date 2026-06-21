from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import artifact_entry, write_json, write_manifest, write_status


DEFAULT_OUT_DIR = "review_samples/dataclean_v4_20062026"
PRODUCER_STEP = "DATACLEAN_V4_REVIEW_SAMPLES"

CANONICAL_STATUS_FILES = [
    "outputs/status/01_FREEZE_CURRENTDATA_V3_FOR_CLEAN_V4_SMALL.status.json",
    "outputs/status/02_AUDIT_CURRENT_V3_FOR_CLEAN_V4_SMALL.status.json",
    "outputs/status/03_ENTITY_ALIAS_MAP_V4_SMALL.status.json",
    "outputs/status/04_ENTITY_EVENT_SCORING_V4_SMALL.status.json",
    "outputs/status/05_ARTICLE_TYPE_AND_NOISE_FILTER_V4_SMALL.status.json",
    "outputs/status/06_DEDUP_NEWS_V4_SMALL.status.json",
    "outputs/status/07_TICKER_DATE_EVIDENCE_PACK_V4_SMALL.status.json",
    "outputs/status/08_RENDER_CONTEXT_EVIDENCE_V4_SMALL.status.json",
    "outputs/status/09_RATIONALE_PROMPT_EVIDENCE_ID_V4_SMALL.status.json",
    "outputs/status/10_STRICT_EVIDENCE_SCHEMA_VALIDATION_V4_SMALL.status.json",
    "outputs/status/11_EVIDENCE_GROUNDING_JUDGE_V4_STAGE0_COMBINED.status.json",
    "outputs/status/12_TRACK_SPLIT_AND_TRAIN_POOL_V4_SMALL.status.json",
    "outputs/status/13_REGENERATE_RATIONALES_V4_STAGE0_SMALL.status.json",
    "outputs/status/13_REGENERATE_RATIONALES_V4_STAGE0_EXTRA_CANDIDATES.status.json",
    "outputs/status/13_COMBINE_RATIONALE_CANDIDATES_V4_STAGE0_SMALL.status.json",
    "outputs/status/14_INDEPENDENT_JUDGE_RERUN_EVIDENCE_V4_STAGE0_COMBINED.status.json",
    "outputs/status/15_FLOW_REWARD_REBUILD_CLEAN_V4_STAGE0_COMBINED.status.json",
    "outputs/status/16_ALIGNMENT_REBUILD_CLEAN_V4_STAGE0_SMALL.status.json",
    "outputs/status/17_BACKTEST_COUNTERFACTUAL_ABLATION_V4.status.json",
    "outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json",
]

METRIC_SNAPSHOT_SOURCES = [
    "outputs/metrics/current_v3_failure_diagnosis_for_clean_v4_small.json",
    "outputs/metrics/ticker_alias_map_v4_small.json",
    "outputs/metrics/current_entity_event_scores_v4_small.json",
    "outputs/metrics/current_article_type_scores_v4_small.json",
    "outputs/metrics/current_deduped_news_v4_small.json",
    "outputs/metrics/ticker_date_evidence_contexts_h1_v4_small.json",
    "outputs/metrics/claim_grounding_evidence_v4_stage0_combined.json",
    "outputs/metrics/independent_inferability_evidence_v4_stage0_combined.json",
    "outputs/metrics/flow_v4_stage0_combined_dataset.json",
    "outputs/metrics/flow_train_v4_stage0_combined.json",
    "outputs/metrics/flow_vs_proxy_clean_v4_stage0_combined.json",
    "outputs/metrics/alignment_dataset_current_clean_v4_stage0_small.json",
    "outputs/metrics/current_clean_v4_test_predictions_full.json",
    "outputs/metrics/backtest_daily_portfolio_current_clean_v4.json",
    "outputs/metrics/counterfactual_directional_current_clean_v4.json",
    "outputs/metrics/ablation_current_clean_v4.json",
    "outputs/metrics/current_clean_v4_step17_metrics.json",
]


def jsonable(value: Any, max_string: int = 1200) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v, max_string) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v, max_string) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v, max_string) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist(), max_string)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return jsonable(value.item(), max_string)
    if value is pd.NA:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, str) and len(value) > max_string:
        return value[:max_string] + f"... [truncated {len(value) - max_string} chars]"
    return value


def read_json_any(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def read_json_or_jsonl_or_text(path: str | Path, max_lines: int = 20) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            if len(rows) >= max_lines:
                break
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                return {"text": text}
        return rows


def read_json_dict(path: str | Path) -> dict[str, Any]:
    data = read_json_any(path)
    return data if isinstance(data, dict) else {}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def sample_rows(df: pd.DataFrame, n: int, group_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    if group_col and group_col in df.columns:
        frames = []
        for _, group in df.sort_values(list(df.columns[: min(3, len(df.columns))])).groupby(group_col, dropna=False):
            frames.append(group.head(max(1, n // max(1, df[group_col].nunique(dropna=False)))))
        sampled = pd.concat(frames, ignore_index=True) if frames else df.head(0)
        if len(sampled) < n:
            sampled = pd.concat([sampled, df.drop(sampled.index, errors="ignore").head(n - len(sampled))])
        return sampled.head(n)
    return df.head(n)


def compact_status_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keep: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            keep[key] = value
        elif isinstance(value, dict) and len(value) <= 10:
            keep[key] = value
        elif isinstance(value, list) and len(value) <= 8:
            keep[key] = value
    return keep


def export_status_summary(out_dir: Path) -> Path:
    rows = []
    for path_text in CANONICAL_STATUS_FILES:
        path = Path(path_text)
        payload = read_json_dict(path)
        rows.append(
            {
                "status_file": path_text,
                "exists": path.exists(),
                "step": payload.get("step"),
                "status": payload.get("status"),
                "next_step_allowed": payload.get("next_step_allowed"),
                "failures": payload.get("failures", []),
                "outputs_created_count": len(payload.get("outputs_created", [])),
                "metrics_compact": compact_status_metrics(payload.get("metrics", {})),
            }
        )
    output = out_dir / "00_status_summary_dataclean_v4_01_18.json"
    write_json(
        output,
        {
            "purpose": "Canonical DataClean V4 status summary for ChatGPT UI review.",
            "master_order": "dataclean_v4_codex_md/00_MASTER_DATACLEAN_V4_ORDER.md",
            "rows": rows,
        },
    )
    return output


def export_source_index(out_dir: Path) -> Path:
    output = out_dir / "01_source_code_index_dataclean_v4.md"
    content = """# Source Code Index For DataClean V4 Review

Use this file as the code-review entrypoint. The sample files in this folder are small artifacts generated by these modules.

| Step | Main code paths | What to review |
|---|---|---|
| 01-02 | `src/repro/freeze_currentdata_baseline.py`, `src/data/audit_current_v3_for_clean_v4.py` | Baseline freeze, failure diagnosis, no overwrite of V3 evidence. |
| 03-07 | `src/data/build_ticker_alias_map_v4.py`, `src/data/evidence_entity_event_score_v4.py`, `src/data/article_type_classifier_v4.py`, `src/data/deduplicate_news_v4.py`, `src/data/build_ticker_date_evidence_pack_v4.py`, `src/data/build_current_train_pool_v4.py` | Entity/event scoring, article type filter, deduplication, evidence-pack construction, train/test separation. |
| 08-10 | `src/llm/render_context_evidence_v4.py`, `src/llm/parse_and_validate_rationale_v4.py`, `src/llm/verify_prompt_v4.py`, `src/llm/verify_rationale_schema_v4.py` | Evidence-id prompt context, strict JSON schema, no repaired/autofixed output. |
| 13 | `src/llm/generate_rationales.py`, `src/llm/combine_rationale_candidates_v4.py`, `prompts/rationale_generation_prompt_evidence_v4.txt` | Qwen3 rationale generation, prompt contract, parse/schema rates, candidate coverage. |
| 11/14 | `src/judges/claim_level_grounding_v4.py`, `src/judges/independent_inferability_judge_v4.py` | Local NLI grounding, cited evidence/signal validation, normal/reversed label-order judge stability. |
| 15 | `src/reward/build_flow_dataset_v4.py`, `src/reward/train_flow_reward_v4.py`, `src/reward/evaluate_flow_vs_proxy_v4.py` | Masked reward targets, flow training smoke, proxy comparison gates. |
| 16 | `src/alignment/build_alignment_current_v4.py` | RWSFT/DPO selection from flow/proxy reward, train-only contract, reward gap. |
| 17 | `src/eval/forecast_prediction.py`, `src/eval/generate_test_predictions_v2.py`, `src/eval/backtest_daily_portfolio_v3.py`, `src/eval/build_counterfactual_clean_v4.py`, `src/eval/run_clean_v4_ablation_suite.py`, `src/eval/clean_v4_step17_gate.py` | Forecast-only prediction, test-only backtest, counterfactual tasks, non-dummy ablations. |
| 18 | `src/repro/currentdata_clean_v4_science_gate.py` | Final pipeline vs claim decision; negative-result-safe gate. |
| Sample export | `src/repro/export_dataclean_v4_samples.py` | Reproducible small sample pack for GitHub/ChatGPT UI review. |
| Tests | `tests/test_rationale_schema_v4.py`, `tests/test_claim_grounding_nli_loader.py`, `tests/test_step15_16_17_smoke_contracts.py` | Contract tests for schema, NLI loader, prediction/backtest/counterfactual gates. |
"""
    output.write_text(content, encoding="utf-8")
    return output


def export_master_order_excerpt(out_dir: Path) -> Path:
    source = Path("dataclean_v4_codex_md/00_MASTER_DATACLEAN_V4_ORDER.md")
    text = source.read_text(encoding="utf-8") if source.exists() else ""
    output = out_dir / "02_master_order_dataclean_v4.md"
    output.write_text(text, encoding="utf-8")
    return output


def export_alias_map_sample(out_dir: Path, n: int) -> Path:
    source = Path("data/quality/ticker_alias_map_v4_small.json")
    data = read_json_dict(source)
    tickers = sorted(data)[:n] if isinstance(data, dict) else []
    output = out_dir / "03_ticker_alias_map_sample.json"
    write_json(output, {"source": str(source), "total_tickers": len(data), "sample": {ticker: data[ticker] for ticker in tickers}})
    return output


def export_dataframe_jsonl(
    out_dir: Path,
    source: str,
    name: str,
    columns: list[str],
    n: int,
    group_col: str | None = None,
) -> Path:
    df = pd.read_parquet(source)
    sampled = sample_rows(df, n=n, group_col=group_col)
    sampled = sampled[existing_columns(sampled, columns)]
    output = out_dir / name
    write_jsonl(output, sampled.to_dict("records"))
    return output


def export_data_cleaning_samples(out_dir: Path, n: int) -> list[Path]:
    common = [
        "sample_id",
        "ticker",
        "timestamp_utc",
        "event_date",
        "split",
        "headline",
        "body",
        "label_5",
        "abnormal_return_h1",
        "article_type",
        "article_type_prelim",
        "target_entity_score",
        "event_specificity_score",
        "text_quality_score",
        "evidence_quality_score",
        "quality_tier",
        "entity_event_keep",
        "entity_event_drop_reason",
        "dedup_keep",
        "cluster_size",
    ]
    return [
        export_dataframe_jsonl(
            out_dir,
            "data/quality/current_entity_event_scores_v4_small.parquet",
            "04_entity_event_scoring_samples.jsonl",
            common,
            n,
            group_col="quality_tier",
        ),
        export_dataframe_jsonl(
            out_dir,
            "data/quality/current_article_type_scores_v4_small.parquet",
            "05_article_type_noise_filter_samples.jsonl",
            common,
            n,
            group_col="article_type",
        ),
        export_dataframe_jsonl(
            out_dir,
            "data/processed/current_deduped_news_v4_small.parquet",
            "06_dedup_news_samples.jsonl",
            common + ["normalized_text_hash", "near_dup_cluster_id", "is_cluster_representative"],
            n,
            group_col="quality_tier",
        ),
    ]


def export_evidence_context_samples(out_dir: Path, n: int) -> list[Path]:
    columns = [
        "sample_id",
        "ticker",
        "event_date",
        "split",
        "horizon",
        "target_label_5",
        "target_direction",
        "num_company_event_evidence",
        "num_context_only_evidence",
        "mean_evidence_quality_score",
        "has_company_event_news",
        "no_news_context_flag",
        "evidence_pack_json",
        "technical_event_tokens_json",
        "clean_context_text",
    ]
    return [
        export_dataframe_jsonl(
            out_dir,
            "data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet",
            "07_evidence_pack_context_samples.jsonl",
            columns,
            n,
            group_col="split",
        ),
        export_dataframe_jsonl(
            out_dir,
            "data/processed/current_clean_train_pool_v4_small.parquet",
            "12_track_train_pool_samples.jsonl",
            columns + ["track", "split_track"],
            n,
            group_col="track",
        ),
        export_dataframe_jsonl(
            out_dir,
            "data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet",
            "08_rendered_context_text_samples.jsonl",
            [
                "sample_id",
                "ticker",
                "event_date",
                "split",
                "num_company_event_evidence",
                "num_context_only_evidence",
                "has_company_event_news",
                "no_news_context_flag",
                "technical_event_tokens_json",
                "clean_context_text",
            ],
            n,
            group_col="split",
        ),
    ]


def export_prompt_and_schema_samples(out_dir: Path) -> list[Path]:
    outputs = []
    prompt_source = Path("prompts/rationale_generation_prompt_evidence_v4.txt")
    prompt_text = prompt_source.read_text(encoding="utf-8") if prompt_source.exists() else ""
    output = out_dir / "09_prompt_template_evidence_v4.md"
    output.write_text(
        "# Prompt Template: Evidence V4\n\n"
        f"Source: `{prompt_source}`\n\n"
        "```text\n"
        f"{prompt_text}"
        "\n```\n",
        encoding="utf-8",
    )
    outputs.append(output)

    for source, name in [
        ("outputs/data_samples/sample_raw_rationale.jsonl", "10_sample_raw_rationale.jsonl"),
        ("outputs/data_samples/sample_parsed_rationale.json", "10_sample_parsed_rationale.json"),
        ("outputs/data_samples/sample_technical_tokens.json", "10_sample_technical_tokens.json"),
    ]:
        src = Path(source)
        dst = out_dir / name
        if src.exists():
            if src.suffix == ".jsonl":
                rows = []
                with src.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
                write_jsonl(dst, rows[:5])
            else:
                write_json(dst, read_json_or_jsonl_or_text(src))
            outputs.append(dst)
    return outputs


def export_rationale_samples(out_dir: Path, n: int) -> Path:
    columns = [
        "sample_id",
        "split",
        "ticker",
        "event_timestamp",
        "candidate_id",
        "model_name",
        "model_path",
        "generator_model",
        "prompt_hash",
        "run_id",
        "stage",
        "context_meta_json",
        "prompt",
        "raw_output",
        "parse_ok",
        "schema_ok",
        "parse_errors",
        "parsed_json",
    ]
    return export_dataframe_jsonl(
        out_dir,
        "data/rationales/parsed/current_clean_train_qwen3_4b_v4_stage0_combined.parquet",
        "13_rationale_generation_samples.jsonl",
        columns,
        n,
        group_col="candidate_id",
    )


def export_grounding_samples(out_dir: Path, n: int) -> Path:
    columns = [
        "sample_id",
        "candidate_id",
        "split",
        "track",
        "status",
        "total_claims",
        "supported_claims",
        "unsupported_claims",
        "unverified_claims",
        "contradiction_claims",
        "news_grounding_score",
        "technical_grounding_score",
        "missing_evidence_id_count",
        "unknown_evidence_id_count",
        "missing_signal_id_count",
        "unknown_signal_id_count",
        "unsupported_news_claim_count",
        "news_claim_count",
        "technical_claim_count",
        "claim_details_json",
    ]
    return export_dataframe_jsonl(
        out_dir,
        "data/judges/claim_grounding_evidence_v4_stage0_combined.parquet",
        "11_claim_grounding_nli_samples.jsonl",
        columns,
        n,
        group_col="status",
    )


def export_independent_judge_samples(out_dir: Path, n: int) -> Path:
    columns = [
        "sample_id",
        "candidate_id",
        "split",
        "track",
        "target_label_5",
        "judge_model",
        "judge_parse_ok",
        "judge_schema_ok",
        "argmax_consistency",
        "l1_probability_delta",
        "true_label_probability",
        "true_label_probability_debiased",
        "p_strong_down_debiased",
        "p_mild_down_debiased",
        "p_neutral_debiased",
        "p_mild_up_debiased",
        "p_strong_up_debiased",
        "raw_judge_outputs_json",
    ]
    return export_dataframe_jsonl(
        out_dir,
        "data/judges/independent_inferability_evidence_v4_stage0_combined.parquet",
        "14_independent_judge_debias_samples.jsonl",
        columns,
        n,
        group_col="track",
    )


def export_flow_samples(out_dir: Path, n: int) -> Path:
    source = Path("data/reward/flow_v4_stage0_combined_dataset.pt")
    data = torch.load(source, map_location="cpu", weights_only=False)
    target_names = list(data.get("target_names", []))
    targets = np.asarray(data.get("target", []), dtype=float)
    masks = np.asarray(data.get("mask", []), dtype=float)
    sample_ids = list(data.get("sample_id", []))
    candidate_ids = list(data.get("candidate_id", []))
    splits = list(data.get("split", []))
    tracks = list(data.get("track", []))
    rows = []
    for idx in range(min(n, len(sample_ids))):
        rows.append(
            {
                "sample_id": sample_ids[idx],
                "candidate_id": candidate_ids[idx] if idx < len(candidate_ids) else None,
                "split": splits[idx] if idx < len(splits) else None,
                "track": tracks[idx] if idx < len(tracks) else None,
                "target": {name: float(targets[idx, j]) for j, name in enumerate(target_names)},
                "mask": {name: float(masks[idx, j]) for j, name in enumerate(target_names)},
            }
        )
    output = out_dir / "15_flow_reward_dataset_samples.json"
    write_json(
        output,
        {
            "source": str(source),
            "rows_total": len(sample_ids),
            "target_names": target_names,
            "metadata": data.get("metadata", {}),
            "samples": rows,
        },
    )
    return output


def read_jsonl_head(path: str | Path, n: int) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def export_alignment_samples(out_dir: Path, n: int) -> list[Path]:
    outputs = []
    for source, name in [
        ("data/alignment/rwsft_current_clean_v4_stage0_small.jsonl", "16_rwsft_alignment_samples.jsonl"),
        ("data/alignment/dpo_current_clean_v4_stage0_small.jsonl", "16_dpo_alignment_pair_samples.jsonl"),
    ]:
        output = out_dir / name
        write_jsonl(output, read_jsonl_head(source, n))
        outputs.append(output)
    scored = export_dataframe_jsonl(
        out_dir,
        "data/alignment/scored_current_clean_v4_stage0_small.parquet",
        "16_scored_alignment_candidates_samples.jsonl",
        [
            "sample_id",
            "split",
            "ticker",
            "candidate_id",
            "track",
            "parse_ok",
            "schema_ok",
            "grounding_status",
            "true_label_probability_debiased",
            "news_grounding_score",
            "technical_grounding_score",
            "proxy_reward",
            "flow_reward_raw",
            "flow_reward_normalized",
            "final_reward",
            "reward_source",
            "parsed_json",
        ],
        n,
        group_col="track",
    )
    outputs.append(scored)
    return outputs


def export_prediction_backtest_samples(out_dir: Path, per_action: int, backtest_days: int) -> list[Path]:
    outputs = []
    pred = pd.read_parquet("outputs/predictions/current_clean_v4_test_predictions.parquet")
    rows = []
    for _, group in pred.sort_values(["event_date", "ticker", "sample_id"]).groupby("action", dropna=False):
        rows.extend(group.head(per_action).to_dict("records"))
    output = out_dir / "17_prediction_forecast_samples.jsonl"
    write_jsonl(output, rows)
    outputs.append(output)

    daily = pd.read_csv("outputs/tables/backtest_daily_returns_current_clean_v4.csv").head(backtest_days)
    output = out_dir / "17_backtest_daily_returns_sample.csv"
    write_csv(output, daily)
    outputs.append(output)
    return outputs


def export_counterfactual_samples(out_dir: Path, per_type: int) -> list[Path]:
    outputs = []
    tasks = pd.read_parquet("data/counterfactual/current_clean_v4_cf_tasks.parquet")
    rows = []
    for _, group in tasks.sort_values(["counterfactual_type", "sample_id"]).groupby("counterfactual_type", dropna=False):
        rows.extend(group.head(per_type).to_dict("records"))
    output = out_dir / "17_counterfactual_task_samples.jsonl"
    write_jsonl(output, rows)
    outputs.append(output)

    fail_examples = read_json_any("outputs/data_samples/counterfactual_fail_examples_current_clean_v4.json")
    output = out_dir / "17_counterfactual_fail_examples.json"
    if isinstance(fail_examples, list):
        fail_examples = fail_examples[: per_type * 4]
    write_json(
        output,
        {
            "source": "outputs/data_samples/counterfactual_fail_examples_current_clean_v4.json",
            "examples": fail_examples,
        },
    )
    outputs.append(output)
    return outputs


def export_tables_and_gate(out_dir: Path) -> list[Path]:
    outputs = []
    for source, name in [
        ("outputs/tables/ablation_current_clean_v4.csv", "17_ablation_current_clean_v4.csv"),
        ("outputs/tables/flow_vs_proxy_clean_v4_stage0_combined.csv", "15_flow_vs_proxy_clean_v4_stage0_combined.csv"),
    ]:
        src = Path(source)
        if src.exists():
            df = pd.read_csv(src)
            output = out_dir / name
            write_csv(output, df)
            outputs.append(output)

    output = out_dir / "18_science_gate_report.json"
    write_json(output, read_json_any("outputs/repro/currentdata_clean_v4_science_gate_report.json"))
    outputs.append(output)
    return outputs


def export_metrics_snapshot(out_dir: Path) -> Path:
    output = out_dir / "metrics_snapshot_dataclean_v4.json"
    write_json(output, {path: read_json_any(path) for path in METRIC_SNAPSHOT_SOURCES if Path(path).exists()})
    return output


def export_readme(out_dir: Path, files: list[Path]) -> Path:
    listed = sorted({path.name for path in files} | {"README.md", "sample_manifest.json"})
    rows = "\n".join(f"- `{name}`" for name in listed)
    content = f"""# DataClean V4 Review Samples

Thư mục này chứa các file sample nhỏ sinh ra từ flow `dataclean_v4_codex_md`.
Mục tiêu là đưa lên GitHub để ChatGPT UI review chất lượng code, prompt, dữ liệu mẫu, rationale, judge, reward, alignment và evaluation mà không cần upload raw dataset hoặc checkpoint lớn.

Recommended review order:

1. `01_source_code_index_dataclean_v4.md`
2. `02_master_order_dataclean_v4.md`
3. `04_entity_event_scoring_samples.jsonl`, `05_article_type_noise_filter_samples.jsonl`, `06_dedup_news_samples.jsonl`
4. `07_evidence_pack_context_samples.jsonl`, `08_rendered_context_text_samples.jsonl`, `09_prompt_template_evidence_v4.md`
5. `13_rationale_generation_samples.jsonl`
6. `11_claim_grounding_nli_samples.jsonl`, `14_independent_judge_debias_samples.jsonl`
7. `15_flow_reward_dataset_samples.json`, `15_flow_vs_proxy_clean_v4_stage0_combined.csv`
8. `16_rwsft_alignment_samples.jsonl`, `16_dpo_alignment_pair_samples.jsonl`
9. `17_prediction_forecast_samples.jsonl`, `17_backtest_daily_returns_sample.csv`, `17_counterfactual_task_samples.jsonl`, `17_ablation_current_clean_v4.csv`
10. `18_science_gate_report.json`, `00_status_summary_dataclean_v4_01_18.json`

Files:

{rows}
"""
    output = out_dir / "README.md"
    output.write_text(content, encoding="utf-8")
    return output


def export_sample_manifest(out_dir: Path, files: list[Path], run_id: str) -> Path:
    source_map = {
        "00_status_summary_dataclean_v4_01_18.json": "outputs/status/*DATACLEAN_V4*.status.json and canonical step statuses 01-18",
        "03_ticker_alias_map_sample.json": "data/quality/ticker_alias_map_v4_small.json",
        "04_entity_event_scoring_samples.jsonl": "data/quality/current_entity_event_scores_v4_small.parquet",
        "05_article_type_noise_filter_samples.jsonl": "data/quality/current_article_type_scores_v4_small.parquet",
        "06_dedup_news_samples.jsonl": "data/processed/current_deduped_news_v4_small.parquet",
        "07_evidence_pack_context_samples.jsonl": "data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet",
        "08_rendered_context_text_samples.jsonl": "data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet",
        "12_track_train_pool_samples.jsonl": "data/processed/current_clean_train_pool_v4_small.parquet",
        "13_rationale_generation_samples.jsonl": "data/rationales/parsed/current_clean_train_qwen3_4b_v4_stage0_combined.parquet",
        "11_claim_grounding_nli_samples.jsonl": "data/judges/claim_grounding_evidence_v4_stage0_combined.parquet",
        "14_independent_judge_debias_samples.jsonl": "data/judges/independent_inferability_evidence_v4_stage0_combined.parquet",
        "15_flow_reward_dataset_samples.json": "data/reward/flow_v4_stage0_combined_dataset.pt",
        "16_rwsft_alignment_samples.jsonl": "data/alignment/rwsft_current_clean_v4_stage0_small.jsonl",
        "16_dpo_alignment_pair_samples.jsonl": "data/alignment/dpo_current_clean_v4_stage0_small.jsonl",
        "17_prediction_forecast_samples.jsonl": "outputs/predictions/current_clean_v4_test_predictions.parquet",
        "17_backtest_daily_returns_sample.csv": "outputs/tables/backtest_daily_returns_current_clean_v4.csv",
        "17_counterfactual_task_samples.jsonl": "data/counterfactual/current_clean_v4_cf_tasks.parquet",
        "17_ablation_current_clean_v4.csv": "outputs/tables/ablation_current_clean_v4.csv",
        "18_science_gate_report.json": "outputs/repro/currentdata_clean_v4_science_gate_report.json",
    }
    output = out_dir / "sample_manifest.json"
    payload = {
        "producer": "src/repro/export_dataclean_v4_samples.py",
        "producer_step": PRODUCER_STEP,
        "run_id": run_id,
        "purpose": "Small GitHub-friendly samples for ChatGPT UI review of DataClean V4 flow.",
        "master_order": "dataclean_v4_codex_md/00_MASTER_DATACLEAN_V4_ORDER.md",
        "sample_files": [artifact_entry(path, PRODUCER_STEP, run_id) for path in sorted(files)],
        "source_map": source_map,
    }
    write_json(output, payload)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--prediction-per-action", type=int, default=3)
    parser.add_argument("--counterfactual-per-type", type=int, default=2)
    parser.add_argument("--backtest-days", type=int, default=20)
    parser.add_argument("--run-id", default="dataclean_v4_review_samples_20062026")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    files.append(export_status_summary(out_dir))
    files.append(export_source_index(out_dir))
    files.append(export_master_order_excerpt(out_dir))
    files.append(export_alias_map_sample(out_dir, args.sample_size))
    files.extend(export_data_cleaning_samples(out_dir, args.sample_size))
    files.extend(export_evidence_context_samples(out_dir, args.sample_size))
    files.extend(export_prompt_and_schema_samples(out_dir))
    files.append(export_grounding_samples(out_dir, args.sample_size))
    files.append(export_rationale_samples(out_dir, args.sample_size))
    files.append(export_independent_judge_samples(out_dir, args.sample_size))
    files.append(export_flow_samples(out_dir, args.sample_size))
    files.extend(export_alignment_samples(out_dir, args.sample_size))
    files.extend(export_prediction_backtest_samples(out_dir, args.prediction_per_action, args.backtest_days))
    files.extend(export_counterfactual_samples(out_dir, args.counterfactual_per_type))
    files.extend(export_tables_and_gate(out_dir))
    files.append(export_metrics_snapshot(out_dir))
    files.append(export_readme(out_dir, files))
    files.append(export_sample_manifest(out_dir, files, args.run_id))

    manifest_path = Path("outputs/manifests/DATACLEAN_V4_REVIEW_SAMPLES.manifest.json")
    write_manifest(
        manifest_path,
        [str(path) for path in files],
        PRODUCER_STEP,
        args.run_id,
        extra={"sample_output_dir": str(out_dir).replace("\\", "/")},
    )
    status_path = Path("outputs/status/DATACLEAN_V4_REVIEW_SAMPLES.status.json")
    write_status(
        status_path,
        step=PRODUCER_STEP,
        status="PASS",
        inputs_checked=[
            "dataclean_v4_codex_md/00_MASTER_DATACLEAN_V4_ORDER.md",
            "outputs/status/18_SCIENCE_GATE_AND_RUNBOOK_V4.status.json",
            "outputs/repro/currentdata_clean_v4_science_gate_report.json",
        ],
        outputs_created=[str(path) for path in files] + [str(manifest_path), str(status_path)],
        metrics={
            "sample_output_dir": str(out_dir).replace("\\", "/"),
            "sample_file_count": len(files),
            "run_id": args.run_id,
        },
        failures=[],
        next_step_allowed=True,
    )

    print(
        json.dumps(
            {
                "output_dir": str(out_dir).replace("\\", "/"),
                "files": [str(path).replace("\\", "/") for path in files],
                "manifest": str(manifest_path).replace("\\", "/"),
                "status": str(status_path).replace("\\", "/"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
