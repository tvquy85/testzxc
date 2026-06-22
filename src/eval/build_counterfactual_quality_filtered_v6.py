from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.build_counterfactual_evidence_v6 import (
    NEGATIVE_TERMS,
    POSITIVE_TERMS,
    TASK_TYPES,
    build_task,
)
from src.eval.counterfactual_direction_rules_v6 import normalized_expected_direction
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "16_6_COUNTERFACTUAL_QUALITY_FILTERED_TASKS_V6"


def term_count(text: Any, terms: list[str]) -> int:
    low = str(text or "").lower()
    return int(sum(len(re.findall(rf"\b{re.escape(term)}\b", low)) for term in terms))


def task_text(task: dict[str, Any], prefix: str) -> str:
    return f"{task.get(f'{prefix}_headline', '')} {task.get(f'{prefix}_body', '')}"


def expected_side(task: dict[str, Any]) -> str:
    expected = normalized_expected_direction(str(task.get("expected_direction") or ""))
    if expected == "up_decrease":
        return "up"
    if expected == "down_decrease":
        return "down"
    return "unknown"


def placeholder_counterfactual(task: dict[str, Any]) -> bool:
    cf_type = str(task.get("counterfactual_type") or "")
    return (
        str(task.get("counterfactual_headline", "")).strip() == f"{cf_type} applied"
        or str(task.get("counterfactual_body", "")).strip() == f"{cf_type} applied"
        or str(task.get("counterfactual_body", "")).strip()
        == "company-specific evidence removed; only non-company context remains"
    )


def repair_placeholder(task: dict[str, Any]) -> dict[str, Any]:
    out = dict(task)
    cf_type = str(out.get("counterfactual_type") or "")
    if cf_type == "remove_positive_evidence":
        out["counterfactual_headline"] = "Counterfactual: favorable company evidence removed"
        out["counterfactual_body"] = (
            "The selected favorable company-specific evidence has been removed. "
            "No favorable company event evidence remains in this counterfactual context."
        )
    elif cf_type == "remove_negative_evidence":
        out["counterfactual_headline"] = "Counterfactual: unfavorable company evidence removed"
        out["counterfactual_body"] = (
            "The selected unfavorable company-specific evidence has been removed. "
            "No unfavorable company event evidence remains in this counterfactual context."
        )
    elif cf_type == "remove_all_company_evidence":
        out["counterfactual_headline"] = "Counterfactual: all company-specific evidence removed"
        out["counterfactual_body"] = (
            "All company-specific evidence has been removed. "
            "Only non-company or neutral market context remains for this counterfactual."
        )
    return out


def repair_semantic_neutralization(task: dict[str, Any]) -> dict[str, Any]:
    out = dict(task)
    cf_type = str(out.get("counterfactual_type") or "")
    if cf_type == "neutralize_positive_evidence":
        out["counterfactual_headline"] = "Counterfactual: favorable language rewritten as neutral company update"
        out["counterfactual_body"] = (
            "The favorable company-specific language has been rewritten as a neutral operational update. "
            "No favorable company event signal is stated in this counterfactual context."
        )
    elif cf_type == "neutralize_negative_evidence":
        out["counterfactual_headline"] = "Counterfactual: unfavorable language rewritten as neutral company update"
        out["counterfactual_body"] = (
            "The unfavorable company-specific language has been rewritten as a neutral operational update. "
            "No unfavorable company event signal is stated in this counterfactual context."
        )
    return out


def token_changed(task: dict[str, Any]) -> bool:
    return str(task.get("original_technical_event_tokens_json") or "") != str(
        task.get("counterfactual_technical_event_tokens_json") or ""
    )


def quality_features(task: dict[str, Any]) -> dict[str, Any]:
    original = task_text(task, "original")
    counterfactual = task_text(task, "counterfactual")
    original_pos = term_count(original, POSITIVE_TERMS)
    original_neg = term_count(original, NEGATIVE_TERMS)
    cf_pos = term_count(counterfactual, POSITIVE_TERMS)
    cf_neg = term_count(counterfactual, NEGATIVE_TERMS)
    cf_type = str(task.get("counterfactual_type") or "")
    side = expected_side(task)
    if side == "up":
        relevant_original = original_pos
        opposite_original = original_neg
        relevant_cf = cf_pos
    elif side == "down":
        relevant_original = original_neg
        opposite_original = original_pos
        relevant_cf = cf_neg
    else:
        relevant_original = abs(float(task.get("evidence_polarity_score") or 0.0))
        opposite_original = 0.0
        relevant_cf = 0.0
    is_news_task = cf_type in {
        "remove_positive_evidence",
        "remove_negative_evidence",
        "neutralize_positive_evidence",
        "neutralize_negative_evidence",
        "remove_all_company_evidence",
    }
    is_technical_task = cf_type in {"neutralize_bearish_technical", "neutralize_bullish_technical"}
    dominant = bool(relevant_original > 0 and relevant_original > opposite_original)
    residual_removed = bool(relevant_cf == 0)
    tech_changed = token_changed(task)
    quality_pass = bool((is_news_task and dominant and residual_removed) or (is_technical_task and tech_changed))
    quality_score = float((relevant_original - opposite_original) + max(0, relevant_original - relevant_cf))
    if is_technical_task:
        quality_score = 1.0 if tech_changed else -1.0
    return {
        "original_pos_terms": int(original_pos),
        "original_neg_terms": int(original_neg),
        "counterfactual_pos_terms": int(cf_pos),
        "counterfactual_neg_terms": int(cf_neg),
        "expected_side": side,
        "relevant_original_terms": float(relevant_original),
        "opposite_original_terms": float(opposite_original),
        "relevant_counterfactual_terms": float(relevant_cf),
        "dominant_expected_polarity": dominant,
        "residual_expected_polarity_removed": residual_removed,
        "token_changed": tech_changed,
        "placeholder_counterfactual": placeholder_counterfactual(task),
        "quality_pass": quality_pass,
        "quality_score": quality_score,
    }


def candidate_tasks(contexts: pd.DataFrame, *, repair_placeholders: bool, semantic_neutralization: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cf_type in TASK_TYPES:
        for _, context_row in contexts.iterrows():
            task = build_task(context_row, cf_type)
            if task is None:
                continue
            before_placeholder = placeholder_counterfactual(task)
            if repair_placeholders and before_placeholder:
                task = repair_placeholder(task)
            semantic_repaired = False
            if semantic_neutralization and cf_type in {"neutralize_positive_evidence", "neutralize_negative_evidence"}:
                task = repair_semantic_neutralization(task)
                semantic_repaired = True
            features = quality_features(task)
            rows.append(
                {
                    **task,
                    **features,
                    "original_placeholder_counterfactual": before_placeholder,
                    "placeholder_repaired": bool(repair_placeholders and before_placeholder),
                    "semantic_neutralization_repaired": semantic_repaired,
                }
            )
    return pd.DataFrame(rows)


def select_balanced_tasks(candidates: pd.DataFrame, *, per_type_limit: int, min_per_type: int) -> tuple[pd.DataFrame, list[str]]:
    failures: list[str] = []
    selected_parts: list[pd.DataFrame] = []
    usable = candidates[candidates["quality_pass"].astype(bool)].copy()
    for cf_type in TASK_TYPES:
        group = usable[usable["counterfactual_type"].eq(cf_type)].copy()
        group = group.sort_values(["quality_score", "sample_id"], ascending=[False, True])
        selected = group.head(per_type_limit).copy()
        if len(selected) < min_per_type:
            failures.append(f"{cf_type} quality-filtered tasks {len(selected)} < {min_per_type}")
        selected_parts.append(selected)
    if not selected_parts:
        return pd.DataFrame(), failures
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["counterfactual_type", "quality_score", "sample_id"], ascending=[True, False, True])
    ordered_rows = []
    max_len = max((len(selected[selected["counterfactual_type"].eq(cf_type)]) for cf_type in TASK_TYPES), default=0)
    for idx in range(max_len):
        for cf_type in TASK_TYPES:
            group = selected[selected["counterfactual_type"].eq(cf_type)]
            if idx < len(group):
                ordered_rows.append(group.iloc[idx])
    if ordered_rows:
        selected = pd.DataFrame(ordered_rows)
    return selected.reset_index(drop=True), failures


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            keep = {
                key: value
                for key, value in row.items()
                if key
                in {
                    "sample_id",
                    "ticker",
                    "event_date",
                    "split",
                    "v6_track",
                    "counterfactual_type",
                    "expected_direction",
                    "evidence_polarity_score",
                    "original_headline",
                    "original_body",
                    "original_technical_event_tokens_json",
                    "counterfactual_headline",
                    "counterfactual_body",
                    "counterfactual_technical_event_tokens_json",
                }
            }
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")


def summarize_by_type(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cf_type in TASK_TYPES:
        cand = candidates[candidates["counterfactual_type"].eq(cf_type)]
        sel = selected[selected["counterfactual_type"].eq(cf_type)]
        rows.append(
            {
                "counterfactual_type": cf_type,
                "candidate_tasks": int(len(cand)),
                "quality_pass_tasks": int(cand["quality_pass"].sum()) if len(cand) else 0,
                "selected_tasks": int(len(sel)),
                "placeholder_candidate_rate": float(cand["placeholder_counterfactual"].mean()) if len(cand) else 0.0,
                "original_placeholder_candidate_rate": float(cand["original_placeholder_counterfactual"].mean()) if len(cand) else 0.0,
                "placeholder_repaired_selected_rate": float(sel["placeholder_repaired"].mean()) if len(sel) else 0.0,
                "semantic_neutralization_repaired_selected_rate": float(sel["semantic_neutralization_repaired"].mean()) if len(sel) else 0.0,
                "mean_quality_score_selected": float(sel["quality_score"].mean()) if len(sel) else 0.0,
                "mean_original_pos_terms_selected": float(sel["original_pos_terms"].mean()) if len(sel) else 0.0,
                "mean_original_neg_terms_selected": float(sel["original_neg_terms"].mean()) if len(sel) else 0.0,
                "mean_counterfactual_pos_terms_selected": float(sel["counterfactual_pos_terms"].mean()) if len(sel) else 0.0,
                "mean_counterfactual_neg_terms_selected": float(sel["counterfactual_neg_terms"].mean()) if len(sel) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="data/processed/current_v6_prediction_contexts.parquet")
    parser.add_argument("--output", default="data/eval/current_v6_counterfactual_quality_filtered_tasks.jsonl")
    parser.add_argument("--candidate-output", default="outputs/tables/16_6_v6_counterfactual_quality_candidates.csv")
    parser.add_argument("--selected-output", default="outputs/tables/16_6_v6_counterfactual_quality_selected.csv")
    parser.add_argument("--summary-output", default="outputs/tables/16_6_v6_counterfactual_quality_by_type.csv")
    parser.add_argument("--metrics", default="outputs/metrics/16_6_v6_counterfactual_quality_filtered_tasks.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--per-type-limit", type=int, default=30)
    parser.add_argument("--min-per-type", type=int, default=8)
    parser.add_argument("--no-repair-placeholders", action="store_true")
    parser.add_argument("--semantic-neutralization", action="store_true")
    parser.add_argument("--step-name", default=STEP)
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.contexts).exists():
        failures.append(f"contexts missing: {args.contexts}")
        contexts = pd.DataFrame()
    else:
        contexts = pd.read_parquet(args.contexts)
    if len(contexts) and "split" in contexts.columns:
        contexts = contexts[contexts["split"].eq("test")].copy()
    if contexts.empty:
        failures.append("test contexts are empty")

    candidates = pd.DataFrame()
    selected = pd.DataFrame()
    summary = pd.DataFrame()
    metrics: dict[str, Any] = {"pipeline_pass": False, "claim_allowed": False}
    if not failures:
        candidates = candidate_tasks(
            contexts,
            repair_placeholders=not args.no_repair_placeholders,
            semantic_neutralization=args.semantic_neutralization,
        )
        selected, selection_failures = select_balanced_tasks(
            candidates,
            per_type_limit=args.per_type_limit,
            min_per_type=args.min_per_type,
        )
        failures.extend(selection_failures)
        summary = summarize_by_type(candidates, selected)
        if selected.empty:
            failures.append("no quality-filtered counterfactual tasks selected")
        metrics = {
            "pipeline_pass": not failures,
            "claim_allowed": False,
            "context_rows": int(len(contexts)),
            "candidate_tasks": int(len(candidates)),
            "quality_pass_tasks": int(candidates["quality_pass"].sum()) if len(candidates) else 0,
            "selected_tasks": int(len(selected)),
            "selected_sample_count": int(selected["sample_id"].astype(str).nunique()) if len(selected) else 0,
            "per_type_limit": int(args.per_type_limit),
            "min_per_type": int(args.min_per_type),
            "repair_placeholders": bool(not args.no_repair_placeholders),
            "semantic_neutralization": bool(args.semantic_neutralization),
            "placeholder_candidate_rate": float(candidates["placeholder_counterfactual"].mean()) if len(candidates) else 0.0,
            "original_placeholder_candidate_rate": float(candidates["original_placeholder_counterfactual"].mean()) if len(candidates) else 0.0,
            "placeholder_repaired_selected_rate": float(selected["placeholder_repaired"].mean()) if len(selected) else 0.0,
            "semantic_neutralization_repaired_selected_rate": float(selected["semantic_neutralization_repaired"].mean()) if len(selected) else 0.0,
            "quality_protocol": (
                "select tasks with dominant expected polarity and removed residual expected-polarity terms; "
                "technical tasks require changed technical-token JSON"
            ),
            "claim_scope": "task_quality_filter_only; requires LLM counterfactual rerun before any faithfulness claim",
        }

    Path(args.candidate_output).parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.candidate_output, index=False)
    Path(args.selected_output).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.selected_output, index=False)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    write_jsonl(args.output, selected.to_dict(orient="records") if len(selected) else [])
    metrics["pipeline_pass"] = not failures
    write_json(args.metrics, metrics)
    write_manifest(
        args.manifest,
        [args.contexts, args.output, args.candidate_output, args.selected_output, args.summary_output, args.metrics],
        args.step_name,
        extra={
            "references": [
                "CheckList behavioral testing: directional behavioral tests should target specific capabilities",
                "Contrast sets: perturbations should be small, meaningful, and locally diagnostic",
                "Counterfactually augmented data: revisions should be internally coherent and avoid gratuitous changes",
            ]
        },
    )
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        args.step_name,
        status,
        [args.contexts],
        [args.output, args.candidate_output, args.selected_output, args.summary_output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
