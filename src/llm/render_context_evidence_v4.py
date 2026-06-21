from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataclean_v4_utils import clean_string, parse_json_list
from src.utils.artifacts import write_manifest, write_status

STEP = "08_RENDER_CONTEXT_EVIDENCE_V4"


def parse_evidence_pack(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = clean_string(value)
    if not text:
        return {"company_evidence": [], "context_evidence": [], "technical_signals": []}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"company_evidence": [], "context_evidence": [], "technical_signals": []}
    except Exception:
        return {"company_evidence": [], "context_evidence": [], "technical_signals": []}


def clip_text(value: Any, max_chars: int) -> str:
    text = clean_string(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [TRUNCATED]"


def normalize_strength(value: Any) -> str:
    text = clean_string(value).lower()
    if text in {"high", "strong"}:
        return "strong"
    if text in {"low", "weak"}:
        return "weak"
    return "medium"


def context_meta_from_row(row: Any) -> dict[str, Any]:
    pack = parse_evidence_pack(row.get("evidence_pack_json", ""))
    company = [item for item in pack.get("company_evidence", []) if isinstance(item, dict)]
    context = [item for item in pack.get("context_evidence", []) if isinstance(item, dict)]
    tech = [item for item in pack.get("technical_signals", []) if isinstance(item, dict)]
    return {
        "company_evidence_ids": [clean_string(item.get("evidence_id")) for item in company if clean_string(item.get("evidence_id"))],
        "context_evidence_ids": [clean_string(item.get("evidence_id")) for item in context if clean_string(item.get("evidence_id"))],
        "evidence_ids": [
            clean_string(item.get("evidence_id"))
            for item in [*company, *context]
            if clean_string(item.get("evidence_id"))
        ],
        "signal_ids": [f"T{idx}" for idx, _ in enumerate(tech, start=1)],
    }


def render_evidence_context(row: Any) -> str:
    pack = parse_evidence_pack(row.get("evidence_pack_json", ""))
    company = [item for item in pack.get("company_evidence", []) if isinstance(item, dict)]
    context = [item for item in pack.get("context_evidence", []) if isinstance(item, dict)]
    tech = [item for item in pack.get("technical_signals", []) if isinstance(item, dict)]
    lines = [
        f"Ticker: {clean_string(row.get('ticker'))}",
        f"Date: {clean_string(row.get('event_date'))}",
        "",
        "Company-specific evidence:",
    ]
    if company:
        for item in company:
            lines.extend(
                [
                    f"[{clean_string(item.get('evidence_id'))}] type={clean_string(item.get('article_type'))} quality={float(item.get('evidence_quality_score', 0.0) or 0.0):.3f}",
                    f"Headline: {clip_text(item.get('headline'), 220)}",
                    f"Excerpt: {clip_text(item.get('body_excerpt'), 360)}",
                ]
            )
    else:
        lines.append("None")
    lines.extend(["", "Context-only evidence:"])
    if context:
        for item in context:
            lines.extend(
                [
                    f"[{clean_string(item.get('evidence_id'))}] type={clean_string(item.get('article_type'))} quality={float(item.get('evidence_quality_score', 0.0) or 0.0):.3f}",
                    f"Headline: {clip_text(item.get('headline'), 220)}",
                    f"Excerpt: {clip_text(item.get('body_excerpt'), 360)}",
                ]
            )
    else:
        lines.append("None")
    lines.extend(["", "Technical signals:"])
    if tech:
        for idx, item in enumerate(tech, start=1):
            lines.append(
                f"[T{idx}] token={clean_string(item.get('token'))} "
                f"direction={clean_string(item.get('direction_prior', item.get('direction', 'neutral')))} "
                f"strength={normalize_strength(item.get('strength', 'medium'))} "
                f"value={clean_string(item.get('value'))} rule={clean_string(item.get('rule'))}"
            )
    else:
        tokens = parse_json_list(row.get("technical_event_tokens_json", ""))
        if tokens:
            for idx, item in enumerate(tokens, start=1):
                if isinstance(item, dict):
                    lines.append(
                        f"[T{idx}] token={clean_string(item.get('token'))} "
                        f"direction={clean_string(item.get('direction_prior', item.get('direction', 'neutral')))} "
                        f"strength={normalize_strength(item.get('strength', 'medium'))}"
                    )
        else:
            lines.append("None")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/ticker_date_evidence_contexts_h1_v4_small.parquet")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output", default="outputs/data_samples/rendered_context_evidence_v4_small.txt")
    parser.add_argument("--status", default=f"outputs/status/{STEP}_SMALL.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}_SMALL.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    df = pd.read_parquet(args.input) if Path(args.input).exists() else pd.DataFrame()
    if df.empty:
        failures.append(f"input missing or empty: {args.input}")
    sample = df.head(args.num_samples).copy() if not df.empty else pd.DataFrame()
    rendered = [render_evidence_context(row) for _, row in sample.iterrows()]
    text = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(rendered)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf-8")
    has_evidence_id = any("[N" in item or "Company-specific evidence:\nNone" in item for item in rendered)
    has_signal_id = any("[T" in item for item in rendered)
    avg_chars = sum(len(item) for item in rendered) / len(rendered) if rendered else 0.0
    if not has_evidence_id:
        failures.append("rendered samples do not include evidence IDs or explicit None")
    if not has_signal_id:
        failures.append("rendered samples do not include technical signal IDs")
    if avg_chars > 3500:
        failures.append(f"average rendered chars {avg_chars:.1f} > 3500")
    metrics = {
        "sample_rows": int(len(sample)),
        "avg_rendered_chars": float(avg_chars),
        "has_evidence_id_or_none": bool(has_evidence_id),
        "has_signal_id": bool(has_signal_id),
    }
    write_manifest(args.manifest, [args.output], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.input], [args.output, args.manifest, args.status], metrics, failures, status == "PASS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
