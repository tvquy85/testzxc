from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "14_RWSFT_DPO_DATASET_REBUILD"
SCORE_COMPONENT_WEIGHTS = {
    "inferability_true_label_prob": 0.45,
    "multi_judge_agreement": 0.20,
    "news_grounding_score": 0.15,
    "technical_grounding_score": 0.15,
    "schema_ok_score": 0.05,
}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _score_components(row) -> tuple[float | None, dict[str, float]]:
    components: dict[str, float] = {}
    for name, weight in SCORE_COMPONENT_WEIGHTS.items():
        value = _as_float(row.get(name))
        if value is None:
            continue
        components[name] = max(0.0, min(1.0, value))
    if not components:
        return None, {}
    observed_weight = sum(SCORE_COMPONENT_WEIGHTS[name] for name in components)
    score = sum(components[name] * SCORE_COMPONENT_WEIGHTS[name] for name in components) / observed_weight
    return float(score), components


def attach_alignment_scores(rationales, inferability, grounding):
    import pandas as pd

    df = rationales.copy()
    if df.empty:
        df["alignment_proxy_score"] = []
        return df
    if "split" in df.columns:
        df = df[df["split"] == "train"].copy()
    if not inferability.empty:
        inf = inferability.groupby(["sample_id", "candidate_id"], dropna=False).agg(
            inferability_true_label_prob=("inferability_true_label_prob", "mean"),
            inferability_std=("inferability_true_label_prob", "std"),
        ).reset_index()
        inf["multi_judge_agreement"] = 1.0 - inf["inferability_std"].fillna(0.0).clip(0, 1)
        df = df.merge(inf.drop(columns=["inferability_std"]), on=["sample_id", "candidate_id"], how="left")
    if not grounding.empty:
        g = grounding.groupby(["sample_id", "candidate_id"], dropna=False).agg(
            news_grounding_score=("news_grounding_score", "mean"),
            technical_grounding_score=("technical_grounding_score", "mean"),
        ).reset_index()
        df = df.merge(g, on=["sample_id", "candidate_id"], how="left")
    df["schema_ok_score"] = df.get("schema_ok", pd.Series([False] * len(df))).fillna(False).astype(bool).astype(float)
    scores = []
    component_names = []
    for _, row in df.iterrows():
        score, components = _score_components(row)
        scores.append(score)
        component_names.append(sorted(components))
    df["alignment_proxy_score"] = scores
    df["alignment_score_components"] = [json.dumps(names) for names in component_names]
    return df


def build_messages(row) -> list[dict[str, str]]:
    prompt = row.get("prompt")
    raw_text = row.get("raw_text", row.get("raw_output", ""))
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(raw_text)}]
    return [{"role": "assistant", "content": str(raw_text)}]


def build_rwsft_records(scored):
    if scored.empty:
        return []
    valid = scored[scored["alignment_proxy_score"].notna()].copy()
    best = valid.sort_values(["sample_id", "alignment_proxy_score"], ascending=[True, False]).groupby("sample_id", as_index=False).head(1)
    records = []
    for _, row in best.iterrows():
        records.append(
            {
                "sample_id": row["sample_id"],
                "candidate_id": int(row.get("candidate_id", 0)),
                "messages": build_messages(row),
                "raw_text": row.get("raw_text"),
                "weight": float(row["alignment_proxy_score"]),
                "score_components": row.get("alignment_score_components"),
                "split": row.get("split"),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", default="data/rationales/parsed/train_candidates_stage1_strict.parquet")
    parser.add_argument("--inferability", default="data/judges/inferability_multi_judge_stage1.parquet")
    parser.add_argument("--grounding", default="data/judges/claim_grounding_scores_stage1.parquet")
    parser.add_argument("--flow-checkpoint", default="checkpoints/flow_reward_v2_stage1")
    parser.add_argument("--output", default="data/alignment/rwsft_train_v2.jsonl")
    parser.add_argument("--summary", default="outputs/metrics/alignment_dataset_v2_summary.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if Path(args.rationales).exists() else pd.DataFrame()
    inferability = pd.read_parquet(args.inferability) if Path(args.inferability).exists() else pd.DataFrame()
    grounding = pd.read_parquet(args.grounding) if Path(args.grounding).exists() else pd.DataFrame()
    if rationales.empty:
        failures.append("rationales input is empty")
    if "split" in rationales.columns and set(rationales["split"].dropna()) - {"train"}:
        failures.append("non-train rows found in rationales")
    scored = attach_alignment_scores(rationales, inferability, grounding)
    records = build_rwsft_records(scored)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "rwsft_examples": int(len(records)),
        "rwsft_mvp_gate": int(len(records)) >= 5000,
        "rwsft_target_gate": int(len(records)) >= 20000,
        "ranking_source": "alignment_proxy_from_step10_11_flow_targets",
        "scored_rows": int(scored["alignment_proxy_score"].notna().sum()) if len(scored) else 0,
    }
    if len(records) < 5000:
        failures.append("RWSFT examples < 5,000 MVP gate")
    write_json(args.summary, summary)
    write_manifest(args.manifest, [args.output, args.summary], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.rationales, args.inferability, args.grounding, args.flow_checkpoint],
        [args.output, args.summary, args.manifest, args.status],
        summary,
        failures,
        status == "PASS",
    )
    print(json.dumps(summary, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
