from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "10_FLOW_REWARD_V6_DECISION_TARGETS"
FORECAST_KEYS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
PROB_COLS = [f"p_{key}" for key in FORECAST_KEYS]
AUXILIARY_FIELDS = [
    "true_label_probability_ensemble",
    "judge_reliability_weight",
    "label_order_kl_mean",
    "judge_disagreement_entropy",
    "news_grounding_score",
    "technical_grounding_score",
    "unsupported_news_claim_rate",
    "evidence_quality_weight",
    "abnormal_return_h1",
    "raw_realized_utility",
    "technical_rule_delta",
    "target_label_5",
    "hard_event_track",
    "volatility_regime",
    "source_split",
]


def stable_val_mask(sample_ids: list[Any], frac: float = 0.2) -> np.ndarray:
    mask = []
    for value in sample_ids:
        digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
        mask.append(int.from_bytes(digest[:4], "little") % 10_000 < int(frac * 10_000))
    out = np.asarray(mask, dtype=bool)
    minimum = max(1, int(round(len(out) * frac)))
    if len(out) and out.sum() < minimum:
        out[:] = False
        out[-minimum:] = True
    return out


def hash_embedding(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def finite_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def action_from_distribution(row: pd.Series) -> str:
    down = float(row.get("p_strong_down", 0.0) or 0.0) + float(row.get("p_mild_down", 0.0) or 0.0)
    up = float(row.get("p_mild_up", 0.0) or 0.0) + float(row.get("p_strong_up", 0.0) or 0.0)
    neutral = float(row.get("p_neutral", 0.0) or 0.0)
    if up > max(down, neutral):
        return "long"
    if down > max(up, neutral):
        return "short"
    return "hold"


def action_value(action: Any) -> float:
    return {"short": -1.0, "hold": 0.0, "long": 1.0}.get(str(action).lower(), 0.0)


def technical_rule_action(tokens_json: Any) -> str:
    try:
        tokens = json.loads(str(tokens_json))
    except Exception:
        tokens = []
    score = 0.0
    for item in tokens if isinstance(tokens, list) else []:
        prior = str(item.get("direction_prior", "")).lower()
        strength = str(item.get("strength", "medium")).lower()
        weight = {"weak": 0.5, "low": 0.5, "medium": 1.0, "high": 1.5, "strong": 1.5}.get(strength, 1.0)
        if "bullish" in prior or "positive" in prior or "outperformance" in prior:
            score += weight
        if "bearish" in prior or "negative" in prior or "underperformance" in prior:
            score -= weight
    if score > 0.5:
        return "long"
    if score < -0.5:
        return "short"
    return "hold"


def volatility_regime(tokens_json: Any) -> str:
    text = str(tokens_json).lower()
    if "high_vol" in text or "high volatility" in text:
        return "high_vol"
    if "low_vol" in text or "low volatility" in text:
        return "low_vol"
    if "normal_vol" in text:
        return "normal_vol"
    return "unknown"


def grounding_scores(row: pd.Series) -> tuple[float, float, float]:
    parsed = parse_json(row.get("parsed_json"))
    meta = parse_json(row.get("context_meta_json"))
    evidence_ids = set(str(x) for x in meta.get("evidence_ids", []))
    signal_ids = set(str(x) for x in meta.get("signal_ids", []))

    news_items = parsed.get("news_rationale", [])
    tech_items = parsed.get("technical_rationale", [])
    news_items = news_items if isinstance(news_items, list) else []
    tech_items = tech_items if isinstance(tech_items, list) else []

    if news_items:
        supported_news = sum(1 for item in news_items if str(item.get("evidence_id", "")) in evidence_ids)
        news_score = supported_news / len(news_items)
        unsupported_rate = 1.0 - news_score
    else:
        news_score = 1.0 if not evidence_ids else 0.5
        unsupported_rate = 0.0

    if tech_items:
        supported_tech = sum(1 for item in tech_items if str(item.get("signal_id", "")) in signal_ids)
        tech_score = supported_tech / len(tech_items)
    else:
        tech_score = 0.5

    return float(news_score), float(tech_score), float(unsupported_rate)


def normalize_by_max(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    max_val = float(vals.max()) if len(vals) else 0.0
    if max_val <= 1e-12:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (vals / max_val).clip(0.0, 1.0)


def build_dataset(rationales: pd.DataFrame, judge: pd.DataFrame, contexts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["sample_id", "candidate_id"]
    df = rationales.copy()
    if df.empty or judge.empty or contexts.empty:
        return pd.DataFrame(), {
            "target": np.zeros((0, 5), dtype=np.float32),
            "mask": np.zeros((0, 5), dtype=np.float32),
            "cond": np.zeros((0, 128), dtype=np.float32),
            "split": [],
            "target_names": FORECAST_KEYS,
        }

    df["candidate_id"] = df["candidate_id"].astype(int)
    judge = judge.copy()
    judge["candidate_id"] = judge["candidate_id"].astype(int)
    context_cols = [
        col
        for col in [
            "sample_id",
            "split",
            "abnormal_return_h1",
            "target_return",
            "target_label_5",
            "mean_evidence_quality_score",
            "technical_event_tokens_json",
            "clean_context_text",
            "v6_track",
            "v6_training_weight",
        ]
        if col in contexts.columns
    ]

    df = df.merge(judge, on=keys, how="inner", suffixes=("", "_judge"))
    df = df.merge(contexts[context_cols].drop_duplicates("sample_id"), on="sample_id", how="left", suffixes=("", "_context"))
    if "split_context" in df.columns:
        df["source_split"] = df["split_context"].fillna(df.get("split", "unknown"))
    else:
        df["source_split"] = df.get("split", pd.Series("unknown", index=df.index))

    returns = finite_series(df, "abnormal_return_h1").fillna(finite_series(df, "target_return", 0.0)).fillna(0.0)
    df["action_from_distribution"] = df.apply(action_from_distribution, axis=1)
    df["technical_rule_action"] = df["technical_event_tokens_json"].apply(technical_rule_action)
    transaction_cost = 0.001
    action_values = df["action_from_distribution"].apply(action_value)
    technical_values = df["technical_rule_action"].apply(action_value)
    df["abnormal_return_h1"] = returns
    df["raw_realized_utility"] = action_values * returns - action_values.abs() * transaction_cost
    df["technical_rule_utility"] = technical_values * returns - technical_values.abs() * transaction_cost
    df["technical_rule_delta"] = df["raw_realized_utility"] - df["technical_rule_utility"]
    df["evidence_quality_weight"] = finite_series(df, "mean_evidence_quality_score", 0.5).fillna(0.5).clip(0.0, 1.0)
    grounding = df.apply(grounding_scores, axis=1, result_type="expand")
    grounding.columns = ["news_grounding_score", "technical_grounding_score", "unsupported_news_claim_rate"]
    for col in grounding.columns:
        df[col] = grounding[col].clip(0.0, 1.0)

    norm_kl = normalize_by_max(df["label_order_kl_mean"])
    norm_entropy = normalize_by_max(df["judge_disagreement_entropy"])
    schema_factor = df["judge_schema_ok"].astype(float)
    df["judge_reliability_weight"] = (
        schema_factor
        * finite_series(df, "argmax_consistency_ensemble", 0.0).fillna(0.0).clip(0.0, 1.0)
        * (1.0 - norm_kl)
        * (1.0 - norm_entropy)
        * df["evidence_quality_weight"]
    ).clip(0.0, 1.0)
    df["hard_event_track"] = df.get("v6_track", pd.Series("unknown", index=df.index)).fillna("unknown")
    df["volatility_regime"] = df["technical_event_tokens_json"].apply(volatility_regime)

    val_mask = stable_val_mask(df["sample_id"].tolist(), frac=0.2)
    flow_split = np.where(val_mask, "val", "train")
    df["flow_split"] = flow_split
    cond_text = (
        df.get("parsed_json", pd.Series("", index=df.index)).fillna("")
        + "\n"
        + df.get("clean_context_text", pd.Series("", index=df.index)).fillna("")
    )
    cond = np.vstack([hash_embedding(text, dim=128) for text in cond_text]).astype(np.float32)
    target = df[PROB_COLS].to_numpy(dtype=np.float32)
    mask = np.ones_like(target, dtype=np.float32)
    data = {
        "target": target,
        "mask": mask,
        "cond": cond,
        "split": flow_split.tolist(),
        "source_split": df["source_split"].astype(str).tolist(),
        "sample_id": df["sample_id"].astype(str).tolist(),
        "candidate_id": df["candidate_id"].astype(int).tolist(),
        "target_names": FORECAST_KEYS,
        "auxiliary_fields": AUXILIARY_FIELDS,
        "auxiliary": {field: df[field].tolist() for field in AUXILIARY_FIELDS},
        "metadata": {
            "target_source": "calibrated_debiased_judge_distribution",
            "utility_is_auxiliary": True,
            "condition_backend": "hash_blake2b_128",
        },
    }
    return df, data


def evaluate_gate(df: pd.DataFrame, data: dict[str, Any], min_rows: int = 2700) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    target = np.asarray(data["target"], dtype=np.float32)
    cond = np.asarray(data["cond"], dtype=np.float32)
    split = list(data.get("split", []))
    source_split = list(data.get("source_split", []))
    target_sums = target.sum(axis=1) if target.ndim == 2 and target.shape[0] else np.asarray([], dtype=float)
    reliability = np.asarray(data.get("auxiliary", {}).get("judge_reliability_weight", []), dtype=float)
    raw_utility = np.asarray(data.get("auxiliary", {}).get("raw_realized_utility", []), dtype=float)
    tech_delta = np.asarray(data.get("auxiliary", {}).get("technical_rule_delta", []), dtype=float)

    metrics = {
        "pipeline_pass": True,
        "claim_allowed": False,
        "rows": int(target.shape[0]) if target.ndim == 2 else 0,
        "target_dim": int(target.shape[1]) if target.ndim == 2 else 0,
        "cond_dim": int(cond.shape[1]) if cond.ndim == 2 else 0,
        "split_distribution": pd.Series(split).value_counts().to_dict() if split else {},
        "source_split_distribution": pd.Series(source_split).value_counts().to_dict() if source_split else {},
        "target_source": "calibrated_debiased_judge_distribution",
        "utility_is_auxiliary": True,
        "target_sum_min": float(target_sums.min()) if len(target_sums) else float("nan"),
        "target_sum_max": float(target_sums.max()) if len(target_sums) else float("nan"),
        "target_sum_ok_rate": float(np.mean(np.abs(target_sums - 1.0) <= 1e-4)) if len(target_sums) else 0.0,
        "judge_reliability_weight_min": float(reliability.min()) if len(reliability) else float("nan"),
        "judge_reliability_weight_max": float(reliability.max()) if len(reliability) else float("nan"),
        "raw_realized_utility_finite_rate": float(np.isfinite(raw_utility).mean()) if len(raw_utility) else 0.0,
        "technical_rule_delta_finite_rate": float(np.isfinite(tech_delta).mean()) if len(tech_delta) else 0.0,
        "hard_event_track_present": bool("hard_event_track" in data.get("auxiliary", {})),
        "volatility_regime_present": bool("volatility_regime" in data.get("auxiliary", {})),
    }

    if metrics["rows"] < min_rows:
        failures.append(f"rows {metrics['rows']} < {min_rows}")
    if metrics["target_dim"] != 5:
        failures.append(f"target_dim {metrics['target_dim']} != 5")
    if not {"train", "val"}.issubset(set(split)):
        failures.append("train/val split missing")
    if metrics["target_sum_ok_rate"] < 1.0:
        failures.append(f"target_sum_ok_rate {metrics['target_sum_ok_rate']:.4f} < 1.0")
    if metrics["cond_dim"] < 128:
        failures.append(f"cond_dim {metrics['cond_dim']} < 128")
    if len(reliability) == 0 or not np.isfinite(reliability).all() or reliability.min() < -1e-8 or reliability.max() > 1 + 1e-8:
        failures.append("judge_reliability_weight is not finite in [0,1]")
    if "test" in set(str(x).lower() for x in source_split):
        failures.append("source_split contains test rows")
    if not metrics["hard_event_track_present"] or not metrics["volatility_regime_present"]:
        failures.append("track/regime auxiliary fields missing")
    if metrics["raw_realized_utility_finite_rate"] < 1.0 or metrics["technical_rule_delta_finite_rate"] < 1.0:
        failures.append("raw_realized_utility or technical_rule_delta has non-finite values")
    metrics["pipeline_pass"] = not failures
    return metrics, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--min-rows", type=int, default=2700)
    args = parser.parse_args()

    failures: list[str] = []
    inputs = [args.rationales, args.judge, args.contexts]
    for path in inputs:
        if not Path(path).exists():
            failures.append(f"missing input: {path}")

    if failures:
        df = pd.DataFrame()
        data = {
            "target": np.zeros((0, 5), dtype=np.float32),
            "mask": np.zeros((0, 5), dtype=np.float32),
            "cond": np.zeros((0, 128), dtype=np.float32),
            "split": [],
            "source_split": [],
            "target_names": FORECAST_KEYS,
            "auxiliary_fields": AUXILIARY_FIELDS,
            "auxiliary": {field: [] for field in AUXILIARY_FIELDS},
            "metadata": {"target_source": "calibrated_debiased_judge_distribution", "utility_is_auxiliary": True},
        }
        metrics = {"pipeline_pass": False, "claim_allowed": False, "rows": 0, "target_dim": 5, "cond_dim": 128}
    else:
        rationales = pd.read_parquet(args.rationales)
        judge = pd.read_parquet(args.judge)
        contexts = pd.read_parquet(args.contexts)
        df, data = build_dataset(rationales, judge, contexts)
        metrics, gate_failures = evaluate_gate(df, data, min_rows=args.min_rows)
        failures.extend(gate_failures)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs,
        [args.output, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "failures": failures, "metrics": metrics}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
