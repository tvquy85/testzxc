from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.judges.judge_calibration_v6 import expected_calibration_error
from src.reward.evaluate_flow_vs_proxy_v4 import preference_pair_accuracy, spearman, top_decile_utility
from src.reward.flow_model_v2 import FlowRewardV2
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "11_FLOW_EVAL_RAW_UTILITY_AND_PROXY"
FORECAST_KEYS = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
REQUIRED_METHODS = [
    "flow_reward_v6",
    "proxy_average_reward",
    "single_best_judge_reward",
    "technical_rule_reward",
    "flow_reward_without_reliability_weight",
    "flow_reward_without_grounding_auxiliary",
]


def metric_wins(flags: dict[str, bool]) -> bool:
    return sum(bool(v) for v in flags.values()) >= 2


def integrate_flow(model: FlowRewardV2, cond: np.ndarray, device: torch.device, steps: int = 16, batch_size: int = 512) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, cond.shape[0], batch_size):
            c = torch.tensor(cond[start : start + batch_size], dtype=torch.float32, device=device)
            z = torch.zeros((c.shape[0], model.target_dim), dtype=torch.float32, device=device)
            for idx in range(steps):
                t = torch.full((c.shape[0], 1), float(idx) / float(steps), dtype=torch.float32, device=device)
                z = z + model(t, z, c) / float(steps)
            preds.append(z.detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, model.target_dim), dtype=float)


def project_rows_to_simplex(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for row_idx, row in enumerate(values.astype(float)):
        finite = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        sorted_values = np.sort(finite)[::-1]
        cumulative = 0.0
        theta = 0.0
        for idx, value in enumerate(sorted_values, start=1):
            cumulative += float(value)
            candidate = (cumulative - 1.0) / idx
            if value - candidate > 0:
                theta = candidate
        projected = np.maximum(finite - theta, 0.0)
        total = projected.sum()
        out[row_idx] = projected / total if total > 0 else np.ones(values.shape[1]) / values.shape[1]
    return out


def canonical_label(value: Any) -> str:
    text = str(value or "neutral").strip().lower().replace(" ", "_").replace("-", "_")
    return text if text in FORECAST_KEYS else "neutral"


def target_indices(labels: list[Any]) -> np.ndarray:
    return np.asarray([FORECAST_KEYS.index(canonical_label(label)) for label in labels], dtype=int)


def brier_multiclass(probs: np.ndarray, y_true: np.ndarray) -> float:
    one_hot = np.zeros_like(probs, dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def kl_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0)
    q = np.clip(q, 1e-9, 1.0)
    return np.sum(p * np.log(p / q), axis=1)


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_rows(p, m) + 0.5 * kl_divergence_rows(q, m)


def proxy_average_score(aux: dict[str, list[Any]]) -> np.ndarray:
    names = [
        "true_label_probability_ensemble",
        "judge_reliability_weight",
        "news_grounding_score",
        "technical_grounding_score",
        "evidence_quality_weight",
    ]
    vals = []
    for name in names:
        vals.append(np.asarray(aux.get(name, []), dtype=float))
    arr = np.vstack(vals).T
    return np.nanmean(arr, axis=1)


def method_metrics(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    rank = spearman(df[score_col], df["raw_realized_utility"])
    pair = preference_pair_accuracy(df, score_col, "raw_realized_utility")
    top = top_decile_utility(df, score_col, "raw_realized_utility")
    tech_top = top_decile_utility(df, score_col, "technical_rule_delta")
    top_news = top_decile_utility(df, score_col, "news_grounding_score")
    top_tech = top_decile_utility(df, score_col, "technical_grounding_score")
    top_unsupported = top_decile_utility(df, score_col, "unsupported_news_claim_rate")
    return {
        "rank_correlation_with_raw_realized_utility": None if rank is None else float(rank),
        "preference_pair_accuracy_by_raw_utility": None if pair is None else float(pair),
        "top_decile_raw_realized_utility": None if top is None else float(top),
        "technical_rule_delta_top_decile": None if tech_top is None else float(tech_top),
        "top_decile_news_grounding_score": None if top_news is None else float(top_news),
        "top_decile_technical_grounding_score": None if top_tech is None else float(top_tech),
        "unsupported_claim_rate_top_decile": None if top_unsupported is None else float(top_unsupported),
    }


def group_wins(df: pd.DataFrame, group_col: str, flow_col: str, proxy_col: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if group_col not in df.columns:
        return out
    for value, group in df.groupby(group_col):
        if len(group) < 10:
            continue
        flow_rank = spearman(group[flow_col], group["raw_realized_utility"])
        proxy_rank = spearman(group[proxy_col], group["raw_realized_utility"])
        out[str(value)] = {
            "rows": int(len(group)),
            "flow_rank": None if flow_rank is None else float(flow_rank),
            "proxy_rank": None if proxy_rank is None else float(proxy_rank),
            "flow_rank_win": bool((flow_rank or 0.0) >= (proxy_rank or 0.0)),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--predictions", default="outputs/tables/11_v6_flow_predictions.csv")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--integration-steps", type=int, default=16)
    args = parser.parse_args()

    failures: list[str] = []
    if not Path(args.dataset).exists():
        failures.append(f"dataset missing: {args.dataset}")
    if not Path(args.model).exists():
        failures.append(f"model missing: {args.model}")
    if failures:
        metrics = {"pipeline_pass": False, "flow_claim_allowed": False, "eval_rows": 0, "methods": {}}
        write_json(args.metrics, metrics)
        write_status(args.status, STEP, "FAIL", [args.dataset, args.model], [args.metrics, args.status], metrics, failures, False)
        return 1

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    target = np.asarray(data["target"], dtype=np.float32)
    cond = np.asarray(data["cond"], dtype=np.float32)
    split = np.asarray(data.get("split", ["train"] * len(target)))
    aux = data.get("auxiliary", {})
    labels = aux.get("target_label_5", ["neutral"] * len(target))
    y_true = target_indices(labels)
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowRewardV2(cond_dim=cond.shape[1], target_dim=target.shape[1]).to(device)
    model.load_state_dict(state["model_state_dict"])
    pred_raw = integrate_flow(model, cond, device, steps=args.integration_steps)
    pred = project_rows_to_simplex(pred_raw)
    true_probs = pred[np.arange(len(pred)), y_true]
    judge_true_probs = np.asarray(aux.get("true_label_probability_ensemble", target[np.arange(len(target)), y_true]), dtype=float)

    df = pd.DataFrame(
        {
            "sample_id": data.get("sample_id", list(range(len(target)))),
            "candidate_id": data.get("candidate_id", [0] * len(target)),
            "split": split,
            "target_label_5": labels,
            "flow_reward_score": true_probs,
            "proxy_average_reward": proxy_average_score(aux),
            "single_best_judge_reward": judge_true_probs,
            "technical_rule_reward": np.asarray(aux.get("technical_rule_delta", np.zeros(len(target))), dtype=float),
            "raw_realized_utility": np.asarray(aux.get("raw_realized_utility", np.zeros(len(target))), dtype=float),
            "technical_rule_delta": np.asarray(aux.get("technical_rule_delta", np.zeros(len(target))), dtype=float),
            "news_grounding_score": np.asarray(aux.get("news_grounding_score", np.zeros(len(target))), dtype=float),
            "technical_grounding_score": np.asarray(aux.get("technical_grounding_score", np.zeros(len(target))), dtype=float),
            "unsupported_news_claim_rate": np.asarray(aux.get("unsupported_news_claim_rate", np.ones(len(target))), dtype=float),
            "hard_event_track": aux.get("hard_event_track", ["unknown"] * len(target)),
            "volatility_regime": aux.get("volatility_regime", ["unknown"] * len(target)),
        }
    )
    eval_df = df[df["split"].eq(args.split)].copy()
    if len(eval_df) < min(300, max(1, int(0.2 * len(df)))):
        failures.append(f"validation rows {len(eval_df)} below Step 11 acceptance")

    eval_idx = eval_df.index.to_numpy()
    flow_metrics = method_metrics(eval_df, "flow_reward_score")
    proxy_metrics = method_metrics(eval_df, "proxy_average_reward")
    single_metrics = method_metrics(eval_df, "single_best_judge_reward")
    technical_metrics = method_metrics(eval_df, "technical_rule_reward")

    flow_rank = flow_metrics["rank_correlation_with_raw_realized_utility"] or 0.0
    proxy_rank = proxy_metrics["rank_correlation_with_raw_realized_utility"] or 0.0
    flow_pair = flow_metrics["preference_pair_accuracy_by_raw_utility"]
    proxy_pair = proxy_metrics["preference_pair_accuracy_by_raw_utility"]
    flow_top = flow_metrics["top_decile_raw_realized_utility"]
    proxy_top = proxy_metrics["top_decile_raw_realized_utility"]
    wins = {
        "rank": flow_rank >= proxy_rank + 0.03,
        "pair": flow_pair is not None and proxy_pair is not None and flow_pair >= proxy_pair + 0.02,
        "top_decile": flow_top is not None and proxy_top is not None and flow_top >= proxy_top,
    }
    core_utility_win = metric_wins(wins)

    kl_to_judge = float(np.mean(kl_divergence_rows(target[eval_idx], pred[eval_idx]))) if len(eval_idx) else float("nan")
    js_to_judge = float(np.mean(js_divergence_rows(target[eval_idx], pred[eval_idx]))) if len(eval_idx) else float("nan")
    proxy_kl_to_judge = 0.0
    distributional_fidelity_not_worse_than_proxy = bool(kl_to_judge <= proxy_kl_to_judge + 0.02)
    flow_unsupported = flow_metrics["unsupported_claim_rate_top_decile"]
    proxy_unsupported = proxy_metrics["unsupported_claim_rate_top_decile"]
    unsupported_not_worse = bool(
        flow_unsupported is not None
        and proxy_unsupported is not None
        and flow_unsupported <= proxy_unsupported + 1e-9
    )
    missing_required_methods = ["flow_reward_without_reliability_weight", "flow_reward_without_grounding_auxiliary"]
    flow_claim_allowed = bool(
        core_utility_win
        and distributional_fidelity_not_worse_than_proxy
        and unsupported_not_worse
        and not missing_required_methods
    )

    methods = {
        "flow_reward_v6": {
            **flow_metrics,
            "kl_to_calibrated_judge_distribution": kl_to_judge,
            "js_to_calibrated_judge_distribution": js_to_judge,
            "brier_true_label_probability": brier_multiclass(pred[eval_idx], y_true[eval_idx]) if len(eval_idx) else None,
            "ece_true_label_probability": expected_calibration_error(true_probs[eval_idx], (np.argmax(pred[eval_idx], axis=1) == y_true[eval_idx]).astype(int)) if len(eval_idx) else None,
            "top_decile_target_probability": top_decile_utility(eval_df, "flow_reward_score", "single_best_judge_reward"),
        },
        "proxy_average_reward": proxy_metrics,
        "single_best_judge_reward": single_metrics,
        "technical_rule_reward": technical_metrics,
        "flow_reward_without_reliability_weight": {"status": "NOT_RUN", "reason": "ablation checkpoint not trained in this step"},
        "flow_reward_without_grounding_auxiliary": {"status": "NOT_RUN", "reason": "ablation checkpoint not trained in this step"},
    }
    metrics = {
        "pipeline_pass": not failures,
        "claim_allowed": flow_claim_allowed,
        "flow_claim_allowed": flow_claim_allowed,
        "flow_reward_improvement": core_utility_win,
        "rows": int(len(df)),
        "eval_rows": int(len(eval_df)),
        "eval_split": args.split,
        "metric_wins": wins,
        "core_utility_win": core_utility_win,
        "distributional_fidelity_not_worse_than_proxy": distributional_fidelity_not_worse_than_proxy,
        "top_decile_unsupported_claim_rate_not_worse_than_proxy": unsupported_not_worse,
        "missing_required_methods": missing_required_methods,
        "methods": methods,
        "metric_wins_by_track": group_wins(eval_df, "hard_event_track", "flow_reward_score", "proxy_average_reward"),
        "metric_wins_by_volatility_regime": group_wins(eval_df, "volatility_regime", "flow_reward_score", "proxy_average_reward"),
    }
    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.predictions, index=False)
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.dataset, args.model, args.metrics, args.predictions], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.dataset, args.model], [args.metrics, args.predictions, args.manifest, args.status], metrics, failures, status == "PASS")
    print(json.dumps({"status": status, "failures": failures, "flow_claim_allowed": flow_claim_allowed, "metric_wins": wins}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
