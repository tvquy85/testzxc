from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reward.flow_model_v2 import FlowRewardV2
from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "13_FLOW_REWARD_EVAL_VS_PROXY"
LABEL_TO_DISPLAY = {
    "strong_down": "Strong Down",
    "mild_down": "Mild Down",
    "neutral": "Neutral",
    "mild_up": "Mild Up",
    "strong_up": "Strong Up",
}
OBSERVED_TARGETS = [
    "inferability_true_label_prob",
    "multi_judge_agreement",
    "news_grounding_score",
    "technical_grounding_score",
]


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            data = json.loads(value)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def true_label_probability(parsed_json: Any, true_label: str) -> float | None:
    parsed = parse_jsonish(parsed_json)
    dist = parsed.get("forecast_distribution", {})
    if not isinstance(dist, dict):
        return None
    key = LABEL_TO_DISPLAY.get(str(true_label))
    if key is None:
        return None
    try:
        value = float(dist.get(key))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def masked_mean(values: Any, masks: Any, indices: list[int]) -> Any:
    import numpy as np

    selected = values[:, indices]
    selected_mask = masks[:, indices] > 0
    denom = selected_mask.sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    valid = denom > 0
    out[valid] = (selected[valid] * selected_mask[valid]).sum(axis=1) / denom[valid]
    return out


def weighted_mean(values: Any, masks: Any, weights: dict[int, float]) -> Any:
    import numpy as np

    out = np.full(values.shape[0], np.nan, dtype=float)
    numer = np.zeros(values.shape[0], dtype=float)
    denom = np.zeros(values.shape[0], dtype=float)
    for idx, weight in weights.items():
        valid = masks[:, idx] > 0
        numer[valid] += values[valid, idx] * weight
        denom[valid] += weight
    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    return out


def flow_predict_scores(checkpoint: str, cond: Any, target_dim: int, target_names: list[str], integration_steps: int, batch_size: int) -> Any:
    import numpy as np
    import torch

    ckpt_path = Path(checkpoint) / "model.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    cond_dim = int(config.get("cond_dim", cond.shape[1]))
    model = FlowRewardV2(cond_dim=cond_dim, target_dim=int(config.get("target_dim", target_dim)))
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, cond.shape[0], batch_size):
            c = torch.tensor(cond[start : start + batch_size], dtype=torch.float32)
            z = torch.zeros((c.shape[0], model.target_dim), dtype=torch.float32)
            steps = max(1, integration_steps)
            for idx in range(steps):
                t = torch.full((c.shape[0], 1), float(idx) / float(steps), dtype=torch.float32)
                z = z + model(t, z, c) / float(steps)
            preds.append(z.cpu().numpy())
    pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, target_dim), dtype=float)
    observed = [target_names.index(name) for name in OBSERVED_TARGETS if name in target_names]
    return np.nanmean(np.clip(pred[:, observed], 0.0, 1.0), axis=1) if observed else np.full(pred.shape[0], np.nan)


def spearman(x: Any, y: Any) -> float | None:
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 2 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return None
    return float(df["x"].rank().corr(df["y"].rank()))


def preference_pair_accuracy(frame: Any, score_col: str, utility_col: str) -> float | None:
    correct = 0
    total = 0
    for _, group in frame.dropna(subset=[score_col, utility_col]).groupby("sample_id"):
        if len(group) < 2:
            continue
        records = group[[score_col, utility_col]].to_numpy(dtype=float)
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                utility_delta = records[i, 1] - records[j, 1]
                if abs(utility_delta) < 1e-9:
                    continue
                score_delta = records[i, 0] - records[j, 0]
                if abs(score_delta) < 1e-9:
                    continue
                total += 1
                correct += int((utility_delta > 0) == (score_delta > 0))
    return float(correct / total) if total else None


def evaluate_method(frame: Any, method: str, score_col: str) -> dict[str, Any]:
    import numpy as np

    df = frame.dropna(subset=[score_col, "realized_utility"]).copy()
    if df.empty:
        return {
            "method": method,
            "rank_correlation_with_realized_utility": None,
            "calibration_error": None,
            "top_k_rationale_quality": None,
            "preference_pair_accuracy": None,
            "variance_by_regime": None,
            "status": "NOT_RUN",
            "eval_rows": 0,
        }
    k = max(1, int(math.ceil(0.10 * len(df))))
    top = df.sort_values(score_col, ascending=False).head(k)
    regime_means = df.groupby("regime", dropna=False)[score_col].mean()
    return {
        "method": method,
        "rank_correlation_with_realized_utility": spearman(df[score_col], df["realized_utility"]),
        "calibration_error": float(np.abs(np.clip(df[score_col].astype(float), 0.0, 1.0) - df["realized_utility"].astype(float)).mean()),
        "top_k_rationale_quality": float(top["realized_utility"].astype(float).mean()),
        "preference_pair_accuracy": preference_pair_accuracy(df, score_col, "realized_utility"),
        "variance_by_regime": float(regime_means.var(ddof=0)) if len(regime_means) > 1 else 0.0,
        "status": "RUN",
        "eval_rows": int(len(df)),
    }


def improvement_count(flow: dict[str, Any], proxy: dict[str, Any]) -> int:
    count = 0
    for metric in ["rank_correlation_with_realized_utility", "top_k_rationale_quality", "preference_pair_accuracy"]:
        if flow.get(metric) is not None and proxy.get(metric) is not None and flow[metric] > proxy[metric]:
            count += 1
    if flow.get("calibration_error") is not None and proxy.get("calibration_error") is not None and flow["calibration_error"] < proxy["calibration_error"]:
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/reward/flow_v2_train_dataset.pt")
    parser.add_argument("--checkpoint", default="checkpoints/flow_reward_v2")
    parser.add_argument("--rationales", default="data/rationales/parsed/train_candidates_stage1_strict.parquet")
    parser.add_argument("--labels", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--split", default="val")
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--min-eval-rows", type=int, default=500)
    parser.add_argument("--integration-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-json", default="outputs/metrics/flow_vs_proxy_eval.json")
    parser.add_argument("--output-csv", default="outputs/tables/flow_vs_proxy_eval.csv")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch

    failures: list[str] = []
    if args.split != "val":
        failures.append("flow-vs-proxy evaluation must use validation split only")
    if not Path(args.dataset).exists():
        failures.append(f"dataset missing: {args.dataset}")
    if not (Path(args.checkpoint) / "model.pt").exists():
        failures.append(f"checkpoint missing: {Path(args.checkpoint) / 'model.pt'}")
    if not Path(args.rationales).exists():
        failures.append(f"rationales missing: {args.rationales}")
    if not Path(args.labels).exists():
        failures.append(f"labels missing: {args.labels}")
    table = pd.DataFrame()
    rows = 0
    eval_rows = 0
    method_rows: list[dict[str, Any]] = []
    if not failures:
        data = torch.load(args.dataset, map_location="cpu", weights_only=False)
        target = np.asarray(data["target"], dtype=np.float32)
        mask = np.asarray(data["mask"], dtype=np.float32)
        cond = np.asarray(data["cond"], dtype=np.float32)
        target_names = list(data["target_names"])
        rows = int(target.shape[0])
        if rows == 0:
            failures.append("dataset has zero rows")
        rationales = pd.read_parquet(args.rationales)
        labels = pd.read_parquet(args.labels)[["sample_id", "label_5", "split"]].rename(columns={"split": "label_split"})
        frame = pd.DataFrame({"sample_id": data["sample_id"], "candidate_id": data["candidate_id"]})
        for idx, name in enumerate(target_names):
            frame[name] = target[:, idx]
            frame[f"{name}_mask"] = mask[:, idx]
        frame = frame.merge(rationales[["sample_id", "candidate_id", "parsed_json", "schema_ok"]], on=["sample_id", "candidate_id"], how="left")
        frame = frame.merge(labels, on="sample_id", how="left")
        frame = frame[(frame["label_split"] == "train") & (frame["schema_ok"].astype(bool))].copy()
        frame["realized_utility"] = [true_label_probability(parsed, label) for parsed, label in zip(frame["parsed_json"], frame["label_5"])]
        hash_values = pd.util.hash_pandas_object(frame[["sample_id", "candidate_id"]].astype(str), index=False).to_numpy(dtype=np.uint64)
        threshold = int((1.0 - args.holdout_frac) * 10_000)
        holdout_mask = (hash_values % 10_000) >= threshold
        frame = frame[holdout_mask].copy()
        eval_indices = frame.index.to_numpy()
        eval_rows = int(len(frame))
        if eval_rows < args.min_eval_rows:
            failures.append(f"eval rows {eval_rows} < required {args.min_eval_rows}")
        observed_indices = [target_names.index(name) for name in OBSERVED_TARGETS if name in target_names]
        frame["proxy_average_reward"] = masked_mean(target[eval_indices], mask[eval_indices], observed_indices)
        frame["single_best_judge_reward"] = target[eval_indices, target_names.index("inferability_true_label_prob")]
        frame.loc[mask[eval_indices, target_names.index("inferability_true_label_prob")] <= 0, "single_best_judge_reward"] = np.nan
        frame["flow_reward_v1"] = weighted_mean(
            target[eval_indices],
            mask[eval_indices],
            {
                target_names.index("inferability_true_label_prob"): 0.5,
                target_names.index("news_grounding_score"): 0.25,
                target_names.index("technical_grounding_score"): 0.25,
            },
        )
        frame["flow_reward_v2"] = flow_predict_scores(args.checkpoint, cond[eval_indices], target.shape[1], target_names, args.integration_steps, args.batch_size)
        regime_index = np.argmax(cond[eval_indices, -3:], axis=1) if cond.shape[1] >= 3 else np.ones(eval_rows, dtype=int)
        regime_map = {0: "low_vol", 1: "normal_vol", 2: "high_vol"}
        frame["regime"] = [regime_map.get(int(idx), "unknown") for idx in regime_index]
        for method, score_col in [
            ("proxy_average_reward", "proxy_average_reward"),
            ("single_best_judge_reward", "single_best_judge_reward"),
            ("flow_reward_v1", "flow_reward_v1"),
            ("flow_reward_v2", "flow_reward_v2"),
        ]:
            method_rows.append(evaluate_method(frame, method, score_col))
        for method in method_rows:
            if method["status"] != "RUN":
                failures.append(f"method did not run: {method['method']}")
            for metric in ["rank_correlation_with_realized_utility", "calibration_error", "top_k_rationale_quality", "preference_pair_accuracy", "variance_by_regime"]:
                if method.get(metric) is None:
                    failures.append(f"method {method['method']} missing metric {metric}")
        table = pd.DataFrame(method_rows)
    claim_improvement = False
    if method_rows:
        by_method = {row["method"]: row for row in method_rows}
        if "flow_reward_v2" in by_method and "proxy_average_reward" in by_method:
            claim_improvement = improvement_count(by_method["flow_reward_v2"], by_method["proxy_average_reward"]) >= 2
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    report = {
        "split": args.split,
        "split_source": "deterministic_train_holdout_no_test_rows",
        "rows": rows,
        "eval_rows": eval_rows,
        "holdout_frac": args.holdout_frac,
        "integration_steps": args.integration_steps,
        "methods": method_rows,
        "claim_improvement": bool(claim_improvement),
    }
    write_json(args.output_json, report)
    write_manifest(args.manifest, [args.dataset, str(Path(args.checkpoint) / "model.pt"), args.rationales, args.labels, args.output_json, args.output_csv], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.dataset, args.checkpoint, args.rationales, args.labels],
        outputs_created=[args.output_json, args.output_csv, args.manifest, args.status],
        metrics={"rows": rows, "eval_rows": eval_rows, "method_count": len(method_rows), "claim_improvement": bool(claim_improvement)},
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
