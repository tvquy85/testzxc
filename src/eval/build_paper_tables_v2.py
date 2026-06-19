from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import artifact_entry, write_json, write_manifest, write_status


STEP = "20_PAPER_TABLES_NO_DUMMY_GATES"
TABLES = [
    "table_1_prediction_main.csv",
    "table_2_explanation_quality.csv",
    "table_3_daily_portfolio_backtest.csv",
    "table_4_counterfactual_directional.csv",
    "table_5_ablation.csv",
    "table_6_scale_and_compute.csv",
]


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON source is not an object: {path}")
    return data


def finite_or_text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_meta(path: Path) -> dict[str, Any]:
    entry = artifact_entry(path, STEP)
    return {
        "source_file": entry["path"],
        "source_sha256": entry["sha256"],
        "source_row_count": entry["row_count"],
        "timestamp": entry["timestamp"],
    }


def metric_row(
    source: Path,
    section: str,
    method: str,
    metric: str,
    value: Any,
    *,
    run_id: str,
    seed: str | int = "all",
    split: str = "unspecified",
    comparator: str = "none",
    status: str = "RUN",
    evidence_allowed: bool = True,
) -> dict[str, Any]:
    return {
        "section": section,
        "method": method,
        "comparator": comparator,
        "metric": metric,
        "value": finite_or_text(value),
        "run_id": run_id,
        "seed": seed,
        "split": split,
        "status": status,
        "evidence_allowed": bool(evidence_allowed),
        **source_meta(source),
    }


def require_file(path: Path, failures: list[str]) -> bool:
    if not path.exists():
        failures.append(f"missing required source: {path}")
        return False
    if path.is_file() and path.stat().st_size == 0:
        failures.append(f"empty required source: {path}")
        return False
    return True


def require_nonempty_table(path: Path, failures: list[str]) -> Any | None:
    import pandas as pd

    if not require_file(path, failures):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        failures.append(f"failed to read table source {path}: {type(exc).__name__}: {exc}")
        return None
    if df.empty:
        failures.append(f"required table source has zero rows: {path}")
        return None
    return df


def require_json(path: Path, failures: list[str]) -> dict[str, Any] | None:
    if not require_file(path, failures):
        return None
    try:
        data = read_json(path)
    except Exception as exc:
        failures.append(f"failed to read JSON source {path}: {type(exc).__name__}: {exc}")
        return None
    if not data:
        failures.append(f"required JSON source is empty: {path}")
        return None
    return data


def build_table_1(root: Path, failures: list[str]) -> list[dict[str, Any]]:
    baseline_path = root / "baseline_suite_aggregate.csv"
    prediction_metrics_path = Path("outputs/metrics/test_predictions_qwen3_dpo_flash_medium.json")
    bootstrap_path = root / "prediction_bootstrap_comparisons.csv"
    rows: list[dict[str, Any]] = []
    baseline = require_nonempty_table(baseline_path, failures)
    if baseline is not None:
        required = {"baseline", "seeds", "accuracy_mean", "macro_f1_mean", "mcc_mean", "brier_mean"}
        missing = required - set(baseline.columns)
        if missing:
            failures.append(f"{baseline_path} missing columns: {sorted(missing)}")
        else:
            for _, item in baseline.iterrows():
                for metric in ["accuracy_mean", "macro_f1_mean", "macro_f1_std", "mcc_mean", "mcc_std", "brier_mean", "train_rows_min", "test_rows_min", "feature_dim"]:
                    if metric in baseline.columns:
                        rows.append(
                            metric_row(
                                baseline_path,
                                "baseline_aggregate",
                                str(item["baseline"]),
                                metric,
                                item[metric],
                                run_id="step18_baseline_suite",
                                seed=f"{int(item['seeds'])}_seeds",
                                split="test",
                            )
                        )
    prediction_metrics = require_json(prediction_metrics_path, failures)
    if prediction_metrics is not None:
        for metric in ["rows", "selected_trading_days", "parse_ok_rate", "schema_ok_rate", "fallback_used"]:
            if metric not in prediction_metrics:
                failures.append(f"{prediction_metrics_path} missing metric: {metric}")
            else:
                rows.append(metric_row(prediction_metrics_path, "firefin_prediction_medium", "FIRE_Fin_Qwen3_DPO", metric, prediction_metrics[metric], run_id=prediction_metrics.get("run_id", "test_prediction_smoke_primary_5000"), split="test"))
        for action, count in prediction_metrics.get("action_distribution", {}).items():
            rows.append(metric_row(prediction_metrics_path, "firefin_prediction_medium", "FIRE_Fin_Qwen3_DPO", f"action_count_{action}", count, run_id=prediction_metrics.get("run_id", "test_prediction_smoke_primary_5000"), split="test"))
    bootstrap = require_nonempty_table(bootstrap_path, failures)
    if bootstrap is not None:
        required = {"baseline", "seed", "n_paired_rows", "delta_accuracy", "delta_macro_f1", "delta_mcc"}
        missing = required - set(bootstrap.columns)
        if missing:
            failures.append(f"{bootstrap_path} missing columns: {sorted(missing)}")
        else:
            metrics = [col for col in bootstrap.columns if col not in {"baseline", "seed"}]
            for _, item in bootstrap.iterrows():
                for metric in metrics:
                    rows.append(
                        metric_row(
                            bootstrap_path,
                            "paired_prediction_bootstrap",
                            "FIRE_Fin_Qwen3_DPO",
                            metric,
                            item[metric],
                            comparator=str(item["baseline"]),
                            run_id="step19_prediction_bootstrap",
                            seed=int(item["seed"]),
                            split="test",
                        )
                    )
    return rows


def build_table_2(metrics_root: Path, failures: list[str]) -> list[dict[str, Any]]:
    sources = {
        "inferability": metrics_root / "inferability_judge_stability_stage1.json",
        "grounding": metrics_root / "claim_grounding_summary_stage1.json",
        "flow_train": metrics_root / "flow_reward_v2_train_metrics_stage1.json",
        "flow_eval": metrics_root / "flow_vs_proxy_eval_stage1.json",
    }
    rows: list[dict[str, Any]] = []
    inferability = require_json(sources["inferability"], failures)
    if inferability is not None:
        for metric in ["rows", "parse_ok_rate", "label_order_consistency", "entropy_mean", "mean_probability_true_label"]:
            if metric in inferability:
                rows.append(metric_row(sources["inferability"], "inferability_judge", "multi_model_judge", metric, inferability[metric], run_id="stage1", split="train"))
            else:
                failures.append(f"{sources['inferability']} missing metric: {metric}")
        for judge, info in inferability.get("judge_inventory", {}).items():
            rows.append(metric_row(sources["inferability"], "judge_inventory", judge, "status", info.get("status"), run_id="stage1", split="train"))
    grounding = require_json(sources["grounding"], failures)
    if grounding is not None:
        rows.append(metric_row(sources["grounding"], "claim_grounding", "claim_level_grounding", "rows", grounding.get("rows"), run_id="stage1", split="train"))
        status_counts = grounding.get("status_counts", {})
        total = sum(int(v) for v in status_counts.values()) if status_counts else 0
        for status, count in status_counts.items():
            rows.append(metric_row(sources["grounding"], "claim_grounding", "claim_level_grounding", f"status_count_{status}", count, run_id="stage1", split="train"))
            if total:
                rows.append(metric_row(sources["grounding"], "claim_grounding", "claim_level_grounding", f"status_rate_{status}", count / total, run_id="stage1", split="train"))
        for claim_type, count in grounding.get("claim_type_counts", {}).items():
            rows.append(metric_row(sources["grounding"], "claim_grounding", "claim_level_grounding", f"claim_type_count_{claim_type}", count, run_id="stage1", split="train"))
    flow_train = require_json(sources["flow_train"], failures)
    if flow_train is not None:
        rows.append(metric_row(sources["flow_train"], "flow_reward_v2", "flow_reward_v2_train", "rows", flow_train.get("rows"), run_id="stage1", split="train"))
        rows.append(metric_row(sources["flow_train"], "flow_reward_v2", "flow_reward_v2_train", "target_dim", flow_train.get("target_dim"), run_id="stage1", split="train"))
        for idx, loss in enumerate(flow_train.get("train_loss_by_epoch", []), start=1):
            rows.append(metric_row(sources["flow_train"], "flow_reward_v2", "flow_reward_v2_train", f"train_loss_epoch_{idx}", loss, run_id="stage1", split="train"))
    flow_eval = require_json(sources["flow_eval"], failures)
    if flow_eval is not None:
        rows.append(metric_row(sources["flow_eval"], "flow_reward_v2_eval", "flow_vs_proxy", "rows", flow_eval.get("rows"), run_id="stage1", split=flow_eval.get("split", "val")))
        rows.append(metric_row(sources["flow_eval"], "flow_reward_v2_eval", "flow_vs_proxy", "claim_improvement", flow_eval.get("claim_improvement"), run_id="stage1", split=flow_eval.get("split", "val")))
        for method in flow_eval.get("methods", []):
            rows.append(metric_row(sources["flow_eval"], "flow_reward_v2_eval", method.get("method", "unknown"), "status", method.get("status"), run_id="stage1", split=flow_eval.get("split", "val"), evidence_allowed=method.get("status") != "PENDING_REAL_EVAL"))
    return rows


def build_table_3(metrics_root: Path, tables_root: Path, failures: list[str]) -> list[dict[str, Any]]:
    backtest_path = metrics_root / "backtest_daily_portfolio_v2.json"
    bootstrap_path = tables_root / "backtest_block_bootstrap_comparisons.csv"
    rows: list[dict[str, Any]] = []
    backtest = require_json(backtest_path, failures)
    if backtest is not None:
        for metric, value in backtest.items():
            rows.append(metric_row(backtest_path, "daily_portfolio", "FIRE_Fin_Qwen3_DPO", metric, value, run_id="step16_medium", split="test"))
    bootstrap = require_nonempty_table(bootstrap_path, failures)
    if bootstrap is not None:
        required = {"baseline", "seed", "n_paired_days", "delta_mean_daily_return", "delta_sharpe"}
        missing = required - set(bootstrap.columns)
        if missing:
            failures.append(f"{bootstrap_path} missing columns: {sorted(missing)}")
        else:
            for _, item in bootstrap.iterrows():
                for metric in [col for col in bootstrap.columns if col not in {"baseline", "seed"}]:
                    rows.append(metric_row(bootstrap_path, "daily_portfolio_block_bootstrap", "FIRE_Fin_Qwen3_DPO", metric, item[metric], comparator=str(item["baseline"]), run_id="step19_backtest_block_bootstrap", seed=int(item["seed"]), split="test"))
    return rows


def build_table_4(metrics_root: Path, failures: list[str]) -> list[dict[str, Any]]:
    source = metrics_root / "counterfactual_directional_v2.json"
    data = require_json(source, failures)
    rows: list[dict[str, Any]] = []
    if data is not None:
        for metric, value in data.items():
            rows.append(metric_row(source, "counterfactual_directional", "FIRE_Fin_Qwen3_DPO", metric, value, run_id="step17_medium", split="test"))
    return rows


def build_table_5(tables_root: Path, failures: list[str]) -> list[dict[str, Any]]:
    source = tables_root / "ablation_suite_results.csv"
    df = require_nonempty_table(source, failures)
    rows: list[dict[str, Any]] = []
    if df is not None:
        required = {"ablation", "status"}
        missing = required - set(df.columns)
        if missing:
            failures.append(f"{source} missing columns: {sorted(missing)}")
        else:
            for _, item in df.iterrows():
                status = str(item["status"])
                evidence_allowed = status == "PASS"
                rows.append(metric_row(source, "ablation_registry", str(item["ablation"]), "status", status, run_id="step19_ablation_registry", split="test", status=status, evidence_allowed=evidence_allowed))
                for metric in [col for col in df.columns if col not in {"ablation", "status"}]:
                    value = item[metric]
                    if status == "PASS" and finite_or_text(value) is None:
                        failures.append(f"ablation {item['ablation']} is PASS but missing metric {metric}")
                    if finite_or_text(value) is not None:
                        rows.append(metric_row(source, "ablation_registry", str(item["ablation"]), metric, value, run_id="step19_ablation_registry", split="test", status=status, evidence_allowed=evidence_allowed))
    return rows


def build_table_6(metrics_root: Path, failures: list[str]) -> list[dict[str, Any]]:
    dataset_source = Path("outputs/manifests/fnspid_dataset_manifest.json")
    env_source = Path("outputs/audit/repro_env_report.json")
    rows: list[dict[str, Any]] = []
    dataset = require_json(dataset_source, failures)
    if dataset is not None:
        for metric, value in dataset.get("metrics", {}).items():
            rows.append(metric_row(dataset_source, "data_scale", "fnspid_locked_split", metric, value, run_id="step03", split="all"))
        processed = dataset.get("processed_cache_scan", {})
        for metric in ["news_thin_rows", "price_ticker_files", "price_rows_from_metadata"]:
            if metric in processed:
                rows.append(metric_row(dataset_source, "data_scale", "fnspid_processed_cache", metric, processed[metric], run_id="step03", split="all"))
    env = require_json(env_source, failures)
    if env is not None:
        rows.append(metric_row(env_source, "environment", "python", "python_version", env.get("python_version"), run_id="step02", split="all"))
        for module, info in env.get("module_inventory", {}).items():
            rows.append(metric_row(env_source, "environment_module_inventory", module, "available", info.get("available"), run_id="step02", split="all"))
            if info.get("version") is not None:
                rows.append(metric_row(env_source, "environment_module_inventory", module, "version", info.get("version"), run_id="step02", split="all"))
        gpu_info = env.get("gpu", {})
        if isinstance(gpu_info, dict):
            for metric, value in gpu_info.items():
                rows.append(metric_row(env_source, "environment_gpu", "gpu", metric, value, run_id="step02", split="all"))
    return rows


def validate_rows(table: str, rows: list[dict[str, Any]], failures: list[str]) -> None:
    if not rows:
        failures.append(f"{table} has no metric rows")
        return
    for idx, row in enumerate(rows):
        for required in ["source_file", "run_id", "seed", "split", "timestamp", "metric", "value"]:
            if required not in row:
                failures.append(f"{table} row {idx} missing required column {required}")
        if row.get("value") is None and row.get("status") == "RUN":
            failures.append(f"{table} row {idx} has missing value for metric {row.get('metric')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", default="outputs/metrics")
    parser.add_argument("--tables-root", default="outputs/tables")
    parser.add_argument("--output-root", default="outputs/tables/final")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    metrics_root = Path(args.metrics_root)
    tables_root = Path(args.tables_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    builders = {
        "table_1_prediction_main.csv": lambda: build_table_1(tables_root, failures),
        "table_2_explanation_quality.csv": lambda: build_table_2(metrics_root, failures),
        "table_3_daily_portfolio_backtest.csv": lambda: build_table_3(metrics_root, tables_root, failures),
        "table_4_counterfactual_directional.csv": lambda: build_table_4(metrics_root, failures),
        "table_5_ablation.csv": lambda: build_table_5(tables_root, failures),
        "table_6_scale_and_compute.csv": lambda: build_table_6(metrics_root, failures),
    }
    outputs: list[str] = []
    table_manifest: dict[str, Any] = {}
    table_row_counts: dict[str, int] = {}
    evidence_row_counts: dict[str, int] = {}
    for table in TABLES:
        rows = builders[table]()
        validate_rows(table, rows, failures)
        out_path = output_root / table
        pd.DataFrame(rows).to_csv(out_path, index=False)
        outputs.append(str(out_path))
        table_row_counts[table] = len(rows)
        evidence_row_counts[table] = sum(1 for row in rows if row.get("evidence_allowed"))
        sources = sorted({row["source_file"] for row in rows if row.get("source_file")})
        table_manifest[table] = {
            "output": artifact_entry(out_path, STEP),
            "sources": [artifact_entry(source, STEP) for source in sources],
            "row_count": len(rows),
            "evidence_row_count": evidence_row_counts[table],
        }
    manifest_path = output_root / "table_manifest.json"
    write_json(manifest_path, table_manifest)
    outputs.append(str(manifest_path))
    write_manifest(args.manifest, outputs, STEP)
    metrics = {
        "tables_created": len(TABLES),
        "table_row_counts": table_row_counts,
        "evidence_row_counts": evidence_row_counts,
    }
    status = "PASS" if not failures else "FAIL"
    write_status(args.status, STEP, status, [args.metrics_root, args.tables_root], outputs + [args.manifest, args.status], metrics, failures, status == "PASS")
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
