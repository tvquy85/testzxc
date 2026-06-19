from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "21_FINAL_REPRO_PACKAGE_AND_AAAI_GATE"
REQUIRED_STATUS = [
    "01_REPO_AUDIT_AND_SAFE_BRANCH",
    "02_CONFIG_PATHS_AND_REPRO_ENV",
    "03_DATA_SCALE_AND_SPLIT_LOCK",
    "04_LEAKAGE_GUARDS_AND_UNIT_TESTS",
    "05_LABELS_ABNORMAL_RETURN_AND_BALANCE",
    "06_TECHNICAL_FEATURES_V2",
    "07_TECH_EVENT_TOKENS_V2",
    "08_STRICT_RATIONALE_SCHEMA_NO_AUTOFIX",
    "09_RATIONALE_GENERATION_SCALEUP_TRAIN_ONLY",
    "10_PROXY_JUDGES_MULTI_MODEL_DEBIASED",
    "11_CLAIM_LEVEL_GROUNDING_JUDGES",
    "12_FLOW_REWARD_MULTITARGET_V2",
    "13_FLOW_REWARD_EVAL_VS_PROXY",
    "14_RWSFT_DPO_DATASET_REBUILD",
    "15_ALIGNMENT_TRAINING_REPRODUCIBLE",
    "16_DAILY_PORTFOLIO_BACKTEST_V2",
    "17_COUNTERFACTUAL_DIRECTIONAL_EVAL_V2",
    "18_BASELINES_EXPANSION_AND_SEEDS",
    "19_ABLATION_AND_STATISTICAL_TESTS",
    "20_PAPER_TABLES_NO_DUMMY_GATES",
]
LOCAL_PATH_PATTERNS = [
    re.compile(r"(?i)e:[/\\]huggingface"),
    re.compile(r"(?i)c:[/\\]users"),
    re.compile(r"(?i)/mnt/e[/\\]huggingface"),
]
SCAN_ROOTS = ["configs", "src", "prompts"]
ALIGNMENT_FILES = [
    "data/alignment/rwsft_train_v2.jsonl",
    "data/alignment/dpo_pairs_train_v2.jsonl",
]


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def check_statuses(blockers: list[str]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for step in REQUIRED_STATUS:
        path = Path(f"outputs/status/{step}.status.json")
        if not path.exists():
            blockers.append(f"missing status: {step}")
            continue
        data = load_json(path)
        status = str(data.get("status"))
        statuses[step] = status
        if status != "PASS":
            blockers.append(f"failed status: {step}")
        if not data.get("next_step_allowed", False):
            blockers.append(f"next_step_allowed=false: {step}")
    return statuses


def check_final_tables(blockers: list[str], warnings: list[str]) -> None:
    final_root = Path("outputs/tables/final")
    manifest = final_root / "table_manifest.json"
    required = [
        "table_1_prediction_main.csv",
        "table_2_explanation_quality.csv",
        "table_3_daily_portfolio_backtest.csv",
        "table_4_counterfactual_directional.csv",
        "table_5_ablation.csv",
        "table_6_scale_and_compute.csv",
    ]
    if not manifest.exists():
        blockers.append("missing final table manifest")
    for name in required:
        path = final_root / name
        if not path.exists():
            blockers.append(f"missing final table: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if "dummy" in text or "placeholder" in text:
            blockers.append(f"dummy/placeholder text in final table: {path}")
        if path.stat().st_size == 0:
            blockers.append(f"empty final table: {path}")
    if manifest.exists():
        data = load_json(manifest)
        for table, info in data.items():
            if not info.get("sources"):
                blockers.append(f"final table has no source mapping: {table}")
            if int(info.get("row_count", 0)) <= 0:
                blockers.append(f"final table has zero rows: {table}")
            if table == "table_5_ablation.csv" and int(info.get("evidence_row_count", 0)) == 0:
                warnings.append("ablation table contains NOT_RUN registry only; not used as evidence")


def check_local_paths(blockers: list[str]) -> None:
    for root in SCAN_ROOTS:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".pyc", ".safetensors", ".pt", ".parquet", ".png", ".jpg", ".jpeg", ".zip"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in LOCAL_PATH_PATTERNS:
                if pattern.search(text):
                    blockers.append(f"hard-coded local path in {path}: {pattern.pattern}")
                    break


def check_alignment_train_only(blockers: list[str]) -> None:
    import pandas as pd

    for file_name in ALIGNMENT_FILES:
        path = Path(file_name)
        if not path.exists():
            blockers.append(f"alignment dataset missing: {path}")
            continue
        try:
            df = pd.read_json(path, lines=True)
        except Exception as exc:
            blockers.append(f"could not read alignment dataset {path}: {type(exc).__name__}: {exc}")
            continue
        if df.empty:
            blockers.append(f"alignment dataset empty: {path}")
            continue
        if "split" not in df.columns:
            blockers.append(f"alignment dataset missing split column: {path}")
            continue
        bad = sorted(set(df["split"].dropna().astype(str)) - {"train"})
        if bad:
            blockers.append(f"test/val split appears in alignment dataset {path}: {bad}")


def check_backtest_daily(blockers: list[str]) -> None:
    status_path = Path("outputs/status/16_DAILY_PORTFOLIO_BACKTEST_V2.status.json")
    daily_path = Path("outputs/tables/backtest_daily_returns_v2.csv")
    if not status_path.exists() or not daily_path.exists():
        blockers.append("daily portfolio backtest status or returns file missing")
        return
    status = load_json(status_path)
    metrics = status.get("metrics", {})
    if int(metrics.get("num_trading_days", 0)) < 20:
        blockers.append("daily portfolio backtest has fewer than 20 trading days")
    if float(metrics.get("nonzero_daily_return_rate", 0.0)) <= 0:
        blockers.append("daily portfolio backtest has zero daily returns")


def check_flow_eval(blockers: list[str]) -> None:
    path = Path("outputs/metrics/flow_vs_proxy_eval_stage1.json")
    if not path.exists():
        blockers.append("flow v2 evaluation missing: outputs/metrics/flow_vs_proxy_eval_stage1.json")
        return
    data = load_json(path)
    methods = data.get("methods", [])
    if not methods:
        blockers.append("flow v2 evaluation has no method rows")
        return
    pending = [m.get("method", "unknown") for m in methods if str(m.get("status")) in {"PENDING_REAL_EVAL", "NOT_RUN"}]
    if pending:
        blockers.append(f"flow v2 evaluation has pending/non-run methods: {pending}")


def check_counterfactual(blockers: list[str]) -> None:
    path = Path("outputs/metrics/counterfactual_directional_v2.json")
    if not path.exists():
        blockers.append("counterfactual directional eval missing")
        return
    data = load_json(path)
    if int(data.get("num_tasks", 0)) <= 0:
        blockers.append("counterfactual directional eval has zero tasks")
    if float(data.get("schema_ok_rate", 0.0)) < 0.8:
        blockers.append("counterfactual directional eval schema_ok_rate below 0.8")


def check_baselines(blockers: list[str]) -> None:
    path = Path("outputs/metrics/baseline_suite_summary.json")
    if not path.exists():
        blockers.append("baseline suite summary missing")
        return
    data = load_json(path)
    required = data.get("required_minimum") or []
    if len(required) < 3:
        blockers.append("less than 3 core baselines configured")
    if not data.get("required_minimum_pass", False):
        blockers.append("less than 3 core baselines run successfully")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/repro/aaai_gate_report.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    blockers: list[str] = []
    warnings: list[str] = []
    statuses = check_statuses(blockers)
    check_final_tables(blockers, warnings)
    check_local_paths(blockers)
    check_alignment_train_only(blockers)
    check_backtest_daily(blockers)
    check_flow_eval(blockers)
    check_counterfactual(blockers)
    check_baselines(blockers)
    report = {
        "decision": "GO" if not blockers else "NO_GO",
        "blocking_issues": blockers,
        "warnings": warnings,
        "statuses": statuses,
    }
    write_json(args.output, report)
    write_manifest(args.manifest, [args.output], STEP)
    status = "PASS" if not blockers else "FAIL"
    write_status(args.status, STEP, status, ["outputs/status", "outputs/tables/final", "configs", "src", "prompts", *ALIGNMENT_FILES], [args.output, args.manifest, args.status], {"blocker_count": len(blockers), "warning_count": len(warnings), "decision": report["decision"]}, blockers, status == "PASS")
    print(json.dumps(report, indent=2))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
