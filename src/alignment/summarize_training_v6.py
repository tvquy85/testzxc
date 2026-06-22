from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import sha256_file, write_json, write_manifest, write_status

STEP = "13_TRAIN_RWSFT_DPO_V6"
DEFAULT_RWSFT_ADAPTER = "outputs/models/qwen3_current_v6_rwsft_adapter/adapter_model.safetensors"
DEFAULT_DPO_ADAPTER = "outputs/models/qwen3_current_v6_dpo_adapter/adapter_model.safetensors"


def load_status(path: str) -> tuple[dict[str, Any], list[str]]:
    p = Path(path)
    if not p.exists():
        return {}, [f"missing status: {path}"]
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {}, [f"invalid status JSON {path}: {type(exc).__name__}: {str(exc)[:200]}"]
    return data, []


def first_existing_adapter(status: dict[str, Any], default_path: str) -> str:
    outputs = status.get("outputs_created", [])
    for raw_path in outputs:
        path = str(raw_path)
        if path.endswith("adapter_model.safetensors") and Path(path).exists():
            return path
    return default_path


def finite_loss_series(metrics: dict[str, Any], keys: list[str]) -> bool:
    for key in keys:
        values = metrics.get(key)
        if values is None:
            continue
        if not isinstance(values, list):
            return False
        for value in values:
            try:
                if not math.isfinite(float(value)):
                    return False
            except Exception:
                return False
    return True


def summarize_child(name: str, status: dict[str, Any], adapter_path: str, min_steps: int) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    metrics = status.get("metrics", {}) if isinstance(status.get("metrics"), dict) else {}
    max_steps = int(metrics.get("max_steps") or 0)
    adapter_exists = Path(adapter_path).exists()
    if status.get("status") != "PASS":
        failures.append(f"{name} status is not PASS")
    if not adapter_exists:
        failures.append(f"{name} adapter missing: {adapter_path}")
    if max_steps < min_steps:
        failures.append(f"{name} max_steps {max_steps} < {min_steps}")
    if not finite_loss_series(metrics, ["losses", "dpo_losses"]):
        failures.append(f"{name} losses are missing or non-finite")
    summary = {
        "status": status.get("status"),
        "max_steps": max_steps,
        "adapter_model": adapter_path,
        "adapter_exists": adapter_exists,
        "adapter_sha256": sha256_file(adapter_path) if adapter_exists else None,
        "loss_first": metrics.get("loss_first", metrics.get("dpo_loss_first")),
        "loss_last": metrics.get("loss_last", metrics.get("dpo_loss_last")),
        "train_records_loaded": metrics.get("train_records_loaded", metrics.get("dpo_train_records_loaded")),
        "max_memory_allocated_bytes": metrics.get("max_memory_allocated_bytes", metrics.get("dpo_max_memory_allocated_bytes")),
    }
    return summary, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwsft", required=True)
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    parser.add_argument("--rwsft-adapter", default=DEFAULT_RWSFT_ADAPTER)
    parser.add_argument("--dpo-adapter", default=DEFAULT_DPO_ADAPTER)
    parser.add_argument("--min-steps", type=int, default=800)
    args = parser.parse_args()

    failures: list[str] = []
    rwsft_status, rwsft_load_failures = load_status(args.rwsft)
    dpo_status, dpo_load_failures = load_status(args.dpo)
    failures.extend(rwsft_load_failures)
    failures.extend(dpo_load_failures)

    rwsft_adapter = first_existing_adapter(rwsft_status, args.rwsft_adapter) if rwsft_status else args.rwsft_adapter
    dpo_adapter = first_existing_adapter(dpo_status, args.dpo_adapter) if dpo_status else args.dpo_adapter
    rwsft_summary, rwsft_failures = summarize_child("RWSFT", rwsft_status, rwsft_adapter, args.min_steps)
    dpo_summary, dpo_failures = summarize_child("DPO", dpo_status, dpo_adapter, args.min_steps)
    failures.extend(rwsft_failures)
    failures.extend(dpo_failures)

    metrics = {
        "pipeline_pass": not failures,
        "claim_allowed": False,
        "min_steps": args.min_steps,
        "rwsft": rwsft_summary,
        "dpo": dpo_summary,
        "adapters_exist": bool(rwsft_summary.get("adapter_exists") and dpo_summary.get("adapter_exists")),
        "training_statuses": {"rwsft": args.rwsft, "dpo": args.dpo},
    }
    write_json(args.metrics, metrics)
    outputs = [args.metrics]
    for path in [rwsft_adapter, dpo_adapter]:
        if Path(path).exists():
            outputs.append(path)
    write_manifest(args.manifest, outputs, STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.rwsft, args.dpo],
        [*outputs, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
