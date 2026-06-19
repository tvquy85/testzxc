from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_KEYS = {
    "step",
    "status",
    "inputs_checked",
    "outputs_created",
    "metrics",
    "failures",
    "next_step_allowed",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(path: str | os.PathLike[str]) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: str | os.PathLike[str]) -> int | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    suffix = p.suffix.lower()
    try:
        if suffix == ".jsonl":
            with open(p, "rb") as f:
                return sum(1 for line in f if line.strip())
        if suffix == ".csv":
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = sum(1 for _ in reader)
            return max(0, rows - 1)
        if suffix == ".json":
            return 1
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq

                return int(pq.ParquetFile(p).metadata.num_rows)
            except Exception:
                return None
    except Exception:
        return None
    return None


def artifact_entry(path: str | os.PathLike[str], producer_step: str, run_id: str | None = None) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p).replace("\\", "/"),
        "row_count": row_count(p),
        "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
        "timestamp": utc_now_iso(),
        "producer_step": producer_step,
        "run_id": run_id,
    }


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_manifest(
    path: str | os.PathLike[str],
    artifact_paths: list[str],
    producer_step: str,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "producer_step": producer_step,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "artifacts": [artifact_entry(p, producer_step, run_id) for p in artifact_paths],
    }
    if extra:
        manifest.update(extra)
    write_json(path, manifest)
    return manifest


def write_status(
    path: str | os.PathLike[str],
    step: str,
    status: str,
    inputs_checked: list[str] | None = None,
    outputs_created: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    failures: list[str] | None = None,
    next_step_allowed: bool | None = None,
) -> dict[str, Any]:
    status = status.upper()
    failures = failures or []
    if next_step_allowed is None:
        next_step_allowed = status == "PASS" and not failures
    payload = {
        "step": step,
        "status": status,
        "inputs_checked": inputs_checked or [],
        "outputs_created": outputs_created or [],
        "metrics": metrics or {},
        "failures": failures,
        "next_step_allowed": bool(next_step_allowed),
    }
    write_json(path, payload)
    return payload

