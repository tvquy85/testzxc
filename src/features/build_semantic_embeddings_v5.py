from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataclean_v4_utils import clean_string
from src.utils.artifacts import write_json, write_manifest, write_status

STEP = "10_FLOW_SEMANTIC_EMBEDDINGS_V5"


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def rationale_text(row: pd.Series) -> str:
    parsed = parse_json(row.get("parsed_json"))
    chunks: list[str] = []
    for item in parsed.get("news_rationale", []) if isinstance(parsed.get("news_rationale"), list) else []:
        if isinstance(item, dict):
            chunks.append(" ".join(clean_string(item.get(k)) for k in ["factor", "direction", "strength"]))
    for item in parsed.get("technical_rationale", []) if isinstance(parsed.get("technical_rationale"), list) else []:
        if isinstance(item, dict):
            chunks.append(" ".join(clean_string(item.get(k)) for k in ["signal", "direction", "strength"]))
    chunks.append(clean_string(parsed.get("conflict_resolution")))
    chunks.append(clean_string(parsed.get("risk_note")))
    if not any(chunks):
        chunks.append(clean_string(row.get("raw_text", row.get("raw_output"))))
    return " ".join(chunk for chunk in chunks if chunk)


def build_texts(rationales: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    context_cols = [col for col in ["sample_id", "split", "target_label_5", "track", "clean_context_text", "evidence_pack_json"] if col in contexts.columns]
    df = rationales.merge(contexts[context_cols], on="sample_id", how="inner", suffixes=("", "_context"))
    df["embedding_text"] = [
        f"Context: {clean_string(row.get('clean_context_text'))}\nRationale: {rationale_text(row)}"
        for _, row in df.iterrows()
    ]
    return df


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text)]


def tfidf_svd_embeddings(texts: list[str], dim: int = 128, max_vocab: int = 1200) -> np.ndarray:
    docs = [tokenize(text) for text in texts]
    vocab_counter = Counter(token for doc in docs for token in set(doc))
    vocab = [tok for tok, _ in vocab_counter.most_common(max_vocab)]
    if not vocab:
        return np.zeros((len(texts), dim), dtype=np.float32)
    vocab_index = {tok: idx for idx, tok in enumerate(vocab)}
    mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row_idx, doc in enumerate(docs):
        counts = Counter(tok for tok in doc if tok in vocab_index)
        total = float(sum(counts.values())) or 1.0
        for tok, count in counts.items():
            mat[row_idx, vocab_index[tok]] = float(count) / total
    n_docs = max(1, len(docs))
    df = np.maximum(1.0, (mat > 0).sum(axis=0))
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    mat *= idf.astype(np.float32)
    mat -= mat.mean(axis=0, keepdims=True)
    try:
        u, s, _ = np.linalg.svd(mat, full_matrices=False)
        emb = u[:, : min(dim, u.shape[1])] * s[: min(dim, s.shape[0])]
    except np.linalg.LinAlgError:
        emb = mat[:, : min(dim, mat.shape[1])]
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])), mode="constant")
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norm, 1e-8)
    emb[~np.isfinite(emb)] = 0.0
    return emb.astype(np.float32)


def local_sentence_transformer_embeddings(texts: list[str], model_name: str, hf_home: str | None) -> tuple[np.ndarray | None, str | None]:
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, local_files_only=True)
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def model_name_for_key(model_key: str) -> str:
    return {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "all_minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
    }.get(model_key, model_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationales", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--model-key", default="minilm")
    parser.add_argument("--config", required=True)
    parser.add_argument("--hf-home", default="E:/huggingface")
    parser.add_argument("--output-npy", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--manifest", default="outputs/manifests/10_FLOW_SEMANTIC_EMBEDDINGS_V5.manifest.json")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    failures: list[str] = []
    rationales = pd.read_parquet(args.rationales) if Path(args.rationales).exists() else pd.DataFrame()
    contexts = pd.read_parquet(args.contexts) if Path(args.contexts).exists() else pd.DataFrame()
    if rationales.empty:
        failures.append(f"rationales missing or empty: {args.rationales}")
    if contexts.empty:
        failures.append(f"contexts missing or empty: {args.contexts}")

    df = build_texts(rationales, contexts) if not rationales.empty and not contexts.empty else pd.DataFrame()
    if df.empty and not failures:
        failures.append("rationales and contexts do not join on sample_id")
    texts = df["embedding_text"].astype(str).tolist() if len(df) else []
    backend = "not_run"
    model_name = model_name_for_key(args.model_key)
    model_failure = None
    embeddings: np.ndarray
    if texts:
        embeddings_local, model_failure = local_sentence_transformer_embeddings(texts, model_name, args.hf_home)
        if embeddings_local is not None:
            embeddings = embeddings_local
            backend = "sentence_transformers_local"
        else:
            embeddings = tfidf_svd_embeddings(texts, dim=args.dim)
            backend = "tfidf_svd_numpy"
    else:
        embeddings = np.zeros((0, args.dim), dtype=np.float32)

    if len(embeddings) != len(df):
        failures.append(f"embedding rows {len(embeddings)} != index rows {len(df)}")
    if embeddings.size and (not np.isfinite(embeddings).all()):
        failures.append("embeddings contain non-finite values")
    if embeddings.shape[1] < 128:
        failures.append(f"embedding dim {embeddings.shape[1]} < 128")

    Path(args.output_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, embeddings)
    df = df.reset_index(drop=True).copy()
    df["row_idx"] = np.arange(len(df), dtype=int)
    index_cols = [col for col in ["sample_id", "candidate_id", "split", "target_label_5", "track"] if col in df.columns]
    if "row_idx" not in index_cols:
        index_cols.append("row_idx")
    Path(args.output_index).parent.mkdir(parents=True, exist_ok=True)
    df[index_cols].to_parquet(args.output_index, index=False)
    metrics = {
        "pipeline_pass": not failures,
        "claim_allowed": False,
        "rows": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "backend": backend,
        "model_name": model_name,
        "model_failure": model_failure,
        "local_files_only": True,
        "mean_l2_norm": float(np.linalg.norm(embeddings, axis=1).mean()) if len(embeddings) else 0.0,
    }
    write_json(args.metrics, metrics)
    write_manifest(args.manifest, [args.output_npy, args.output_index, args.metrics], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        [args.rationales, args.contexts, args.config],
        [args.output_npy, args.output_index, args.metrics, args.manifest, args.status],
        metrics,
        failures,
        status == "PASS",
    )
    print(json.dumps({"status": status, "metrics": metrics, "failures": failures}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
