from __future__ import annotations

import argparse
import sys

from src.alignment.train_rwsft_v2 import main as train_rwsft_v2_main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="qwen3_4b")
    parser.add_argument("--config", default="configs/default_paths.yaml")
    parser.add_argument("--train-file", "--train-jsonl", dest="train_file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--metrics", default="outputs/metrics/14_rwsft_train_medium.json")
    parser.add_argument("--status", default="outputs/status/14_RWSFT_TRAIN_MEDIUM.status.json")
    parser.add_argument("--manifest", default="outputs/manifests/14_RWSFT_TRAIN_MEDIUM.manifest.json")
    parser.add_argument("--hf-home", default="E:/huggingface")
    parser.add_argument("--extra-site-packages", action="append", default=["C:/Python/Python311/Lib/site-packages"])
    args = parser.parse_args()

    # The v2 trainer already implements the real QLoRA path and records adapter artifacts.
    # grad_accum/save_steps are accepted for command compatibility; v2 trainer currently
    # uses max_steps as the hard technical gate.
    sys.argv = [
        "train_rwsft_v2",
        "--train",
        args.train_file,
        "--model",
        args.model_key,
        "--config",
        args.config,
        "--hf-home",
        args.hf_home,
        "--output-dir",
        args.output_dir,
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        str(args.batch_size),
        "--max-seq-length",
        str(args.max_seq_len),
        "--learning-rate",
        str(args.lr),
        "--metrics",
        args.metrics,
        "--status",
        args.status,
        "--manifest",
        args.manifest,
    ]
    for path in args.extra_site_packages:
        sys.argv.extend(["--extra-site-packages", path])
    return train_rwsft_v2_main()


if __name__ == "__main__":
    raise SystemExit(main())
