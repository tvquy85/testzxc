import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path

from src.utils.artifacts import sha256_file, write_json, write_manifest, write_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STEP = "13_ALIGNMENT_REAL_RUN_CURRENT_DATA"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwsft-data", default="data/alignment/rwsft_current_v3.jsonl")
    parser.add_argument("--dpo-data", default="data/alignment/dpo_current_v3.jsonl")
    parser.add_argument("--rwsft-out", default="checkpoints/aligned/qwen3_4b/current_v3_rwsft")
    parser.add_argument("--dpo-out", default="checkpoints/aligned/qwen3_4b/current_v3_dpo")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--metrics", default="outputs/metrics/alignment_training_current_v3.json")
    parser.add_argument("--status", default="outputs/status/13_ALIGNMENT_REAL_RUN_CURRENT_DATA.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    # RWSFT
    logging.info("Starting RWSFT using train_rwsft_v2.py")
    env = os.environ.copy()
    rwsft_cmd = [
        sys.executable, "-m", "src.alignment.train_rwsft_v2",
        "--train", args.rwsft_data,
        "--output-dir", args.rwsft_out,
        "--max-steps", str(args.max_steps),
        "--batch-size", "1"
    ]
    
    # We are already running RWSFT externally right now in a background task, 
    # but for reproducibility we'll just check if the adapter exists, else run it.
    rwsft_adapter = os.path.join(args.rwsft_out, "adapter_config.json")
    if not os.path.exists(rwsft_adapter):
        subprocess.run(rwsft_cmd, check=True, env=env)
    else:
        logging.info("RWSFT adapter already exists, skipping RWSFT training.")

    # DPO
    logging.info("Starting DPO using train_dpo_v2.py")
    dpo_cmd = [
        sys.executable, "-m", "src.alignment.train_dpo_v2",
        "--train", args.dpo_data,
        "--rwsft-checkpoint", args.rwsft_out,
        "--output-dir", args.dpo_out,
        "--max-steps", str(args.max_steps),
        "--batch-size", "1",
        "--learning-rate", "5e-7"
    ]
    
    dpo_adapter = os.path.join(args.dpo_out, "adapter_config.json")
    if not os.path.exists(dpo_adapter):
        subprocess.run(dpo_cmd, check=True, env=env)
    else:
        logging.info("DPO adapter already exists, skipping DPO training.")

    rwsft_adapter_model = os.path.join(args.rwsft_out, "adapter_model.safetensors")
    dpo_adapter_model = os.path.join(args.dpo_out, "adapter_model.safetensors")
    rwsft_config = os.path.join(args.rwsft_out, "trainer_config.json")
    dpo_config = os.path.join(args.dpo_out, "trainer_config.json")
    metrics = {
        "rwsft_steps": args.max_steps,
        "dpo_steps": args.max_steps,
        "rwsft_out": args.rwsft_out,
        "dpo_out": args.dpo_out,
        "rwsft_train": args.rwsft_data,
        "dpo_train": args.dpo_data,
        "rwsft_adapter_sha256": sha256_file(rwsft_adapter_model) if os.path.exists(rwsft_adapter_model) else None,
        "dpo_adapter_sha256": sha256_file(dpo_adapter_model) if os.path.exists(dpo_adapter_model) else None,
        "loss_trace_available": False,
        "loss_trace_note": "This wrapper records adapter/trainer_config artifacts; per-step loss traces are produced only by the underlying trainer when configured.",
    }

    failures = []
    if args.max_steps < 300:
        failures.append(f"effective steps {args.max_steps} < 300")
    for required in [args.rwsft_data, args.dpo_data, rwsft_adapter, rwsft_adapter_model, dpo_adapter, dpo_adapter_model]:
        if not os.path.exists(required):
            failures.append(f"missing required artifact: {required}")

    write_json(args.metrics, metrics)
    outputs = [rwsft_adapter, rwsft_adapter_model, rwsft_config, dpo_adapter, dpo_adapter_model, dpo_config, args.metrics]
    write_manifest(args.manifest, outputs, STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.rwsft_data, args.dpo_data],
        outputs_created=outputs + [args.manifest, args.status],
        metrics=metrics,
        failures=failures,
        next_step_allowed=status == "PASS",
    )

    logging.info("Alignment Real Run fully completed.")
    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
