import os
import json
import torch
import platform
import transformers

def main():
    report = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else None,
        "transformers_version": transformers.__version__,
        "hf_home": os.environ.get("HF_HOME", "~/.cache/huggingface")
    }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="outputs/status/env_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
