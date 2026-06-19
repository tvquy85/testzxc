import os
import json
import argparse
import yaml

def find_models(search_roots, target_substrings):
    found = {k: None for k in target_substrings}
    
    for root in search_roots:
        expanded_root = os.path.expanduser(os.path.expandvars(root))
        if not os.path.exists(expanded_root):
            continue
            
        # We perform a constrained search (e.g., up to depth 2 or 3) to avoid huge delays
        for dirpath, dirnames, filenames in os.walk(expanded_root):
            if ".locks" in dirpath.split(os.sep):
                continue
            
            # check current directory name against substrings
            dir_name = os.path.basename(dirpath)
            for target in list(found.keys()):
                if found[target] is None and target in dir_name:
                    found[target] = dirpath
            
            # optionally, check if it's a model path but let's stick to dir_name
            # prune search if we've found all
            if all(v is not None for v in found.values()):
                break
                
        if all(v is not None for v in found.values()):
            break

    return found

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="outputs/status/model_inventory.json")
    parser.add_argument("--config", type=str, default="configs/local_paths.yaml")
    args = parser.parse_args()

    # Known search paths based on instructions and AGENTS.md
    search_roots = [
        "e:/huggingface",
        os.environ.get("HF_HOME", "") + "/hub",
        "~/.cache/huggingface/hub",
        "./models",
        "../models",
        "D:/models",
        "E:/models"
    ]
    search_roots = [r for r in search_roots if r]

    targets = {
        "main_explanation_llm": ["Qwen2.5-3B", "Qwen3-4B", "Llama-3.2-3B"],
        "qwen3_judge": ["Qwen3-4B"],
        "deepseek_reasoning_judge": ["DeepSeek-R1-Distill-Qwen-1.5B"],
        "fingpt_forecaster": ["FinGPT"],
        "finbert": ["ProsusAI--finbert"],
        "nli_cross_encoder": ["cross-encoder--nli-deberta-v3-small"],
        "moment_small": ["AutonLab--MOMENT-1-small"],
        "chronos_bolt_small": ["amazon--chronos-bolt-small"],
        "ttm_r2": ["ibm-granite--granite-timeseries-ttm-r2"],
        "fnspid_cache": ["Zihan1004--FNSPID"]
    }
    
    # Flatten targets for search
    all_substrings = set(item for sublist in targets.values() for item in sublist)
    
    found_paths = find_models(search_roots, list(all_substrings))
    
    # Map back to our config keys
    config_paths = {}
    for key, val_list in targets.items():
        config_paths[key] = None
        for val in val_list:
            if found_paths.get(val):
                config_paths[key] = found_paths[val]
                # Format to forward slashes for cross-platform neatness
                config_paths[key] = config_paths[key].replace('\\', '/')
                break
                
    # Prepare model inventory
    inventory = {k: v for k, v in found_paths.items() if v is not None}
    
    # Save inventory json
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(inventory, f, indent=2)

    # Save local_paths.yaml
    yaml_content = {
        "project_root": "./",
        "hf_home": os.environ.get("HF_HOME", None),
        "models": {k: config_paths[k] for k in targets.keys() if k != "fnspid_cache"},
        "datasets": {
            "fnspid_cache": config_paths["fnspid_cache"]
        }
    }
    
    os.makedirs(os.path.dirname(args.config), exist_ok=True)
    with open(args.config, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(json.dumps(inventory, indent=2))
    print(f"Inventory saved to {args.output}")
    print(f"Config saved to {args.config}")

if __name__ == "__main__":
    main()
