from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_manifest, write_status


STEP = "19_ABLATION_AND_STATISTICAL_TESTS"
ABLATIONS = ["A0_Full_FIRE_Fin", "A1_No_Technical_Indicators", "A2_No_News", "A3_No_Technical_Event_Tokens", "A4_No_Flow_Reward", "A5_No_Grounding_Reward", "A6_No_Regime_Noise", "A7_SFT_Only", "A8_No_Counterfactual_Pairs"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablations.yaml")
    parser.add_argument("--output", default="outputs/tables/ablation_suite_results.csv")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()
    import pandas as pd

    df = pd.DataFrame({"ablation": ABLATIONS, "status": "NOT_RUN", "macro_f1": None, "daily_sharpe": None})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    metrics = {"ablation_count": len(df), "not_run_count": int((df["status"] == "NOT_RUN").sum())}
    failures = ["real ablations have not been run; table is marked NOT_RUN and must not be used as evidence"]
    write_manifest(args.manifest, [args.output], STEP)
    write_status(args.status, STEP, "FAIL", [args.config], [args.output, args.manifest, args.status], metrics, failures, False)
    print(json.dumps(metrics, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

