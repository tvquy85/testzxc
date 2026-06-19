from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/tables/baseline_suite_by_seed.csv")
    parser.add_argument("--output", default="outputs/tables/baseline_suite_aggregate.csv")
    args = parser.parse_args()
    import pandas as pd
    from pathlib import Path

    df = pd.read_csv(args.input)
    agg = df.groupby("baseline", dropna=False).agg(status=("status", lambda x: "RUN" if (x == "RUN").all() else "NOT_RUN"), seeds=("seed", "nunique")).reset_index()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.output, index=False)
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

