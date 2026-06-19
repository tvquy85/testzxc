from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "05_LABELS_ABNORMAL_RETURN_AND_BALANCE"


def distribution(df, group_cols):
    grouped = df.groupby(group_cols + ["label_5"], dropna=False).size().reset_index(name="count")
    totals = df.groupby(group_cols, dropna=False).size().reset_index(name="total")
    merged = grouped.merge(totals, on=group_cols, how="left")
    merged["fraction"] = merged["count"] / merged["total"]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/labels/labels_h1_abnormal.parquet")
    parser.add_argument("--features", default="data/indicators/technical_features_h1.parquet")
    parser.add_argument("--output", default="outputs/metrics/label_distribution_h1.json")
    parser.add_argument("--split-table", default="outputs/tables/label_distribution_by_split.csv")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    import pandas as pd

    failures: list[str] = []
    df = pd.read_parquet(args.labels)
    if "event_date" in df.columns:
        df["year"] = pd.to_datetime(df["event_date"]).dt.year
    if "regime_label" not in df.columns and Path(args.features).exists():
        features = pd.read_parquet(args.features, columns=["sample_id", "regime_label"])
        df = df.merge(features, on="sample_id", how="left")

    report = {
        "rows": int(len(df)),
        "total": df["label_5"].value_counts(dropna=False).astype(int).to_dict(),
        "by_split": distribution(df, ["split"]).to_dict(orient="records") if "split" in df.columns else [],
        "by_year": distribution(df, ["year"]).to_dict(orient="records") if "year" in df.columns else [],
        "by_ticker": distribution(df, ["ticker"]).to_dict(orient="records") if "ticker" in df.columns else [],
        "by_regime": distribution(df, ["regime_label"]).to_dict(orient="records") if "regime_label" in df.columns else [],
    }
    if len(report["total"]) != 5:
        failures.append("label distribution does not include exactly 5 classes")
    Path(args.split_table).parent.mkdir(parents=True, exist_ok=True)
    if "split" in df.columns:
        distribution(df, ["split"]).to_csv(args.split_table, index=False)
    else:
        failures.append("labels missing split column")
        pd.DataFrame().to_csv(args.split_table, index=False)
    write_json(args.output, report)
    write_manifest(args.manifest, [args.output, args.split_table], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.labels, args.features],
        outputs_created=[args.output, args.split_table, args.manifest, args.status],
        metrics={"rows": int(len(df)), "class_count": len(report["total"])},
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps({"rows": report["rows"], "total": report["total"]}, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

