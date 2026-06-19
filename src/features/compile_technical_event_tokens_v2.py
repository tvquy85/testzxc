from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.artifacts import write_json, write_manifest, write_status


STEP = "07_TECH_EVENT_TOKENS_V2"


def strength_from_abs(value: float, medium: float, high: float) -> str:
    value = abs(float(value))
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def add_token(tokens: list[dict[str, Any]], token: str, value: float, direction: str, strength: str, column: str, rule: str) -> None:
    tokens.append(
        {
            "token": token,
            "value": None if pd.isna(value) else float(value),
            "direction_prior": direction,
            "strength": strength,
            "evidence_column": column,
            "rule": rule,
        }
    )


def compile_tokens(row: pd.Series) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    rsi = row.get("RSI_14")
    if pd.notna(rsi):
        if rsi >= 70:
            add_token(tokens, "RSI_OVERBOUGHT", rsi, "bearish_reversal_risk", "high" if rsi >= 80 else "medium", "RSI_14", "RSI_14 >= 70")
        elif rsi <= 30:
            add_token(tokens, "RSI_OVERSOLD", rsi, "bullish_reversal_potential", "high" if rsi <= 20 else "medium", "RSI_14", "RSI_14 <= 30")

    macd_hist = row.get("MACD_hist")
    if pd.notna(macd_hist):
        if macd_hist > 0:
            add_token(tokens, "MACD_BULLISH", macd_hist, "bullish_momentum", strength_from_abs(macd_hist, 0.1, 1.0), "MACD_hist", "MACD_hist > 0")
        elif macd_hist < 0:
            add_token(tokens, "MACD_BEARISH", macd_hist, "bearish_momentum", strength_from_abs(macd_hist, 0.1, 1.0), "MACD_hist", "MACD_hist < 0")

    price_sma = row.get("price_vs_SMA20")
    if pd.notna(price_sma):
        if price_sma >= 0:
            add_token(tokens, "PRICE_ABOVE_SMA20", price_sma, "bullish_trend", strength_from_abs(price_sma, 0.01, 0.05), "price_vs_SMA20", "price_vs_SMA20 >= 0")
        else:
            add_token(tokens, "PRICE_BELOW_SMA20", price_sma, "bearish_trend", strength_from_abs(price_sma, 0.01, 0.05), "price_vs_SMA20", "price_vs_SMA20 < 0")

    bb = row.get("Bollinger_position_20")
    if pd.notna(bb):
        if bb >= 0.9:
            add_token(tokens, "BOLLINGER_UPPER_PRESSURE", bb, "bearish_reversal_risk", "high" if bb >= 1.0 else "medium", "Bollinger_position_20", "Bollinger_position_20 >= 0.9")
        elif bb <= 0.1:
            add_token(tokens, "BOLLINGER_LOWER_PRESSURE", bb, "bullish_reversal_potential", "high" if bb <= 0.0 else "medium", "Bollinger_position_20", "Bollinger_position_20 <= 0.1")

    vol_z = row.get("volume_zscore_20")
    if pd.notna(vol_z):
        if vol_z >= 1.5:
            add_token(tokens, "VOLUME_SPIKE", vol_z, "attention_or_breakout_risk", "high" if vol_z >= 3.0 else "medium", "volume_zscore_20", "volume_zscore_20 >= 1.5")
        elif vol_z <= -1.0:
            add_token(tokens, "VOLUME_DRY_UP", vol_z, "low_conviction", "high" if vol_z <= -2.0 else "medium", "volume_zscore_20", "volume_zscore_20 <= -1.0")

    rel = row.get("relative_strength_vs_market_5d")
    if pd.notna(rel):
        if rel >= 0.02:
            add_token(tokens, "MARKET_OUTPERFORMANCE_5D", rel, "bullish_relative_strength", strength_from_abs(rel, 0.02, 0.05), "relative_strength_vs_market_5d", "relative_strength_vs_market_5d >= 0.02")
        elif rel <= -0.02:
            add_token(tokens, "MARKET_UNDERPERFORMANCE_5D", rel, "bearish_relative_weakness", strength_from_abs(rel, 0.02, 0.05), "relative_strength_vs_market_5d", "relative_strength_vs_market_5d <= -0.02")

    regime = row.get("regime_label")
    rank = row.get("market_vol_20d_rank")
    if regime in {"high_vol", "normal_vol", "low_vol"}:
        direction = {"high_vol": "higher_uncertainty", "normal_vol": "baseline_uncertainty", "low_vol": "lower_uncertainty"}[regime]
        add_token(tokens, f"{str(regime).upper()}_REGIME", rank if pd.notna(rank) else 0.5, direction, "medium", "regime_label", "regime_label in {low_vol, normal_vol, high_vol}")
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/indicators/technical_features_h1_v2.parquet")
    parser.add_argument("--output", default="data/indicators/technical_event_tokens_h1_v2.parquet")
    parser.add_argument("--rules-output", default="outputs/manifests/technical_token_rules.json")
    parser.add_argument("--status", default=f"outputs/status/{STEP}.status.json")
    parser.add_argument("--manifest", default=f"outputs/manifests/{STEP}.manifest.json")
    args = parser.parse_args()

    failures: list[str] = []
    features = pd.read_parquet(args.features)
    out = features[["sample_id", "ticker", "event_date", "regime_label"]].copy()
    token_lists = features.apply(compile_tokens, axis=1)
    out["technical_event_tokens_json"] = token_lists.apply(lambda x: json.dumps(x, ensure_ascii=False))
    out["technical_summary_text"] = token_lists.apply(lambda toks: "; ".join(t["token"] for t in toks))
    missing_rate = float((token_lists.apply(len) == 0).mean()) if len(out) else 1.0
    invalid = [i for i, toks in enumerate(token_lists.head(1000)) if any(not {"token", "value", "direction_prior", "strength", "rule"}.issubset(t) for t in toks)]
    if invalid:
        failures.append("token schema invalid in sample rows")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    rules = {
        "schema": ["token", "value", "direction_prior", "strength", "evidence_column", "rule"],
        "missing_token_policy": "No-token samples are not perfect grounding; downstream judges must handle not_applicable separately.",
        "supported_families": ["RSI", "MACD", "SMA20", "Bollinger", "volume", "market_relative_strength", "volatility_regime"],
    }
    write_json(args.rules_output, rules)
    write_manifest(args.manifest, [args.output, args.rules_output], STEP)
    status = "PASS" if not failures else "FAIL"
    write_status(
        args.status,
        STEP,
        status,
        inputs_checked=[args.features],
        outputs_created=[args.output, args.rules_output, args.manifest, args.status],
        metrics={"rows": int(len(out)), "missing_token_rate": missing_rate},
        failures=failures,
        next_step_allowed=status == "PASS",
    )
    print(json.dumps({"rows": len(out), "missing_token_rate": missing_rate}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

