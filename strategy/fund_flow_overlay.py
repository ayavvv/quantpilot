"""Overlay Eastmoney fund-flow labels onto existing A-share model signals."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from collector.eastmoney_fund_flow import fetch_fund_flow_rank


def _to_float_or_none(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def classify_fund_flow_overlay(
    row: pd.Series,
    *,
    confirm_pct: float = 3.0,
    risk_pct: float = -3.0,
    confirm_rank: int = 500,
) -> str:
    if pd.isna(row.get("fund_flow_rank")):
        return "model_only_no_fund_flow"

    main_pct = _to_float_or_none(row.get("main_net_inflow_pct"))
    main_value = _to_float_or_none(row.get("main_net_inflow")) or 0.0
    rank = int(row.get("fund_flow_rank") or 999999)
    if main_value < 0 or (main_pct is not None and main_pct <= risk_pct):
        return "risk_flag_main_outflow"
    if main_pct is not None and main_pct >= confirm_pct and main_value > 0 and rank <= confirm_rank:
        return "fund_flow_confirm"
    if main_pct is None and main_value > 0 and rank <= confirm_rank:
        return "fund_flow_rank_confirm"
    if main_value > 0:
        return "fund_flow_watch"
    return "neutral"


def overlay_priority(label: str) -> int:
    if label.startswith("risk_flag"):
        return 0
    if label in {"fund_flow_confirm", "fund_flow_rank_confirm"}:
        return 1
    if label == "fund_flow_watch":
        return 2
    return 3


def build_fund_flow_overlay(
    signal_df: pd.DataFrame,
    fund_flow_df: pd.DataFrame,
    *,
    signal_top_n: int = 100,
    confirm_pct: float = 3.0,
    risk_pct: float = -3.0,
    confirm_rank: int = 500,
) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()

    signals = signal_df.copy()
    if "rank" not in signals.columns:
        signals = signals.sort_values("score", ascending=False).reset_index(drop=True)
        signals["rank"] = np.arange(1, len(signals) + 1)
    signals = signals.sort_values("rank").head(signal_top_n).copy()
    signals = signals.rename(columns={"score": "model_score", "rank": "model_rank"})

    fund = fund_flow_df.copy()
    if fund.empty:
        merged = signals.copy()
        for col in [
            "fund_flow_rank",
            "main_net_inflow",
            "main_net_inflow_pct",
            "super_net_inflow",
            "big_net_inflow",
            "fund_flow_source",
            "update_time",
        ]:
            merged[col] = np.nan
    else:
        keep_cols = [
            "code",
            "fund_flow_rank",
            "name",
            "latest_price",
            "change_pct",
            "main_net_inflow",
            "main_net_inflow_pct",
            "super_net_inflow",
            "super_net_inflow_pct",
            "big_net_inflow",
            "big_net_inflow_pct",
            "fund_flow_source",
            "update_time",
        ]
        fund = fund[[col for col in keep_cols if col in fund.columns]]
        merged = signals.merge(fund, on="code", how="left")

    merged["fund_flow_label"] = merged.apply(
        lambda row: classify_fund_flow_overlay(
            row,
            confirm_pct=confirm_pct,
            risk_pct=risk_pct,
            confirm_rank=confirm_rank,
        ),
        axis=1,
    )
    merged["fund_flow_priority"] = merged["fund_flow_label"].map(overlay_priority)
    merged["model_top5"] = merged["model_rank"] <= 5
    merged = merged.sort_values(["fund_flow_priority", "model_rank"]).reset_index(drop=True)

    columns = [
        "code",
        "signal_date",
        "model_rank",
        "model_score",
        "model_top5",
        "fund_flow_label",
        "fund_flow_rank",
        "name",
        "latest_price",
        "change_pct",
        "main_net_inflow",
        "main_net_inflow_pct",
        "super_net_inflow",
        "super_net_inflow_pct",
        "big_net_inflow",
        "big_net_inflow_pct",
        "fund_flow_source",
        "update_time",
    ]
    return merged[[col for col in columns if col in merged.columns]]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay Eastmoney fund-flow labels onto A-share model signals.")
    parser.add_argument("--signal-csv", default=os.environ.get("SIGNAL_CSV", "~/quantpilot_data/signals/signal_latest.csv"))
    parser.add_argument("--fund-flow-csv", default=os.environ.get("FUND_FLOW_CSV", ""))
    parser.add_argument("--fetch-latest", action="store_true")
    parser.add_argument("--fund-flow-source", choices=["auto", "push2", "datacenter"], default="auto")
    parser.add_argument("--fund-flow-limit", type=int, default=5000)
    parser.add_argument("--signal-top-n", type=int, default=100)
    parser.add_argument("--confirm-pct", type=float, default=3.0)
    parser.add_argument("--risk-pct", type=float, default=-3.0)
    parser.add_argument("--confirm-rank", type=int, default=500)
    parser.add_argument("--output", default="~/quantpilot_data/output/eastmoney_fund_flow_signal_overlay_latest.csv")
    parser.add_argument("--rank-output", default="~/quantpilot_data/output/eastmoney_fund_flow_rank_latest.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    signal_path = Path(args.signal_csv).expanduser()
    if not signal_path.exists():
        raise FileNotFoundError(f"signal csv not found: {signal_path}")
    signal_df = pd.read_csv(signal_path)

    if args.fetch_latest or not args.fund_flow_csv:
        fund_flow_df = fetch_fund_flow_rank(limit=args.fund_flow_limit, source=args.fund_flow_source)
        rank_output = Path(args.rank_output).expanduser()
        rank_output.parent.mkdir(parents=True, exist_ok=True)
        fund_flow_df.to_csv(rank_output, index=False)
    else:
        fund_flow_path = Path(args.fund_flow_csv).expanduser()
        if not fund_flow_path.exists():
            raise FileNotFoundError(f"fund-flow csv not found: {fund_flow_path}")
        fund_flow_df = pd.read_csv(fund_flow_path)

    overlay = build_fund_flow_overlay(
        signal_df,
        fund_flow_df,
        signal_top_n=args.signal_top_n,
        confirm_pct=args.confirm_pct,
        risk_pct=args.risk_pct,
        confirm_rank=args.confirm_rank,
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_csv(output, index=False)
    print(overlay.head(min(len(overlay), 30)).to_string(index=False))
    print(f"Wrote overlay: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
