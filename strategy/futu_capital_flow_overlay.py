"""Overlay Futu capital-flow labels onto existing A-share model signals."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from collector.config import settings
from collector.futu_client import FutuClient


FLOW_FIELDS = [
    "in_flow",
    "main_in_flow",
    "super_in_flow",
    "big_in_flow",
    "mid_in_flow",
    "sml_in_flow",
]


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def _window_sum(df: pd.DataFrame, field: str, window: int) -> float | None:
    if df.empty or field not in df.columns:
        return None
    values = pd.to_numeric(df[field], errors="coerce").dropna().tail(window)
    if values.empty:
        return None
    return float(values.sum())


def _positive_count(df: pd.DataFrame, field: str, window: int) -> int | None:
    if df.empty or field not in df.columns:
        return None
    values = pd.to_numeric(df[field], errors="coerce").dropna().tail(window)
    if values.empty:
        return None
    return int((values > 0).sum())


def summarize_capital_flow(
    code: str,
    flow_records: list[dict[str, Any]],
    distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize recent Futu capital-flow history into overlay features."""

    summary: dict[str, Any] = {
        "code": code,
        "capital_flow_status": "empty",
        "capital_flow_count": 0,
        "capital_flow_latest_date": "",
    }
    distribution = distribution or {}

    if flow_records:
        df = pd.DataFrame(flow_records).copy()
        if "date" in df.columns:
            df = df.sort_values("date")
        latest = df.iloc[-1].to_dict()
        summary.update(
            {
                "capital_flow_status": "ok",
                "capital_flow_count": len(df),
                "capital_flow_latest_date": latest.get("date", ""),
            }
        )
        for field in FLOW_FIELDS:
            summary[f"latest_{field}"] = _number(latest.get(field))
        for window in [3, 5, 10]:
            summary[f"main_{window}d_sum"] = _window_sum(df, "main_in_flow", window)
            summary[f"super_{window}d_sum"] = _window_sum(df, "super_in_flow", window)
            summary[f"big_{window}d_sum"] = _window_sum(df, "big_in_flow", window)
            summary[f"main_positive_{window}d"] = _positive_count(df, "main_in_flow", window)

    for field in [
        "net_main",
        "net_super",
        "net_big",
        "net_mid",
        "net_small",
        "capital_in_main",
        "capital_out_main",
        "update_time",
    ]:
        if field in distribution:
            summary[f"distribution_{field}"] = distribution.get(field)

    in_main = _number(distribution.get("capital_in_main"))
    out_main = _number(distribution.get("capital_out_main"))
    if in_main is not None and out_main and out_main > 0:
        summary["distribution_main_in_out_ratio"] = in_main / out_main

    return summary


def classify_capital_flow_overlay(
    row: pd.Series,
    *,
    confirm_latest_main: float = 10_000_000.0,
    confirm_5d_main: float = 20_000_000.0,
    risk_latest_main: float = -5_000_000.0,
    risk_5d_main: float = -20_000_000.0,
    min_positive_5d: int = 3,
) -> str:
    if row.get("capital_flow_status") != "ok":
        return "model_only_no_capital_flow"

    latest_main = _number(row.get("latest_main_in_flow")) or 0.0
    main_5d = _number(row.get("main_5d_sum")) or 0.0
    positive_5d = int(row.get("main_positive_5d") or 0)
    net_main = _number(row.get("distribution_net_main"))

    if latest_main <= risk_latest_main or main_5d <= risk_5d_main:
        return "risk_flag_main_outflow"

    distribution_ok = net_main is None or net_main > 0
    if (
        latest_main >= confirm_latest_main
        and main_5d >= confirm_5d_main
        and positive_5d >= min_positive_5d
        and distribution_ok
    ):
        return "capital_flow_confirm"

    if latest_main > 0 and (main_5d > 0 or (net_main is not None and net_main > 0)):
        return "capital_flow_watch"

    return "neutral"


def overlay_priority(label: str) -> int:
    if label.startswith("risk_flag"):
        return 0
    if label == "capital_flow_confirm":
        return 1
    if label == "capital_flow_watch":
        return 2
    return 3


def build_capital_flow_overlay(
    signal_df: pd.DataFrame,
    capital_flow_df: pd.DataFrame,
    *,
    signal_top_n: int = 30,
    confirm_latest_main: float = 10_000_000.0,
    confirm_5d_main: float = 20_000_000.0,
    risk_latest_main: float = -5_000_000.0,
    risk_5d_main: float = -20_000_000.0,
    min_positive_5d: int = 3,
) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()

    signals = signal_df.copy()
    if "rank" not in signals.columns:
        signals = signals.sort_values("score", ascending=False).reset_index(drop=True)
        signals["rank"] = np.arange(1, len(signals) + 1)
    signals = signals.sort_values("rank").head(signal_top_n).copy()
    signals = signals.rename(columns={"score": "model_score", "rank": "model_rank"})

    flow = capital_flow_df.copy()
    merged = signals.merge(flow, on="code", how="left")
    merged["capital_flow_status"] = merged["capital_flow_status"].fillna("missing")
    merged["capital_flow_label"] = merged.apply(
        lambda row: classify_capital_flow_overlay(
            row,
            confirm_latest_main=confirm_latest_main,
            confirm_5d_main=confirm_5d_main,
            risk_latest_main=risk_latest_main,
            risk_5d_main=risk_5d_main,
            min_positive_5d=min_positive_5d,
        ),
        axis=1,
    )
    merged["capital_flow_priority"] = merged["capital_flow_label"].map(overlay_priority)
    merged["model_top5"] = merged["model_rank"] <= 5
    merged = merged.sort_values(["capital_flow_priority", "model_rank"]).reset_index(drop=True)

    columns = [
        "code",
        "signal_date",
        "model_rank",
        "model_score",
        "model_top5",
        "capital_flow_label",
        "capital_flow_status",
        "capital_flow_latest_date",
        "capital_flow_count",
        "latest_main_in_flow",
        "latest_super_in_flow",
        "latest_big_in_flow",
        "main_3d_sum",
        "main_5d_sum",
        "main_10d_sum",
        "main_positive_5d",
        "distribution_net_main",
        "distribution_net_super",
        "distribution_net_big",
        "distribution_main_in_out_ratio",
        "distribution_update_time",
    ]
    return merged[[col for col in columns if col in merged.columns]]


def fetch_capital_flow_summaries(
    codes: list[str],
    *,
    host: str,
    port: int,
    start: str,
    end: str,
    period: str = "DAY",
    include_distribution: bool = True,
    connect_timeout: float = 8.0,
) -> pd.DataFrame:
    client = FutuClient(host, port)
    client.connect_timeout = connect_timeout
    if not client.connect():
        raise RuntimeError(f"failed to connect Futu OpenD at {host}:{port}")

    summaries: list[dict[str, Any]] = []
    try:
        for code in codes:
            try:
                flow_records = client.get_capital_flow(code, period_type=period, start=start, end=end)
                distribution = client.get_capital_distribution(code) if include_distribution else {}
                summaries.append(summarize_capital_flow(code, flow_records, distribution))
            except Exception as exc:
                summaries.append(
                    {
                        "code": code,
                        "capital_flow_status": "error",
                        "capital_flow_error": str(exc),
                        "capital_flow_count": 0,
                        "capital_flow_latest_date": "",
                    }
                )
    finally:
        client.disconnect()

    return pd.DataFrame(summaries)


def _default_start(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay Futu capital-flow labels onto A-share model signals.")
    parser.add_argument("--signal-csv", default=os.environ.get("SIGNAL_CSV", "~/quantpilot_data/signals/signal_latest.csv"))
    parser.add_argument("--capital-flow-csv", default=os.environ.get("CAPITAL_FLOW_CSV", ""))
    parser.add_argument("--fetch-latest", action="store_true")
    parser.add_argument("--host", default=os.environ.get("FUTU_HOST", settings.futu_host))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FUTU_PORT", settings.futu_port)))
    parser.add_argument("--period", default="DAY")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--include-distribution", action="store_true")
    parser.add_argument("--connect-timeout", type=float, default=8.0)
    parser.add_argument("--signal-top-n", type=int, default=30)
    parser.add_argument("--confirm-latest-main", type=float, default=10_000_000.0)
    parser.add_argument("--confirm-5d-main", type=float, default=20_000_000.0)
    parser.add_argument("--risk-latest-main", type=float, default=-5_000_000.0)
    parser.add_argument("--risk-5d-main", type=float, default=-20_000_000.0)
    parser.add_argument("--min-positive-5d", type=int, default=3)
    parser.add_argument("--output", default="~/quantpilot_data/output/futu_capital_flow_signal_overlay_latest.csv")
    parser.add_argument("--flow-output", default="~/quantpilot_data/output/futu_capital_flow_latest.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    signal_path = Path(args.signal_csv).expanduser()
    if not signal_path.exists():
        raise FileNotFoundError(f"signal csv not found: {signal_path}")
    signal_df = pd.read_csv(signal_path)

    signals = signal_df.copy()
    if "rank" not in signals.columns:
        signals = signals.sort_values("score", ascending=False).reset_index(drop=True)
        signals["rank"] = np.arange(1, len(signals) + 1)
    codes = signals.sort_values("rank").head(args.signal_top_n)["code"].astype(str).tolist()

    if args.fetch_latest or not args.capital_flow_csv:
        start = args.start or _default_start(args.days)
        capital_flow_df = fetch_capital_flow_summaries(
            codes,
            host=args.host,
            port=args.port,
            start=start,
            end=args.end,
            period=args.period,
            include_distribution=args.include_distribution,
            connect_timeout=args.connect_timeout,
        )
        flow_output = Path(args.flow_output).expanduser()
        flow_output.parent.mkdir(parents=True, exist_ok=True)
        capital_flow_df.to_csv(flow_output, index=False)
    else:
        capital_flow_path = Path(args.capital_flow_csv).expanduser()
        if not capital_flow_path.exists():
            raise FileNotFoundError(f"capital-flow csv not found: {capital_flow_path}")
        capital_flow_df = pd.read_csv(capital_flow_path)

    overlay = build_capital_flow_overlay(
        signal_df,
        capital_flow_df,
        signal_top_n=args.signal_top_n,
        confirm_latest_main=args.confirm_latest_main,
        confirm_5d_main=args.confirm_5d_main,
        risk_latest_main=args.risk_latest_main,
        risk_5d_main=args.risk_5d_main,
        min_positive_5d=args.min_positive_5d,
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_csv(output, index=False)
    print(overlay.head(min(len(overlay), 30)).to_string(index=False))
    print(f"Wrote overlay: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
