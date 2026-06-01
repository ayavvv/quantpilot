"""Overlay major-flow proxy labels onto existing model signals."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.major_force import MajorForceConfig, scan_major_force


RISK_STAGES = {"washout_or_risk", "distribution_risk"}
OVERHEAT_STAGES = {"markup_or_overheated"}


def _contains_reason(row: pd.Series, token: str) -> bool:
    return token in str(row.get("major_reason", ""))


def classify_overlay(row: pd.Series, confirm_rank: int = 100, confirm_score: float = 80.0) -> str:
    """Return a human-readable overlay label.

    The label intentionally does not produce an automatic buy/sell instruction.
    Prior validation showed the daily-bar proxy should be used as a secondary
    signal or risk flag, not as a standalone entry model.
    """

    if pd.isna(row.get("major_score")):
        return "model_only_no_major_signal"

    stage = str(row.get("major_stage", ""))
    if stage in RISK_STAGES or _contains_reason(row, "sharp_down_day"):
        return "risk_flag_major_washout"
    if stage in OVERHEAT_STAGES or _contains_reason(row, "price_extended"):
        return "risk_flag_overheated"
    rank_value = row.get("major_rank")
    major_rank = 999999 if pd.isna(rank_value) else int(rank_value)
    if (
        stage in {"stealth_accumulation", "accumulation_candidate"}
        and float(row.get("major_score") or 0.0) >= confirm_score
        and major_rank <= confirm_rank
    ):
        return "secondary_confirm_accumulation"
    if float(row.get("major_score") or 0.0) >= confirm_score:
        return "major_watch_no_entry"
    return "neutral"


def overlay_priority(label: str) -> int:
    if label.startswith("risk_flag"):
        return 0
    if label == "secondary_confirm_accumulation":
        return 1
    if label == "major_watch_no_entry":
        return 2
    return 3


def build_major_force_overlay(
    signal_df: pd.DataFrame,
    major_df: pd.DataFrame,
    *,
    signal_top_n: int = 50,
    confirm_rank: int = 100,
    confirm_score: float = 80.0,
) -> pd.DataFrame:
    """Merge model signals with major-flow proxy rows and add overlay labels."""

    if signal_df.empty:
        return pd.DataFrame()

    signals = signal_df.copy()
    if "rank" not in signals.columns:
        signals = signals.sort_values("score", ascending=False).reset_index(drop=True)
        signals["rank"] = np.arange(1, len(signals) + 1)
    signals = signals.sort_values("rank").head(signal_top_n).copy()
    signals = signals.rename(columns={"score": "model_score", "rank": "model_rank"})

    major = major_df.copy()
    if major.empty:
        merged = signals.copy()
        for col in [
            "major_score",
            "major_rank",
            "major_stage",
            "major_reason",
            "major_cmf_20",
            "major_amount_ratio_5_20",
            "major_today_chg_pct",
        ]:
            merged[col] = np.nan
    else:
        major_cols = {
            "score": "major_score",
            "rank": "major_rank",
            "stage": "major_stage",
            "reason": "major_reason",
            "cmf_20": "major_cmf_20",
            "amount_ratio_5_20": "major_amount_ratio_5_20",
            "today_chg_pct": "major_today_chg_pct",
        }
        keep_cols = ["code"] + [col for col in major_cols if col in major.columns]
        major = major[keep_cols].rename(columns=major_cols)
        merged = signals.merge(major, on="code", how="left")

    merged["overlay_label"] = merged.apply(
        lambda row: classify_overlay(row, confirm_rank=confirm_rank, confirm_score=confirm_score),
        axis=1,
    )
    merged["overlay_priority"] = merged["overlay_label"].map(overlay_priority)
    merged["model_top5"] = merged["model_rank"] <= 5
    merged = merged.sort_values(["overlay_priority", "model_rank"]).reset_index(drop=True)

    columns = [
        "code",
        "signal_date",
        "model_rank",
        "model_score",
        "model_top5",
        "overlay_label",
        "major_rank",
        "major_score",
        "major_stage",
        "major_reason",
        "major_cmf_20",
        "major_amount_ratio_5_20",
        "major_today_chg_pct",
    ]
    return merged[[col for col in columns if col in merged.columns]]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay major-flow proxy labels onto A-share model signals.")
    parser.add_argument("--signal-csv", default=os.environ.get("SIGNAL_CSV", "~/quantpilot_data/signals/signal_latest.csv"))
    parser.add_argument("--major-csv", default=os.environ.get("MAJOR_FORCE_CSV", "~/quantpilot_data/output/major_force_latest.csv"))
    parser.add_argument("--qlib-dir", default=os.environ.get("QLIB_DATA_DIR", "~/quantpilot_data/qlib_data"))
    parser.add_argument("--rescan-major", action="store_true", help="Recompute a wider major-force universe from Qlib.")
    parser.add_argument("--major-top-n", type=int, default=500, help="Major-force rows to scan when --rescan-major is set.")
    parser.add_argument("--signal-top-n", type=int, default=50)
    parser.add_argument("--confirm-rank", type=int, default=100)
    parser.add_argument("--confirm-score", type=float, default=80.0)
    parser.add_argument("--output", default="~/quantpilot_data/output/major_force_signal_overlay_latest.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    signal_path = Path(args.signal_csv).expanduser()
    if not signal_path.exists():
        raise FileNotFoundError(f"signal csv not found: {signal_path}")

    signal_df = pd.read_csv(signal_path)
    if args.rescan_major:
        major_df = scan_major_force(
            Path(args.qlib_dir).expanduser(),
            config=MajorForceConfig(),
            top_n=args.major_top_n,
        )
    else:
        major_path = Path(args.major_csv).expanduser()
        if not major_path.exists():
            raise FileNotFoundError(f"major-force csv not found: {major_path}")
        major_df = pd.read_csv(major_path)

    overlay = build_major_force_overlay(
        signal_df,
        major_df,
        signal_top_n=args.signal_top_n,
        confirm_rank=args.confirm_rank,
        confirm_score=args.confirm_score,
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_csv(output, index=False)

    print(overlay.head(min(len(overlay), 30)).to_string(index=False))
    print(f"Wrote overlay: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
