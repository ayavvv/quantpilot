#!/usr/bin/env python3
"""Repair Qlib instruments metadata from feature bins."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FREQ = "day"
PROBE_FIELDS = ("close", "open", "high", "low", "volume")
MARKETS = (
    ("sh", "SH."),
    ("sz", "SZ."),
    ("hk", "HK."),
    ("us", "US."),
    ("macro", "MACRO."),
)


def _fname_to_code(dirname: str) -> str:
    if dirname.startswith("_qlib_"):
        return dirname[len("_qlib_"):]
    return dirname.upper()


def _load_calendar(qlib_dir: Path) -> list[str]:
    cal_path = qlib_dir / "calendars" / f"{FREQ}.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"Calendar file not found: {cal_path}")
    lines = [line.strip() for line in cal_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Calendar file is empty: {cal_path}")
    return lines


def _read_range_from_bin(bin_path: Path, calendar: list[str]) -> tuple[str, str] | None:
    if not bin_path.exists():
        return None

    data = np.fromfile(str(bin_path), dtype="<f4")
    if len(data) <= 1:
        return None

    start_idx = int(data[0])
    values = data[1:]
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return None

    first_idx = start_idx + int(finite_idx[0])
    last_idx = start_idx + int(finite_idx[-1])
    if first_idx >= len(calendar) or last_idx >= len(calendar):
        raise IndexError(f"Bin range exceeds calendar: {bin_path}")

    return calendar[first_idx], calendar[last_idx]


def rebuild_instruments(qlib_dir: Path) -> tuple[int, str | None]:
    calendar = _load_calendar(qlib_dir)
    features_dir = qlib_dir / "features"
    instruments_dir = qlib_dir / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)

    instruments: dict[str, tuple[str, str]] = {}

    for feat_dir in sorted(features_dir.iterdir()):
        if not feat_dir.is_dir():
            continue

        code = _fname_to_code(feat_dir.name)
        best_range: tuple[str, str] | None = None

        for field in PROBE_FIELDS:
            value_range = _read_range_from_bin(feat_dir / f"{field}.{FREQ}.bin", calendar)
            if value_range is None:
                continue
            if best_range is None:
                best_range = value_range
            else:
                best_range = (
                    min(best_range[0], value_range[0]),
                    max(best_range[1], value_range[1]),
                )

        if best_range is not None:
            instruments[code] = best_range

    all_path = instruments_dir / "all.txt"
    all_lines = [f"{code}\t{start}\t{end}" for code, (start, end) in sorted(instruments.items())]
    all_path.write_text("\n".join(all_lines) + "\n")

    for market_tag, prefix in MARKETS:
        market_path = instruments_dir / f"{market_tag}.txt"
        market_lines = [
            f"{code}\t{start}\t{end}"
            for code, (start, end) in sorted(instruments.items())
            if code.startswith(prefix)
        ]
        if market_lines:
            market_path.write_text("\n".join(market_lines) + "\n")

    latest_a_share = None
    for code, (_, end) in instruments.items():
        if code.startswith(("SH.", "SZ.")) and (latest_a_share is None or end > latest_a_share):
            latest_a_share = end

    return len(instruments), latest_a_share


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Qlib instruments metadata from feature bins")
    parser.add_argument("--qlib-dir", required=True, help="Qlib data directory")
    args = parser.parse_args()

    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    count, latest_a_share = rebuild_instruments(qlib_dir)
    suffix = f", latest A-share={latest_a_share}" if latest_a_share else ""
    print(f"Rebuilt instruments metadata: {count} codes{suffix}")


if __name__ == "__main__":
    main()
