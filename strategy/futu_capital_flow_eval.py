"""Forward-return validation for archived Futu capital-flow overlays."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from converter.incremental import QlibBinReader


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("expected at least one integer")
    return tuple(values)


def discover_overlay_files(archive_dir: str | Path) -> list[Path]:
    path = Path(archive_dir).expanduser()
    if not path.exists():
        return []
    return sorted(path.glob("*_overlay.csv"))


def load_archived_overlays(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["overlay_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "signal_date" not in result.columns:
        raise ValueError("archived overlay is missing signal_date")
    if "capital_flow_label" not in result.columns:
        raise ValueError("archived overlay is missing capital_flow_label")
    return result


def _code_to_fname(code: str) -> str:
    replace_names = ["CON", "PRN", "AUX", "NUL"] + [f"COM{i}" for i in range(10)] + [f"LPT{i}" for i in range(10)]
    if str(code).upper() in replace_names:
        return "_qlib_" + str(code)
    return str(code)


def _read_close(reader: QlibBinReader, qlib_dir: Path, code: str) -> pd.Series:
    feat_dir = qlib_dir / "features" / _code_to_fname(code).lower()
    bin_path = feat_dir / "close.day.bin"
    if not bin_path.exists():
        return pd.Series(dtype="float64")
    data = np.fromfile(str(bin_path), dtype="<f4")
    if len(data) == 0:
        return pd.Series(dtype="float64")
    start_idx = int(data[0])
    values = data[1:]
    end_idx = start_idx + len(values)
    dates = reader.calendar[start_idx:end_idx]
    return pd.Series(values.astype("float64"), index=dates, name="close")


def _forward_return(close: pd.Series, as_of_date: str, horizon: int, entry_lag_days: int) -> float:
    valid = close[pd.to_numeric(close, errors="coerce").notna()]
    if as_of_date not in valid.index:
        return np.nan
    loc = valid.index.get_loc(as_of_date)
    if isinstance(loc, slice) or isinstance(loc, np.ndarray):
        return np.nan
    entry_idx = int(loc) + entry_lag_days
    exit_idx = entry_idx + horizon
    if entry_idx < 0 or exit_idx >= len(valid):
        return np.nan
    entry = float(valid.iloc[entry_idx])
    exit_ = float(valid.iloc[exit_idx])
    if not np.isfinite(entry) or not np.isfinite(exit_) or entry <= 0:
        return np.nan
    return exit_ / entry - 1.0


def evaluate_archived_capital_flow_overlays(
    qlib_dir: str | Path,
    overlays: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 3, 5),
    entry_lag_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate archived Futu overlay labels against future close returns.

    Returns ``(summary, rows)``. The row-level output contains one row per
    overlay pick and horizon; the summary groups by label and compares each
    label against the same-date archived overlay universe.
    """

    if overlays.empty:
        return pd.DataFrame(), pd.DataFrame()

    qlib_path = Path(qlib_dir).expanduser()
    reader = QlibBinReader(qlib_path)
    close_cache: dict[str, pd.Series] = {}
    row_records = []

    for _, row in overlays.iterrows():
        code = str(row.get("code", ""))
        signal_date = str(row.get("signal_date", ""))[:10]
        if not code or not signal_date:
            continue
        close = close_cache.get(code)
        if close is None:
            close = _read_close(reader, qlib_path, code)
            close_cache[code] = close
        if close.empty:
            continue
        base = row.to_dict()
        for horizon in horizons:
            value = _forward_return(close, signal_date, horizon, entry_lag_days)
            if np.isfinite(value):
                out = dict(base)
                out["horizon"] = horizon
                out["fwd_return"] = float(value)
                row_records.append(out)

    rows = pd.DataFrame(row_records)
    if rows.empty:
        return pd.DataFrame(), rows

    universe = (
        rows.groupby(["signal_date", "horizon"], as_index=False)
        .agg(universe_return=("fwd_return", "mean"), universe_count=("code", "count"))
    )
    daily = (
        rows.groupby(["signal_date", "horizon", "capital_flow_label"], as_index=False)
        .agg(
            selected_count=("code", "count"),
            avg_model_rank=("model_rank", "mean"),
            avg_return=("fwd_return", "mean"),
            median_return=("fwd_return", "median"),
            hit_rate=("fwd_return", lambda s: float((s > 0).mean())),
        )
        .merge(universe, on=["signal_date", "horizon"], how="left")
    )
    daily["alpha"] = daily["avg_return"] - daily["universe_return"]
    summary = (
        daily.groupby(["capital_flow_label", "horizon"], as_index=False)
        .agg(
            date_count=("signal_date", "nunique"),
            avg_selected_count=("selected_count", "mean"),
            avg_model_rank=("avg_model_rank", "mean"),
            avg_return=("avg_return", "mean"),
            median_return=("median_return", "median"),
            avg_universe_return=("universe_return", "mean"),
            avg_alpha=("alpha", "mean"),
            win_rate_days=("avg_return", lambda s: float((s > 0).mean())),
            avg_hit_rate=("hit_rate", "mean"),
        )
        .sort_values(["horizon", "capital_flow_label"])
        .reset_index(drop=True)
    )
    return summary, rows


def _as_int(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _as_float(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _summary_row_payload(row: pd.Series) -> dict[str, object]:
    return {
        "horizon": _as_int(row.get("horizon")),
        "date_count": _as_int(row.get("date_count")),
        "avg_return": _as_float(row.get("avg_return")),
        "avg_universe_return": _as_float(row.get("avg_universe_return")),
        "avg_alpha": _as_float(row.get("avg_alpha")),
        "avg_hit_rate": _as_float(row.get("avg_hit_rate")),
    }


def build_capital_flow_promotion_gate(
    summary: pd.DataFrame,
    *,
    min_date_count: int = 20,
    min_confirming_horizons: int = 2,
    risk_alpha_threshold: float = -0.005,
    confirm_alpha_threshold: float = 0.005,
    risk_max_hit_rate: float = 0.45,
    confirm_min_hit_rate: float = 0.55,
) -> dict[str, object]:
    """Build a conservative evidence gate before labels become trade rules."""

    criteria = {
        "min_date_count": int(min_date_count),
        "min_confirming_horizons": int(min_confirming_horizons),
        "risk_alpha_threshold": float(risk_alpha_threshold),
        "confirm_alpha_threshold": float(confirm_alpha_threshold),
        "risk_max_hit_rate": float(risk_max_hit_rate),
        "confirm_min_hit_rate": float(confirm_min_hit_rate),
    }
    gate: dict[str, object] = {
        "gate_version": "futu_capital_flow_promotion_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_action": "insufficient_samples",
        "criteria": criteria,
        "decisions": [],
        "message": "No forward-return samples are available yet; keep capital-flow labels advisory.",
    }

    required = {"capital_flow_label", "horizon", "date_count", "avg_alpha", "avg_hit_rate"}
    if summary.empty or not required.issubset(summary.columns):
        return gate

    decisions = []
    for label in sorted(summary["capital_flow_label"].dropna().astype(str).unique()):
        label_df = summary[summary["capital_flow_label"].astype(str) == label].copy()
        payload_rows = [_summary_row_payload(row) for _, row in label_df.sort_values("horizon").iterrows()]
        eligible_rows = [row for row in payload_rows if row["date_count"] >= min_date_count]
        kind = "monitor"
        matching_rows: list[dict[str, object]] = []
        status = "keep_advisory"
        recommendation = "Keep collecting samples before changing trading rules."

        if label.startswith("risk_flag"):
            kind = "risk"
            matching_rows = [
                row
                for row in eligible_rows
                if row["avg_alpha"] <= risk_alpha_threshold and row["avg_hit_rate"] <= risk_max_hit_rate
            ]
            if not eligible_rows:
                status = "insufficient_samples"
            elif len(matching_rows) >= min_confirming_horizons:
                status = "candidate_filter_review"
                recommendation = "Review this label for an automatic filter or score downgrade."
        elif label == "capital_flow_confirm":
            kind = "confirm"
            matching_rows = [
                row
                for row in eligible_rows
                if row["avg_alpha"] >= confirm_alpha_threshold and row["avg_hit_rate"] >= confirm_min_hit_rate
            ]
            if not eligible_rows:
                status = "insufficient_samples"
            elif len(matching_rows) >= min_confirming_horizons:
                status = "candidate_boost_review"
                recommendation = "Review this label for a score boost or tie-breaker."
        elif not eligible_rows:
            status = "insufficient_samples"

        decisions.append(
            {
                "label": label,
                "kind": kind,
                "status": status,
                "recommendation": recommendation,
                "eligible_horizon_count": len(eligible_rows),
                "matching_horizon_count": len(matching_rows),
                "rows": payload_rows,
            }
        )

    gate["decisions"] = decisions
    statuses = {str(decision["status"]) for decision in decisions}
    if "candidate_filter_review" in statuses:
        gate["overall_action"] = "review_filter"
        gate["message"] = "Capital-flow risk labels have enough evidence for manual filter review."
    elif "candidate_boost_review" in statuses:
        gate["overall_action"] = "review_boost"
        gate["message"] = "Capital-flow confirm labels have enough evidence for manual boost review."
    elif statuses and statuses != {"insufficient_samples"}:
        gate["overall_action"] = "keep_advisory"
        gate["message"] = "Evidence is not strong enough to promote capital-flow labels; keep advisory."

    return gate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate archived Futu capital-flow overlays.")
    parser.add_argument("--qlib-dir", default=os.environ.get("QLIB_DATA_DIR", "~/quantpilot_data/qlib_data"))
    parser.add_argument("--archive-dir", default="~/quantpilot_data/capital_flow/futu")
    parser.add_argument("--overlay-csv", action="append", default=[])
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--entry-lag-days", type=int, default=1)
    parser.add_argument("--output-dir", default="~/quantpilot_data/output/futu_capital_flow_eval")
    parser.add_argument("--gate-min-date-count", type=int, default=int(os.environ.get("CAPITAL_FLOW_GATE_MIN_DATE_COUNT", "20")))
    parser.add_argument("--gate-min-confirming-horizons", type=int, default=int(os.environ.get("CAPITAL_FLOW_GATE_MIN_CONFIRMING_HORIZONS", "2")))
    parser.add_argument("--gate-risk-alpha-threshold", type=float, default=float(os.environ.get("CAPITAL_FLOW_GATE_RISK_ALPHA_THRESHOLD", "-0.005")))
    parser.add_argument("--gate-confirm-alpha-threshold", type=float, default=float(os.environ.get("CAPITAL_FLOW_GATE_CONFIRM_ALPHA_THRESHOLD", "0.005")))
    parser.add_argument("--gate-risk-max-hit-rate", type=float, default=float(os.environ.get("CAPITAL_FLOW_GATE_RISK_MAX_HIT_RATE", "0.45")))
    parser.add_argument("--gate-confirm-min-hit-rate", type=float, default=float(os.environ.get("CAPITAL_FLOW_GATE_CONFIRM_MIN_HIT_RATE", "0.55")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = [Path(item).expanduser() for item in args.overlay_csv]
    if not paths:
        paths = discover_overlay_files(args.archive_dir)
    overlays = load_archived_overlays(paths)
    summary, rows = evaluate_archived_capital_flow_overlays(
        args.qlib_dir,
        overlays,
        horizons=_parse_int_tuple(args.horizons),
        entry_lag_days=max(0, args.entry_lag_days),
    )

    if summary.empty:
        print("No forward-return rows produced.")
    else:
        display = summary.copy()
        for col in ["avg_return", "median_return", "avg_universe_return", "avg_alpha", "win_rate_days", "avg_hit_rate"]:
            if col in display.columns:
                display[col] = display[col].map(lambda v: f"{v:.2%}")
        print(display.to_string(index=False))

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    rows_path = output_dir / "rows.csv"
    gate_path = output_dir / "gate.json"
    summary.to_csv(summary_path, index=False)
    rows.to_csv(rows_path, index=False)
    gate = build_capital_flow_promotion_gate(
        summary,
        min_date_count=max(1, args.gate_min_date_count),
        min_confirming_horizons=max(1, args.gate_min_confirming_horizons),
        risk_alpha_threshold=args.gate_risk_alpha_threshold,
        confirm_alpha_threshold=args.gate_confirm_alpha_threshold,
        risk_max_hit_rate=args.gate_risk_max_hit_rate,
        confirm_min_hit_rate=args.gate_confirm_min_hit_rate,
    )
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote rows: {rows_path}")
    print(f"Wrote gate: {gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
