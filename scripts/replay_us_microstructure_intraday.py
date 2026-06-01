"""Build intraday replay calibration samples from collected US microstructure data."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.collect_us_microstructure import DEFAULT_NAS_DIR, _copy_to_nas
from strategy.us_microstructure_features import (
    MicrostructureFeatureConfig,
    build_feature_table,
    normalize_us_symbols,
)
from strategy.us_microstructure_signals import MicrostructureSignalConfig, score_microstructure_signals


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
DEFAULT_BASE_DIR = Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DATA_DIR / "us_microstructure")))
US_EASTERN = ZoneInfo("America/New_York")


def _date_default() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_symbols(value: str) -> list[str]:
    return normalize_us_symbols(item for item in str(value or "").split(",") if item.strip())


def _feature_cache_path(base_dir: Path, date: str) -> Path:
    return base_dir / "features_1m" / f"date={date}" / "part-us-microstructure-features.parquet"


def load_or_build_features(
    base_dir: str | Path,
    *,
    date: str,
    symbols: Iterable[object] | None = None,
    book_levels: int = 5,
    rebuild: bool = False,
) -> pd.DataFrame:
    base = Path(base_dir).expanduser()
    cache_path = _feature_cache_path(base, date)
    if cache_path.exists() and not rebuild:
        frame = pd.read_parquet(cache_path)
        if symbols:
            allowed = set(normalize_us_symbols(symbols))
            frame = frame[frame["symbol"].astype(str).isin(allowed)].copy()
        return frame
    return build_feature_table(
        base,
        date=date,
        symbols=symbols,
        config=MicrostructureFeatureConfig(book_levels=max(1, int(book_levels))),
    )


def _regular_session_elapsed(cutoff: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(cutoff)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    eastern = timestamp.tz_convert(US_EASTERN)
    session_open = eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    session_close = eastern.replace(hour=16, minute=0, second=0, microsecond=0)
    if eastern < session_open:
        return 0
    clipped = min(eastern, session_close)
    return max(0, int((clipped - session_open).total_seconds() // 60) + 1)


def _regular_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    frame = features.copy()
    frame["minute"] = pd.to_datetime(frame["minute"], errors="coerce", utc=True)
    frame = frame[frame["minute"].notna()].copy()
    if "is_regular_session" in frame.columns:
        frame = frame[frame["is_regular_session"].fillna(False)].copy()
    return frame.sort_values(["symbol", "minute"]).reset_index(drop=True)


def build_replay_cutoffs(
    features: pd.DataFrame,
    *,
    interval_minutes: int,
    min_elapsed_minutes: int,
    max_horizon_minutes: int,
) -> list[pd.Timestamp]:
    regular = _regular_features(features)
    if regular.empty:
        return []
    minutes = sorted(pd.to_datetime(regular["minute"], errors="coerce", utc=True).dropna().unique())
    if not minutes:
        return []
    latest_available = pd.Timestamp(minutes[-1])
    cutoffs: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    interval = max(1, int(interval_minutes))
    for raw_minute in minutes:
        minute = pd.Timestamp(raw_minute)
        elapsed = _regular_session_elapsed(minute)
        if elapsed < int(min_elapsed_minutes):
            continue
        if minute + pd.Timedelta(minutes=int(max_horizon_minutes)) > latest_available:
            continue
        if elapsed % interval != 0:
            continue
        if minute not in seen:
            cutoffs.append(minute)
            seen.add(minute)
    return cutoffs


def _apply_elapsed_coverage(features: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = _regular_features(features)
    frame = frame[frame["minute"] <= cutoff].copy()
    if frame.empty:
        return frame
    expected = max(1, _regular_session_elapsed(cutoff))
    for symbol, indexes in frame.groupby("symbol", sort=False).groups.items():
        part = frame.loc[indexes]
        trade_minutes = int(part.get("has_trade_data", pd.Series(False, index=part.index)).fillna(False).sum())
        book_minutes = int(part.get("has_book_data", pd.Series(False, index=part.index)).fillna(False).sum())
        quote_minutes = int(part.get("has_quote_data", pd.Series(False, index=part.index)).fillna(False).sum())
        coverage_minutes = int(
            (
                part.get("has_trade_data", pd.Series(False, index=part.index)).fillna(False)
                | part.get("has_book_data", pd.Series(False, index=part.index)).fillna(False)
            ).sum()
        )
        frame.loc[indexes, "regular_session_minutes_seen"] = int(len(part))
        frame.loc[indexes, "trade_coverage_minutes"] = trade_minutes
        frame.loc[indexes, "book_coverage_minutes"] = book_minutes
        frame.loc[indexes, "quote_coverage_minutes"] = quote_minutes
        frame.loc[indexes, "coverage_minutes"] = coverage_minutes
        frame.loc[indexes, "trade_coverage_ratio_regular"] = min(1.0, trade_minutes / expected)
        frame.loc[indexes, "book_coverage_ratio_regular"] = min(1.0, book_minutes / expected)
        frame.loc[indexes, "quote_coverage_ratio_regular"] = min(1.0, quote_minutes / expected)
        frame.loc[indexes, "coverage_ratio_regular"] = min(1.0, coverage_minutes / expected)
    return frame


def _price_series_by_symbol(features: pd.DataFrame) -> dict[str, pd.Series]:
    regular = _regular_features(features)
    if regular.empty or "reference_price" not in regular.columns:
        return {}
    result: dict[str, pd.Series] = {}
    for symbol, part in regular.groupby("symbol", sort=True):
        clean = part[["minute", "reference_price"]].copy()
        clean["reference_price"] = pd.to_numeric(clean["reference_price"], errors="coerce")
        clean = clean.dropna(subset=["minute", "reference_price"]).sort_values("minute")
        if clean.empty:
            continue
        series = clean.drop_duplicates("minute", keep="last").set_index("minute")["reference_price"].astype("float64")
        result[str(symbol)] = series
    return result


def _price_at_or_after(series: pd.Series, timestamp: pd.Timestamp) -> tuple[pd.Timestamp | pd.NaT, float]:
    if series.empty:
        return pd.NaT, np.nan
    index = pd.DatetimeIndex(series.index)
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    position = index.searchsorted(ts)
    if position >= len(series):
        return pd.NaT, np.nan
    return pd.Timestamp(index[position]), float(series.iloc[position])


def build_intraday_replay(
    features: pd.DataFrame,
    *,
    date: str,
    horizons_minutes: tuple[int, ...] = (30, 60),
    cutoff_interval_minutes: int = 30,
    min_elapsed_minutes: int = 60,
    min_event_score: float = 65.0,
    entry_lag_minutes: int = 1,
    benchmark: str = "US.SPY",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    max_horizon = max(int(item) for item in horizons_minutes) if horizons_minutes else 0
    cutoffs = build_replay_cutoffs(
        features,
        interval_minutes=cutoff_interval_minutes,
        min_elapsed_minutes=min_elapsed_minutes,
        max_horizon_minutes=max_horizon,
    )
    signal_cfg = MicrostructureSignalConfig()
    gate = {"state": "warmup", "validated": False, "reason": "intraday replay calibration"}
    events: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        cutoff_features = _apply_elapsed_coverage(features, cutoff)
        if cutoff_features.empty:
            continue
        signals = score_microstructure_signals(cutoff_features, config=signal_cfg, validation_gate=gate, include_diagnostic=True)
        if signals.empty:
            continue
        signals = signals[pd.to_numeric(signals["side_score"], errors="coerce").fillna(0.0) >= float(min_event_score)].copy()
        if signals.empty:
            continue
        signals["signal_date"] = date
        signals["cutoff_time"] = pd.Timestamp(cutoff).isoformat()
        signals["elapsed_regular_minutes"] = _regular_session_elapsed(cutoff)
        signals["validation_scope"] = "intraday_replay"
        signals["is_final_report"] = False
        signals["event_id"] = (
            signals["signal_date"].astype(str)
            + "|"
            + signals["cutoff_time"].astype(str)
            + "|"
            + signals["symbol"].astype(str)
            + "|"
            + signals["side"].astype(str)
        )
        events.append(signals)
    if not events:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    event_frame = pd.concat(events, ignore_index=True)
    event_frame = event_frame.drop_duplicates("event_id", keep="last").sort_values(["cutoff_time", "rank"]).reset_index(drop=True)
    returns = compute_intraday_replay_returns(
        event_frame,
        features,
        horizons_minutes=horizons_minutes,
        entry_lag_minutes=entry_lag_minutes,
        benchmark=benchmark,
    )
    metrics = build_intraday_replay_metrics(returns)
    return event_frame, returns, metrics


def compute_intraday_replay_returns(
    events: pd.DataFrame,
    features: pd.DataFrame,
    *,
    horizons_minutes: tuple[int, ...] = (30, 60),
    entry_lag_minutes: int = 1,
    benchmark: str = "US.SPY",
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    prices = _price_series_by_symbol(features)
    benchmark_series = prices.get(str(benchmark), pd.Series(dtype="float64"))
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        symbol = str(event.get("symbol") or "")
        series = prices.get(symbol, pd.Series(dtype="float64"))
        cutoff = pd.Timestamp(event.get("cutoff_time"))
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        entry_target = cutoff + pd.Timedelta(minutes=int(entry_lag_minutes))
        entry_time, entry_price = _price_at_or_after(series, entry_target)
        bench_entry_time, bench_entry_price = _price_at_or_after(benchmark_series, entry_target)
        if not math.isfinite(entry_price) or entry_price <= 0:
            continue
        for horizon in horizons_minutes:
            exit_target = cutoff + pd.Timedelta(minutes=int(horizon))
            exit_time, exit_price = _price_at_or_after(series, exit_target)
            bench_exit_time, bench_exit_price = _price_at_or_after(benchmark_series, exit_target)
            if not math.isfinite(exit_price) or exit_price <= 0:
                continue
            fwd = exit_price / entry_price - 1.0
            bench = (
                bench_exit_price / bench_entry_price - 1.0
                if math.isfinite(bench_entry_price) and math.isfinite(bench_exit_price) and bench_entry_price > 0
                else np.nan
            )
            side = str(event.get("side") or "").lower()
            alpha = fwd - bench if side == "accumulation" and math.isfinite(bench) else np.nan
            if side == "distribution" and math.isfinite(bench):
                alpha = bench - fwd
            directional_hit = bool(fwd > 0) if side == "accumulation" else bool(fwd < 0)
            rows.append(
                {
                    "event_id": event.get("event_id"),
                    "validation_scope": "intraday_replay",
                    "signal_date": event.get("signal_date"),
                    "cutoff_time": event.get("cutoff_time"),
                    "symbol": symbol,
                    "side": side,
                    "side_score": float(event.get("side_score") or 0.0),
                    "confidence": event.get("confidence"),
                    "data_quality_pass": bool(event.get("data_quality_pass")),
                    "horizon_minutes": int(horizon),
                    "entry_time": pd.Timestamp(entry_time).isoformat() if not pd.isna(entry_time) else "",
                    "exit_time": pd.Timestamp(exit_time).isoformat() if not pd.isna(exit_time) else "",
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "fwd_return": float(fwd),
                    "benchmark": benchmark,
                    "benchmark_entry_time": pd.Timestamp(bench_entry_time).isoformat() if not pd.isna(bench_entry_time) else "",
                    "benchmark_exit_time": pd.Timestamp(bench_exit_time).isoformat() if not pd.isna(bench_exit_time) else "",
                    "benchmark_return": float(bench) if math.isfinite(bench) else np.nan,
                    "directional_alpha": float(alpha) if math.isfinite(alpha) else np.nan,
                    "directional_hit": directional_hit,
                }
            )
    return pd.DataFrame(rows)


def build_intraday_replay_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    frame = returns.copy()
    frame["horizon_minutes"] = pd.to_numeric(frame["horizon_minutes"], errors="coerce").astype("Int64")
    frame["directional_hit"] = frame["directional_hit"].astype(bool)
    frame["directional_alpha"] = pd.to_numeric(frame["directional_alpha"], errors="coerce")
    frame["fwd_return"] = pd.to_numeric(frame["fwd_return"], errors="coerce")
    rows: list[dict[str, object]] = []
    for (side, horizon), part in frame.groupby(["side", "horizon_minutes"], sort=True):
        quality = part[part["data_quality_pass"].astype(bool)] if "data_quality_pass" in part.columns else part
        sample = quality if not quality.empty else part
        rows.append(
            {
                "side": side,
                "horizon_minutes": int(horizon),
                "observation_count": int(len(sample)),
                "quality_observation_count": int(len(quality)),
                "signal_day_count": int(sample["signal_date"].nunique()),
                "cutoff_count": int(sample["cutoff_time"].nunique()),
                "avg_return": float(sample["fwd_return"].mean()),
                "avg_alpha": float(sample["directional_alpha"].mean()) if "directional_alpha" in sample else np.nan,
                "hit_rate": float(sample["directional_hit"].mean()) if not sample.empty else 0.0,
                "max_symbol_sample_share": float(sample["symbol"].value_counts(normalize=True).iloc[0]) if not sample.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["side", "horizon_minutes"]).reset_index(drop=True)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    result = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise ValueError("expected at least one horizon")
    return tuple(result)


def _sync_outputs(paths: list[Path], *, base_dir: Path, nas_host: str, nas_dir: str) -> list[dict[str, str]]:
    results = []
    if not nas_host or not nas_dir:
        return results
    for path in paths:
        status, remote_path, error = _copy_to_nas(path, base_dir, nas_host, nas_dir)
        results.append({"local_path": str(path), "nas_path": remote_path, "status": status, "error": error})
    return results


def write_intraday_replay_outputs(
    base_dir: str | Path,
    *,
    date: str,
    events: pd.DataFrame,
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    status: dict[str, object],
) -> dict[str, Path]:
    output_dir = Path(base_dir).expanduser() / "validation" / "intraday_replay" / f"date={date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "events": output_dir / "intraday_replay_events.parquet",
        "returns": output_dir / "intraday_replay_returns.parquet",
        "metrics": output_dir / "intraday_replay_metrics.csv",
        "status": output_dir / "status.json",
    }
    events.to_parquet(outputs["events"], index=False)
    returns.to_parquet(outputs["returns"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    outputs["status"].write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_status = Path(base_dir).expanduser() / "validation" / "intraday_replay" / "latest_status.json"
    latest_status.parent.mkdir(parents=True, exist_ok=True)
    latest_status.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs["latest_status"] = latest_status
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build intraday replay calibration samples for US microstructure signals.")
    parser.add_argument("--base-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR)))
    parser.add_argument("--date", default=os.environ.get("US_MICROSTRUCTURE_DATE", _date_default()))
    parser.add_argument("--symbols", default=os.environ.get("US_MICROSTRUCTURE_REPLAY_SYMBOLS", ""))
    parser.add_argument("--horizons-minutes", default=os.environ.get("US_MICROSTRUCTURE_REPLAY_HORIZONS_MINUTES", "30,60"))
    parser.add_argument("--cutoff-interval-minutes", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_REPLAY_CUTOFF_INTERVAL_MINUTES", "30")))
    parser.add_argument("--min-elapsed-minutes", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_REPLAY_MIN_ELAPSED_MINUTES", "60")))
    parser.add_argument("--entry-lag-minutes", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_REPLAY_ENTRY_LAG_MINUTES", "1")))
    parser.add_argument("--min-event-score", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_REPLAY_MIN_EVENT_SCORE", "65")))
    parser.add_argument("--benchmark", default=os.environ.get("US_MICROSTRUCTURE_BENCHMARK", "US.SPY"))
    parser.add_argument("--book-levels", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_BOOK_LEVELS", "5")))
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--nas-host", default=os.environ.get("US_MICROSTRUCTURE_NAS_HOST", ""))
    parser.add_argument("--nas-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_DIR", DEFAULT_NAS_DIR))
    parser.add_argument("--no-nas-sync", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir).expanduser()
    symbols = _parse_symbols(args.symbols)
    horizons = _parse_int_tuple(args.horizons_minutes)
    features = load_or_build_features(
        base_dir,
        date=args.date,
        symbols=symbols,
        book_levels=args.book_levels,
        rebuild=bool(args.rebuild_features),
    )
    events, returns, metrics = build_intraday_replay(
        features,
        date=args.date,
        horizons_minutes=horizons,
        cutoff_interval_minutes=args.cutoff_interval_minutes,
        min_elapsed_minutes=args.min_elapsed_minutes,
        min_event_score=args.min_event_score,
        entry_lag_minutes=args.entry_lag_minutes,
        benchmark=args.benchmark,
    )
    status: dict[str, object] = {
        "date": args.date,
        "generated_at": _utc_now_iso(),
        "base_dir": str(base_dir),
        "symbol_count": int(events["symbol"].nunique()) if not events.empty and "symbol" in events else 0,
        "cutoff_count": int(events["cutoff_time"].nunique()) if not events.empty and "cutoff_time" in events else 0,
        "event_count": int(len(events)),
        "quality_event_count": int(events["data_quality_pass"].astype(bool).sum()) if not events.empty and "data_quality_pass" in events else 0,
        "return_count": int(len(returns)),
        "quality_return_count": int(returns["data_quality_pass"].astype(bool).sum()) if not returns.empty and "data_quality_pass" in returns else 0,
        "metric_count": int(len(metrics)),
        "horizons_minutes": list(horizons),
        "cutoff_interval_minutes": int(args.cutoff_interval_minutes),
        "min_elapsed_minutes": int(args.min_elapsed_minutes),
        "min_event_score": float(args.min_event_score),
        "benchmark": str(args.benchmark),
    }
    outputs = write_intraday_replay_outputs(
        base_dir,
        date=args.date,
        events=events,
        returns=returns,
        metrics=metrics,
        status=status,
    )
    if not args.no_nas_sync:
        sync_results = _sync_outputs(list(outputs.values()), base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)
        status["nas_sync"] = sync_results
        outputs["status"].write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs["latest_status"].write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.nas_host and args.nas_dir:
            _sync_outputs([outputs["status"], outputs["latest_status"]], base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)
    print(
        "Intraday replay: events={events} quality_events={quality_events} returns={returns} cutoffs={cutoffs}".format(
            events=status["event_count"],
            quality_events=status["quality_event_count"],
            returns=status["return_count"],
            cutoffs=status["cutoff_count"],
        )
    )
    print(f"Wrote intraday replay status: {outputs['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
