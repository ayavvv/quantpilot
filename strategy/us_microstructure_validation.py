"""Forward validation for US microstructure major-flow signals.

There is no historical full-tape archive yet, so validation must be built from
signals generated after live collection starts. This module maintains that
forward ledger and promotes a side only when the collected outcomes pass the
configured sample-size, alpha, hit-rate, recency, Wilson-bound, and concentration
checks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from converter.incremental import QlibBinReader
from strategy.us_microstructure_features import normalize_us_symbol


@dataclass(frozen=True)
class ForwardValidationConfig:
    horizons: tuple[int, ...] = (1, 3, 5)
    benchmark: str = "US.SPY"
    entry_lag_days: int = 1
    min_event_score: float = 70.0
    promotion_horizon: int = 5
    min_signal_days_per_side: int = 20
    min_observations_per_side: int = 100
    min_alpha: float = 0.0075
    min_hit_rate: float = 0.58
    min_recent_hit_rate: float = 0.55
    recent_signal_days: int = 20
    min_wilson_lower: float = 0.50
    max_symbol_sample_share: float = 0.20


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _date_from_signal_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("date="):
            return part.split("=", 1)[1][:10]
    return ""


def discover_signal_files(base_dir: str | Path, *, start_date: str = "", end_date: str = "") -> list[Path]:
    root = Path(base_dir).expanduser() / "signals"
    if not root.exists():
        return []
    files = sorted(root.glob("date=*/us_major_flow_signals.csv"))
    result = []
    for path in files:
        signal_date = _date_from_signal_path(path)
        if start_date and signal_date < start_date:
            continue
        if end_date and signal_date > end_date:
            continue
        result.append(path)
    return result


def load_signal_events(
    signal_files: Iterable[str | Path],
    *,
    min_event_score: float = 70.0,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_path in signal_files:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        signal_date = _date_from_signal_path(path)
        if "signal_date" not in df.columns:
            df["signal_date"] = signal_date
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    events = pd.concat(frames, ignore_index=True)
    if "symbol" not in events.columns:
        return pd.DataFrame()
    events["symbol"] = events["symbol"].map(normalize_us_symbol)
    events["signal_date"] = events["signal_date"].astype(str).str[:10]
    if "side" in events.columns:
        events["side"] = events["side"].astype(str).str.lower()
    else:
        events["side"] = ""
    events["side_score"] = pd.to_numeric(events.get("side_score", np.nan), errors="coerce")
    if "confidence" in events.columns:
        events["confidence"] = events["confidence"].astype(str).str.lower()
    else:
        events["confidence"] = ""
    if "data_quality_pass" in events.columns:
        events["data_quality_pass"] = events["data_quality_pass"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    else:
        events["data_quality_pass"] = False
    events = events[
        (events["symbol"] != "")
        & (events["signal_date"].str.len() == 10)
        & (events["side"].isin({"accumulation", "distribution"}))
        & (events["side_score"] >= float(min_event_score))
        & (events["confidence"].isin({"watch", "high"}))
        & (events["data_quality_pass"])
    ].copy()
    if events.empty:
        return pd.DataFrame()
    events["event_id"] = events["signal_date"] + "|" + events["symbol"] + "|" + events["side"]
    keep_order = [
        "event_id",
        "signal_date",
        "symbol",
        "side",
        "side_score",
        "rank",
        "confidence",
        "stage",
        "reason",
        "data_quality_pass",
    ]
    for column in keep_order:
        if column not in events.columns:
            events[column] = np.nan
    events = events.sort_values(["signal_date", "side_score"], ascending=[True, False])
    events = events.drop_duplicates("event_id", keep="last")
    return events.reset_index(drop=True)


def _code_to_fname(code: str) -> str:
    replace_names = ["CON", "PRN", "AUX", "NUL"] + [f"COM{i}" for i in range(10)] + [f"LPT{i}" for i in range(10)]
    if str(code).upper() in replace_names:
        return "_qlib_" + str(code)
    return str(code)


def _read_close_from_qlib(reader: QlibBinReader, qlib_dir: Path, code: str) -> pd.Series:
    feat_dir = qlib_dir / "features" / _code_to_fname(code).lower()
    bin_path = feat_dir / "close.day.bin"
    if not bin_path.exists():
        return pd.Series(dtype="float64")
    data = np.fromfile(str(bin_path), dtype="<f4")
    if len(data) == 0:
        return pd.Series(dtype="float64")
    start_idx = int(data[0])
    values = data[1:]
    dates = reader.calendar[start_idx : start_idx + len(values)]
    return pd.Series(values.astype("float64"), index=pd.Index(dates, name="date"), name=code).dropna()


def load_price_history_from_qlib(qlib_dir: str | Path, symbols: Iterable[str]) -> dict[str, pd.Series]:
    qlib_path = Path(qlib_dir).expanduser()
    if not qlib_path.exists():
        return {}
    reader = QlibBinReader(qlib_path)
    prices: dict[str, pd.Series] = {}
    for raw_symbol in sorted({normalize_us_symbol(symbol) for symbol in symbols if normalize_us_symbol(symbol)}):
        close = _read_close_from_qlib(reader, qlib_path, raw_symbol)
        if not close.empty:
            prices[raw_symbol] = close
    return prices


def load_price_history_from_csv(path: str | Path) -> dict[str, pd.Series]:
    target = Path(path).expanduser()
    if not target.exists():
        return {}
    df = pd.read_csv(target)
    if df.empty:
        return {}
    if "symbol" not in df.columns and "code" in df.columns:
        df["symbol"] = df["code"]
    required = {"date", "symbol", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"price CSV missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["date"] = df["date"].astype(str).str[:10]
    df["symbol"] = df["symbol"].map(normalize_us_symbol)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    result: dict[str, pd.Series] = {}
    for symbol, part in df.dropna(subset=["close"]).groupby("symbol", sort=True):
        series = part.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["close"]
        result[str(symbol)] = series.astype("float64")
    return result


def merge_price_history(*sources: dict[str, pd.Series]) -> dict[str, pd.Series]:
    merged: dict[str, pd.Series] = {}
    for source in sources:
        for symbol, series in source.items():
            normalized = normalize_us_symbol(symbol)
            if not normalized or series.empty:
                continue
            existing = merged.get(normalized)
            if existing is None or existing.empty:
                merged[normalized] = series.sort_index()
            else:
                combined = pd.concat([existing, series]).sort_index()
                merged[normalized] = combined[~combined.index.duplicated(keep="last")]
    return merged


def _forward_return(close: pd.Series, signal_date: str, horizon: int, entry_lag_days: int) -> float:
    if close.empty:
        return np.nan
    valid = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if signal_date not in valid.index:
        return np.nan
    signal_idx = int(valid.index.get_loc(signal_date))
    entry_idx = signal_idx + int(entry_lag_days)
    exit_idx = entry_idx + int(horizon)
    if entry_idx < 0 or exit_idx >= len(valid):
        return np.nan
    entry = float(valid.iloc[entry_idx])
    exit_ = float(valid.iloc[exit_idx])
    if not math.isfinite(entry) or not math.isfinite(exit_) or entry <= 0:
        return np.nan
    return exit_ / entry - 1.0


def compute_forward_returns(
    events: pd.DataFrame,
    prices: dict[str, pd.Series],
    *,
    config: ForwardValidationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ForwardValidationConfig()
    if events.empty:
        return pd.DataFrame()
    benchmark = normalize_us_symbol(cfg.benchmark)
    benchmark_close = prices.get(benchmark, pd.Series(dtype="float64"))
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        symbol = normalize_us_symbol(event.get("symbol"))
        close = prices.get(symbol, pd.Series(dtype="float64"))
        base = event.to_dict()
        for horizon in cfg.horizons:
            fwd = _forward_return(close, str(event.get("signal_date"))[:10], int(horizon), cfg.entry_lag_days)
            bench = _forward_return(benchmark_close, str(event.get("signal_date"))[:10], int(horizon), cfg.entry_lag_days)
            if not math.isfinite(fwd):
                continue
            side = str(event.get("side") or "").lower()
            directional_alpha = fwd - bench if side == "accumulation" and math.isfinite(bench) else np.nan
            if side == "distribution" and math.isfinite(bench):
                directional_alpha = bench - fwd
            directional_hit = bool(fwd > 0) if side == "accumulation" else bool(fwd < 0)
            out = dict(base)
            out.update(
                {
                    "horizon": int(horizon),
                    "fwd_return": float(fwd),
                    "benchmark": benchmark,
                    "benchmark_return": float(bench) if math.isfinite(bench) else np.nan,
                    "directional_alpha": float(directional_alpha) if math.isfinite(directional_alpha) else np.nan,
                    "directional_hit": directional_hit,
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def _wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return (centre - margin) / denom


def build_rule_metrics(
    forward_returns: pd.DataFrame,
    *,
    config: ForwardValidationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ForwardValidationConfig()
    if forward_returns.empty:
        return pd.DataFrame()

    rows = []
    prepared = forward_returns.copy()
    prepared["signal_date"] = prepared["signal_date"].astype(str).str[:10]
    prepared["side"] = prepared["side"].astype(str).str.lower()
    prepared["horizon"] = pd.to_numeric(prepared["horizon"], errors="coerce").astype("Int64")
    prepared["directional_hit"] = prepared["directional_hit"].astype(bool)
    prepared["directional_alpha"] = pd.to_numeric(prepared["directional_alpha"], errors="coerce")
    prepared["fwd_return"] = pd.to_numeric(prepared["fwd_return"], errors="coerce")

    for (side, horizon), part in prepared.groupby(["side", "horizon"], sort=True):
        part = part.dropna(subset=["fwd_return"])
        if part.empty:
            continue
        wins = int(part["directional_hit"].sum())
        observations = int(len(part))
        signal_days = int(part["signal_date"].nunique())
        recent_dates = sorted(part["signal_date"].unique())[-max(1, int(cfg.recent_signal_days)) :]
        recent = part[part["signal_date"].isin(recent_dates)]
        symbol_share = float(part["symbol"].value_counts(normalize=True).iloc[0]) if observations else 0.0
        rows.append(
            {
                "side": side,
                "horizon": int(horizon),
                "observation_count": observations,
                "signal_day_count": signal_days,
                "avg_return": float(part["fwd_return"].mean()),
                "avg_benchmark_return": float(part["benchmark_return"].mean()) if "benchmark_return" in part else np.nan,
                "avg_alpha": float(part["directional_alpha"].mean()),
                "hit_rate": wins / observations if observations else 0.0,
                "recent_signal_day_count": int(len(recent_dates)),
                "recent_hit_rate": float(recent["directional_hit"].mean()) if not recent.empty else 0.0,
                "wilson_lower": _wilson_lower_bound(wins, observations),
                "max_symbol_sample_share": symbol_share,
            }
        )
    return pd.DataFrame(rows).sort_values(["side", "horizon"]).reset_index(drop=True)


def build_active_gate(
    metrics: pd.DataFrame,
    *,
    config: ForwardValidationConfig | None = None,
) -> dict[str, object]:
    cfg = config or ForwardValidationConfig()
    criteria = asdict(cfg)
    validated_sides = {"accumulation": False, "distribution": False}
    side_reasons: dict[str, str] = {}
    side_metrics: dict[str, object] = {}

    for side in validated_sides:
        row = pd.DataFrame()
        if not metrics.empty:
            row = metrics[(metrics["side"] == side) & (metrics["horizon"] == cfg.promotion_horizon)]
        if row.empty:
            side_reasons[side] = f"missing {cfg.promotion_horizon}d validation metrics"
            side_metrics[side] = {}
            continue
        record = row.iloc[-1].to_dict()
        side_metrics[side] = {key: (_safe_float(value, 0.0) if not isinstance(value, str) else value) for key, value in record.items()}
        checks = {
            "signal_days": int(record.get("signal_day_count") or 0) >= cfg.min_signal_days_per_side,
            "observations": int(record.get("observation_count") or 0) >= cfg.min_observations_per_side,
            "alpha": _safe_float(record.get("avg_alpha"), 0.0) >= cfg.min_alpha,
            "hit_rate": _safe_float(record.get("hit_rate"), 0.0) >= cfg.min_hit_rate,
            "recent_hit_rate": _safe_float(record.get("recent_hit_rate"), 0.0) >= cfg.min_recent_hit_rate,
            "wilson_lower": _safe_float(record.get("wilson_lower"), 0.0) > cfg.min_wilson_lower,
            "concentration": _safe_float(record.get("max_symbol_sample_share"), 1.0) <= cfg.max_symbol_sample_share,
        }
        validated_sides[side] = all(checks.values())
        failed = [name for name, passed in checks.items() if not passed]
        side_reasons[side] = "passed" if not failed else "failed: " + ", ".join(failed)

    validated = any(validated_sides.values())
    state = "validated" if validated else "warmup"
    return {
        "state": state,
        "validated": validated,
        "validated_sides": validated_sides,
        "reason": "at least one side passed promotion gates" if validated else "forward validation sample not promoted yet",
        "side_reasons": side_reasons,
        "side_metrics": side_metrics,
        "criteria": criteria,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }


def write_validation_outputs(
    base_dir: str | Path,
    *,
    events: pd.DataFrame,
    forward_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    gate: dict[str, object],
) -> dict[str, Path]:
    validation_dir = Path(base_dir).expanduser() / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "signal_events": validation_dir / "signal_events.parquet",
        "forward_returns": validation_dir / "forward_returns.parquet",
        "rule_metrics_csv": validation_dir / "rule_metrics.csv",
        "active_gate": validation_dir / "active_gate.json",
    }
    events.to_parquet(outputs["signal_events"], index=False)
    forward_returns.to_parquet(outputs["forward_returns"], index=False)
    metrics.to_csv(outputs["rule_metrics_csv"], index=False)
    outputs["active_gate"].write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs
