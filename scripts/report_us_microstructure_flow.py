"""Build a daily US microstructure major-flow report.

The report is deliberately validation-aware. Without a promoted validation gate
it writes a warmup report and diagnostic/watch candidates only; it will not mark
signals as high-confidence.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.collect_us_microstructure import _copy_to_nas
from scripts.us_microstructure_readiness import check_manifest
from strategy.us_microstructure_confidence import build_confidence_gap
from strategy.us_microstructure_features import (
    MicrostructureFeatureConfig,
    compute_microstructure_features,
    normalize_us_symbols,
    read_microstructure_inputs,
    write_feature_table,
)
from strategy.us_microstructure_signals import (
    MicrostructureSignalConfig,
    load_validation_gate,
    score_microstructure_signals,
)


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
DEFAULT_BASE_DIR = Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DATA_DIR / "us_microstructure")))
DEFAULT_NAS_DIR = "/volume1/docker/quantpilot/us_microstructure"
US_EASTERN = ZoneInfo("America/New_York")


def _parse_symbols(value: str) -> list[str]:
    return normalize_us_symbols(item for item in str(value or "").split(",") if item.strip())


def _default_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _is_final_report(report_date: str, *, now: datetime | None = None, close_buffer_minutes: int = 10) -> bool:
    session_close = datetime.strptime(report_date, "%Y-%m-%d").replace(
        hour=16,
        minute=int(close_buffer_minutes),
        second=0,
        microsecond=0,
        tzinfo=US_EASTERN,
    )
    timestamp = now or datetime.now(tz=US_EASTERN)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=US_EASTERN)
    else:
        timestamp = timestamp.astimezone(US_EASTERN)
    return timestamp >= session_close


def _money(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    sign = "-" if val < 0 else ""
    val = abs(val)
    if val >= 1_000_000_000:
        return f"{sign}${val / 1_000_000_000:.2f}B"
    if val >= 1_000_000:
        return f"{sign}${val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"{sign}${val / 1_000:.1f}K"
    return f"{sign}${val:.0f}"


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _bps(value: object) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _score(value: object) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _count(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _raw_counts(inputs: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {kind: int(len(frame)) for kind, frame in inputs.items()}


def _coverage_summary(features: pd.DataFrame) -> dict[str, object]:
    if features.empty:
        return {
            "symbol_count": 0,
            "minute_count": 0,
            "regular_minute_count": 0,
            "trade_minutes": 0,
            "book_minutes": 0,
            "quote_minutes": 0,
            "regular_trade_minutes": 0,
            "regular_book_minutes": 0,
            "regular_quote_minutes": 0,
        }
    regular_mask = (
        features["is_regular_session"].fillna(False)
        if "is_regular_session" in features.columns
        else pd.Series(True, index=features.index)
    )
    regular = features[regular_mask]
    return {
        "symbol_count": int(features["symbol"].nunique()),
        "minute_count": int(len(features)),
        "regular_minute_count": int(len(regular)),
        "trade_minutes": int(features.get("has_trade_data", pd.Series(dtype=bool)).fillna(False).sum()),
        "book_minutes": int(features.get("has_book_data", pd.Series(dtype=bool)).fillna(False).sum()),
        "quote_minutes": int(features.get("has_quote_data", pd.Series(dtype=bool)).fillna(False).sum()),
        "regular_trade_minutes": int(regular.get("has_trade_data", pd.Series(dtype=bool)).fillna(False).sum()),
        "regular_book_minutes": int(regular.get("has_book_data", pd.Series(dtype=bool)).fillna(False).sum()),
        "regular_quote_minutes": int(regular.get("has_quote_data", pd.Series(dtype=bool)).fillna(False).sum()),
    }


def _last_numeric(part: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in part.columns:
        return default
    values = pd.to_numeric(part[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.iloc[-1])


def _median_numeric(part: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in part.columns:
        return default
    values = pd.to_numeric(part[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.median())


def _sum_numeric(part: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in part.columns:
        return default
    values = pd.to_numeric(part[column], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if values.empty:
        return default
    return float(values.sum())


def _data_quality_summary(features: pd.DataFrame, cfg: MicrostructureSignalConfig) -> dict[str, object]:
    return _data_quality_summary_with_manifest(features, cfg, manifest_quality=None)


def _data_quality_summary_with_manifest(
    features: pd.DataFrame,
    cfg: MicrostructureSignalConfig,
    *,
    manifest_quality: dict[str, object] | None,
) -> dict[str, object]:
    manifest_checked = manifest_quality is not None
    nas_upload_complete = bool(manifest_quality.get("ok")) if manifest_quality is not None else True
    manifest_status_counts = manifest_quality.get("status_counts", {}) if isinstance(manifest_quality, dict) else {}
    manifest_issues = manifest_quality.get("issues", []) if isinstance(manifest_quality, dict) else []
    if features.empty:
        return {
            "symbol_count": 0,
            "eligible_symbol_count": 0,
            "high_confidence_data_quality_ok": False,
            "nas_manifest_checked": manifest_checked,
            "nas_upload_complete": nas_upload_complete,
            "manifest_count": int(manifest_quality.get("manifest_count") or 0) if isinstance(manifest_quality, dict) else 0,
            "manifest_status_counts": manifest_status_counts if isinstance(manifest_status_counts, dict) else {},
            "manifest_issues": manifest_issues if isinstance(manifest_issues, list) else [],
            "min_required_coverage": float(cfg.min_data_coverage),
            "min_required_trade_count": int(cfg.min_trade_count),
            "min_required_dollar_volume": float(cfg.min_dollar_volume),
            "max_allowed_duplicate_sequence_rate": 0.01,
            "max_allowed_spread_bps": float(cfg.max_spread_bps),
            "symbols": [],
        }

    rows = []
    for symbol, group in features.groupby("symbol", sort=True):
        part = group.sort_values("minute")
        if "is_regular_session" in part.columns:
            regular_part = part[part["is_regular_session"].fillna(False)]
        else:
            regular_part = part
        coverage = _last_numeric(part, "coverage_ratio_regular")
        trade_coverage = _last_numeric(part, "trade_coverage_ratio_regular", coverage)
        book_coverage = _last_numeric(part, "book_coverage_ratio_regular", coverage)
        quote_coverage = _last_numeric(part, "quote_coverage_ratio_regular")
        trade_count = int(pd.to_numeric(regular_part.get("trade_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        raw_trade_count = int(_sum_numeric(regular_part, "raw_trade_count", float(trade_count)))
        duplicate_sequence_count = int(_sum_numeric(regular_part, "duplicate_sequence_count", 0.0))
        dollar_volume = float(pd.to_numeric(regular_part.get("dollar_volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        duplicate_rate = (
            duplicate_sequence_count / raw_trade_count
            if raw_trade_count > 0
            else _median_numeric(regular_part, "duplicate_sequence_rate")
        )
        spread_bps = _median_numeric(regular_part, "spread_bps", cfg.max_spread_bps)
        eligible = (
            coverage >= cfg.min_data_coverage
            and trade_coverage >= cfg.min_data_coverage
            and book_coverage >= cfg.min_data_coverage
            and trade_count >= cfg.min_trade_count
            and dollar_volume >= cfg.min_dollar_volume
            and duplicate_rate < 0.01
            and spread_bps <= cfg.max_spread_bps
        )
        rows.append(
            {
                "symbol": str(symbol),
                "eligible": bool(eligible),
                "coverage_ratio_regular": coverage,
                "trade_coverage_ratio_regular": trade_coverage,
                "book_coverage_ratio_regular": book_coverage,
                "quote_coverage_ratio_regular": quote_coverage,
                "trade_count": trade_count,
                "raw_trade_count": raw_trade_count,
                "duplicate_sequence_count": duplicate_sequence_count,
                "dollar_volume": dollar_volume,
                "duplicate_sequence_rate": duplicate_rate,
                "spread_bps": spread_bps,
            }
        )

    ratios = [float(row["coverage_ratio_regular"]) for row in rows]
    trade_ratios = [float(row["trade_coverage_ratio_regular"]) for row in rows]
    book_ratios = [float(row["book_coverage_ratio_regular"]) for row in rows]
    eligible_count = sum(1 for row in rows if row["eligible"])
    raw_trade_count = sum(int(row.get("raw_trade_count") or 0) for row in rows)
    duplicate_sequence_count = sum(int(row.get("duplicate_sequence_count") or 0) for row in rows)
    return {
        "symbol_count": len(rows),
        "eligible_symbol_count": int(eligible_count),
        "high_confidence_data_quality_ok": eligible_count > 0 and nas_upload_complete,
        "nas_manifest_checked": manifest_checked,
        "nas_upload_complete": nas_upload_complete,
        "manifest_count": int(manifest_quality.get("manifest_count") or 0) if isinstance(manifest_quality, dict) else 0,
        "manifest_status_counts": manifest_status_counts if isinstance(manifest_status_counts, dict) else {},
        "manifest_issues": manifest_issues if isinstance(manifest_issues, list) else [],
        "raw_trade_count": int(raw_trade_count),
        "duplicate_sequence_count": int(duplicate_sequence_count),
        "duplicate_sequence_rate": duplicate_sequence_count / raw_trade_count if raw_trade_count > 0 else 0.0,
        "min_required_coverage": float(cfg.min_data_coverage),
        "min_required_trade_count": int(cfg.min_trade_count),
        "min_required_dollar_volume": float(cfg.min_dollar_volume),
        "max_allowed_duplicate_sequence_rate": 0.01,
        "max_allowed_spread_bps": float(cfg.max_spread_bps),
        "min_coverage_ratio_regular": min(ratios) if ratios else 0.0,
        "median_coverage_ratio_regular": float(pd.Series(ratios).median()) if ratios else 0.0,
        "median_trade_coverage_ratio_regular": float(pd.Series(trade_ratios).median()) if trade_ratios else 0.0,
        "median_book_coverage_ratio_regular": float(pd.Series(book_ratios).median()) if book_ratios else 0.0,
        "symbols": rows,
    }


def _manifest_gate_reason(manifest_quality: dict[str, object]) -> str:
    issues = manifest_quality.get("issues", [])
    if isinstance(issues, list) and issues:
        return "; ".join(str(issue) for issue in issues)
    return "NAS raw upload manifest gate did not pass"


def _gate_for_report(gate: dict[str, object], manifest_quality: dict[str, object]) -> dict[str, object]:
    if bool(manifest_quality.get("ok")) or not bool(gate.get("validated")):
        return gate
    updated = dict(gate)
    reason = f"NAS/raw manifest gate failed: {_manifest_gate_reason(manifest_quality)}"
    updated["validated"] = False
    updated["state"] = "disabled"
    updated["reason"] = reason
    validated_sides = updated.get("validated_sides")
    if isinstance(validated_sides, dict):
        updated["validated_sides"] = {side: False for side in validated_sides}
    side_reasons = updated.get("side_reasons")
    if isinstance(side_reasons, dict):
        updated["side_reasons"] = {side: reason for side in side_reasons}
    return updated


def _apply_manifest_quality_to_signals(
    signals: pd.DataFrame,
    manifest_quality: dict[str, object],
) -> pd.DataFrame:
    if signals.empty:
        return signals
    result = signals.copy()
    manifest_ok = bool(manifest_quality.get("ok"))
    result["nas_upload_complete"] = manifest_ok
    if manifest_ok:
        return result
    reason = f"NAS/raw manifest gate failed: {_manifest_gate_reason(manifest_quality)}"
    result["data_quality_pass"] = False
    result["validation_reason"] = reason
    if "confidence" in result.columns:
        result.loc[result["confidence"].astype(str).str.lower() == "high", "confidence"] = "watch"
    return result


def _apply_final_report_gate_to_signals(signals: pd.DataFrame, *, is_final_report: bool) -> pd.DataFrame:
    if bool(is_final_report) or signals.empty or "confidence" not in signals.columns:
        return signals
    result = signals.copy()
    high_mask = result["confidence"].astype(str).str.lower() == "high"
    if not bool(high_mask.any()):
        return result
    result.loc[high_mask, "confidence"] = "watch"
    if "validation_reason" in result.columns:
        result.loc[high_mask, "validation_reason"] = "not final post-close report"
    return result


def _candidate_view(signals: pd.DataFrame, *, top_n: int, min_score: float) -> pd.DataFrame:
    if signals.empty:
        return signals
    view = signals[signals["side_score"] >= float(min_score)].copy()
    if view.empty:
        view = signals.head(top_n).copy()
    return view.head(top_n)


def _validation_progress(validation_gate: dict[str, object]) -> dict[str, object]:
    criteria = validation_gate.get("criteria", {})
    if not isinstance(criteria, dict):
        criteria = {}
    side_reasons = validation_gate.get("side_reasons", {})
    if not isinstance(side_reasons, dict):
        side_reasons = {}
    side_metrics = validation_gate.get("side_metrics", {})
    if not isinstance(side_metrics, dict):
        side_metrics = {}
    validated_sides = validation_gate.get("validated_sides", {})
    if not isinstance(validated_sides, dict):
        validated_sides = {}
    default_side_validated = bool(validation_gate.get("validated")) and not validated_sides

    sides = []
    for side in ("accumulation", "distribution"):
        metrics = side_metrics.get(side, {})
        if not isinstance(metrics, dict):
            metrics = {}
        observations = _count(metrics.get("observation_count"))
        signal_days = _count(metrics.get("signal_day_count"))
        min_observations = _count(criteria.get("min_observations_per_side"))
        min_signal_days = _count(criteria.get("min_signal_days_per_side"))
        sides.append(
            {
                "side": side,
                "validated": bool(validated_sides.get(side, default_side_validated)),
                "reason": str(side_reasons.get(side) or ""),
                "observation_count": observations,
                "min_observations": min_observations,
                "observation_progress": observations / min_observations if min_observations > 0 else 0.0,
                "signal_day_count": signal_days,
                "min_signal_days": min_signal_days,
                "signal_day_progress": signal_days / min_signal_days if min_signal_days > 0 else 0.0,
                "avg_alpha": _number(metrics.get("avg_alpha"), 0.0),
                "min_alpha": _number(criteria.get("min_alpha"), 0.0),
                "hit_rate": _number(metrics.get("hit_rate"), 0.0),
                "min_hit_rate": _number(criteria.get("min_hit_rate"), 0.0),
                "recent_hit_rate": _number(metrics.get("recent_hit_rate"), 0.0),
                "min_recent_hit_rate": _number(criteria.get("min_recent_hit_rate"), 0.0),
                "wilson_lower": _number(metrics.get("wilson_lower"), 0.0),
                "min_wilson_lower": _number(criteria.get("min_wilson_lower"), 0.0),
                "max_symbol_sample_share": _number(metrics.get("max_symbol_sample_share"), 0.0),
                "max_allowed_symbol_sample_share": _number(criteria.get("max_symbol_sample_share"), 0.0),
            }
        )

    return {
        "state": str(validation_gate.get("state") or "warmup"),
        "validated": bool(validation_gate.get("validated")),
        "reason": str(validation_gate.get("reason") or ""),
        "signal_file_count": _count(validation_gate.get("signal_file_count")),
        "event_count": _count(validation_gate.get("event_count")),
        "forward_return_count": _count(validation_gate.get("forward_return_count")),
        "shadow_min_event_score": _number(validation_gate.get("shadow_min_event_score"), 65.0),
        "shadow_event_count": _count(validation_gate.get("shadow_event_count")),
        "shadow_forward_return_count": _count(validation_gate.get("shadow_forward_return_count")),
        "exploration_min_event_score": _number(validation_gate.get("exploration_min_event_score"), 50.0),
        "exploration_event_count": _count(validation_gate.get("exploration_event_count")),
        "exploration_forward_return_count": _count(validation_gate.get("exploration_forward_return_count")),
        "price_symbol_count": _count(validation_gate.get("price_symbol_count")),
        "promotion_horizon": _count(criteria.get("promotion_horizon")),
        "benchmark": str(criteria.get("benchmark") or ""),
        "sides": sides,
        "criteria": criteria,
    }


def _validation_min_event_score(validation_gate: dict[str, object]) -> float:
    criteria = validation_gate.get("criteria", {})
    if not isinstance(criteria, dict):
        return 70.0
    return _number(criteria.get("min_event_score"), 70.0)


def _truthy_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default)
    return values.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _validation_eligibility_summary(signals: pd.DataFrame, *, min_event_score: float) -> dict[str, object]:
    if signals.empty:
        return {
            "signal_count": 0,
            "min_event_score": float(min_event_score),
            "max_side_score": 0.0,
            "score_pass_count": 0,
            "near_score_count": 0,
            "watch_or_high_count": 0,
            "data_quality_pass_count": 0,
            "final_report_count": 0,
            "validation_eligible_count": 0,
            "validation_eligible_if_final_count": 0,
            "blocking_counts": {},
        }

    frame = signals.copy()
    symbols = frame.get("symbol", pd.Series("", index=frame.index)).astype(str).str.strip()
    sides = frame.get("side", pd.Series("", index=frame.index)).astype(str).str.lower()
    side_scores = pd.to_numeric(frame.get("side_score", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    confidence = frame.get("confidence", pd.Series("", index=frame.index)).astype(str).str.lower()
    final_report = _truthy_series(frame, "is_final_report", False)
    data_quality = _truthy_series(frame, "data_quality_pass", False)

    symbol_pass = symbols != ""
    side_pass = sides.isin({"accumulation", "distribution"})
    score_pass = side_scores >= float(min_event_score)
    near_score = (side_scores >= float(min_event_score) - 5.0) & (side_scores < float(min_event_score))
    confidence_pass = confidence.isin({"watch", "high"})
    eligible_if_final = symbol_pass & side_pass & score_pass & confidence_pass & data_quality
    eligible = eligible_if_final & final_report
    return {
        "signal_count": int(len(frame)),
        "min_event_score": float(min_event_score),
        "max_side_score": float(side_scores.max()) if len(side_scores) else 0.0,
        "score_pass_count": int(score_pass.sum()),
        "near_score_count": int(near_score.sum()),
        "watch_or_high_count": int(confidence_pass.sum()),
        "data_quality_pass_count": int(data_quality.sum()),
        "final_report_count": int(final_report.sum()),
        "validation_eligible_count": int(eligible.sum()),
        "validation_eligible_if_final_count": int(eligible_if_final.sum()),
        "blocking_counts": {
            "missing_symbol": int((~symbol_pass).sum()),
            "invalid_side": int((~side_pass).sum()),
            "score_below_min": int((~score_pass).sum()),
            "not_watch_or_high": int((~confidence_pass).sum()),
            "data_quality_failed": int((~data_quality).sum()),
            "not_final_report": int((~final_report).sum()),
        },
    }


def _eligibility_markdown(summary: dict[str, object]) -> str:
    blockers = summary.get("blocking_counts", {})
    if not isinstance(blockers, dict):
        blockers = {}
    lines = [
        f"- Ledger-eligible events now: `{summary.get('validation_eligible_count', 0)}`",
        f"- Ledger-eligible if this were final: `{summary.get('validation_eligible_if_final_count', 0)}`",
        f"- Score pass / near score: `{summary.get('score_pass_count', 0)}` / `{summary.get('near_score_count', 0)}`; max score `{_score(summary.get('max_side_score'))}`",
        f"- Watch-or-high / data-quality-pass / final rows: `{summary.get('watch_or_high_count', 0)}` / `{summary.get('data_quality_pass_count', 0)}` / `{summary.get('final_report_count', 0)}`",
        "- Blockers: "
        + ", ".join(f"{key}={value}" for key, value in sorted(blockers.items())),
    ]
    return "\n".join(lines) + "\n"


def _validation_markdown_table(progress: dict[str, object]) -> str:
    rows = progress.get("sides", [])
    if not isinstance(rows, list) or not rows:
        return "No validation-side rows.\n"
    header = (
        "| Side | Validated | Reason | Obs | Days | Alpha | Hit | Recent Hit | "
        "Wilson | Max Symbol |\n"
    )
    sep = "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        body.append(
            "| {side} | {validated} | {reason} | {obs}/{min_obs} | {days}/{min_days} | {alpha}/{min_alpha} | {hit}/{min_hit} | {recent}/{min_recent} | {wilson}/{min_wilson} | {max_symbol}/{max_allowed} |".format(
                side=row.get("side", ""),
                validated="yes" if row.get("validated") else "no",
                reason=str(row.get("reason") or "").replace("|", "/"),
                obs=int(row.get("observation_count") or 0),
                min_obs=int(row.get("min_observations") or 0),
                days=int(row.get("signal_day_count") or 0),
                min_days=int(row.get("min_signal_days") or 0),
                alpha=_pct(row.get("avg_alpha")),
                min_alpha=_pct(row.get("min_alpha")),
                hit=_pct(row.get("hit_rate")),
                min_hit=_pct(row.get("min_hit_rate")),
                recent=_pct(row.get("recent_hit_rate")),
                min_recent=_pct(row.get("min_recent_hit_rate")),
                wilson=_pct(row.get("wilson_lower")),
                min_wilson=_pct(row.get("min_wilson_lower")),
                max_symbol=_pct(row.get("max_symbol_sample_share")),
                max_allowed=_pct(row.get("max_allowed_symbol_sample_share")),
            )
        )
    return header + sep + "\n".join(body) + "\n"


def _markdown_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No candidates.\n"
    columns = [
        "rank",
        "symbol",
        "side",
        "side_score",
        "confidence",
        "stage",
        "dollar_volume",
        "net_active_dollar",
        "active_buy_ratio",
        "vwap_deviation_bps",
        "spread_bps",
        "reason",
    ]
    header = "| Rank | Symbol | Side | Score | Confidence | Stage | Dollar Vol | Net Active | Buy Ratio | VWAP bps | Spread bps | Reason |\n"
    sep = "|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---|\n"
    body = []
    for _, row in rows[columns].iterrows():
        body.append(
            "| {rank} | {symbol} | {side} | {score} | {confidence} | {stage} | {dollar} | {net} | {buy_ratio} | {vwap} | {spread} | {reason} |".format(
                rank=int(row["rank"]),
                symbol=row["symbol"],
                side=row["side"],
                score=_score(row["side_score"]),
                confidence=row["confidence"],
                stage=row["stage"],
                dollar=_money(row["dollar_volume"]),
                net=_money(row["net_active_dollar"]),
                buy_ratio=_pct(row["active_buy_ratio"]),
                vwap=_bps(row["vwap_deviation_bps"]),
                spread=_bps(row["spread_bps"]),
                reason=str(row["reason"]).replace("|", "/"),
            )
        )
    return header + sep + "\n".join(body) + "\n"


def _load_intraday_replay_summary(base_dir: Path, date: str) -> dict[str, object]:
    replay_dir = base_dir / "validation" / "intraday_replay" / f"date={date}"
    status_path = replay_dir / "status.json"
    metrics_path = replay_dir / "intraday_replay_metrics.csv"
    cumulative_status_path = base_dir / "validation" / "intraday_replay" / "cumulative_status.json"
    cumulative_metrics_path = base_dir / "validation" / "intraday_replay" / "cumulative_metrics.csv"
    status: dict[str, object] = {}
    cumulative_status: dict[str, object] = {}
    metrics: list[dict[str, object]] = []
    cumulative_metrics: list[dict[str, object]] = []
    issues: list[str] = []
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                status = payload
            else:
                issues.append("intraday replay status is not a JSON object")
        except Exception as exc:
            issues.append(f"intraday replay status unreadable: {exc}")
    if metrics_path.exists():
        try:
            metrics = pd.read_csv(metrics_path).to_dict("records")
        except Exception as exc:
            issues.append(f"intraday replay metrics unreadable: {exc}")
    if cumulative_status_path.exists():
        try:
            payload = json.loads(cumulative_status_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cumulative_status = payload
            else:
                issues.append("cumulative intraday replay status is not a JSON object")
        except Exception as exc:
            issues.append(f"cumulative intraday replay status unreadable: {exc}")
    if cumulative_metrics_path.exists():
        try:
            cumulative_metrics = pd.read_csv(cumulative_metrics_path).to_dict("records")
        except Exception as exc:
            issues.append(f"cumulative intraday replay metrics unreadable: {exc}")
    return {
        "exists": status_path.exists(),
        "status_path": str(status_path),
        "metrics_path": str(metrics_path),
        "cumulative_exists": cumulative_status_path.exists(),
        "cumulative_status_path": str(cumulative_status_path),
        "cumulative_metrics_path": str(cumulative_metrics_path),
        "event_count": int(status.get("event_count") or 0),
        "quality_event_count": int(status.get("quality_event_count") or 0),
        "return_count": int(status.get("return_count") or 0),
        "quality_return_count": int(status.get("quality_return_count") or 0),
        "cutoff_count": int(status.get("cutoff_count") or 0),
        "metric_count": int(status.get("metric_count") or 0),
        "horizons_minutes": status.get("horizons_minutes", []),
        "metrics": metrics,
        "cumulative_date_count": int(cumulative_status.get("date_count") or 0),
        "cumulative_first_date": str(cumulative_status.get("first_date") or ""),
        "cumulative_last_date": str(cumulative_status.get("last_date") or ""),
        "cumulative_event_count": int(cumulative_status.get("event_count") or 0),
        "cumulative_quality_event_count": int(cumulative_status.get("quality_event_count") or 0),
        "cumulative_return_count": int(cumulative_status.get("return_count") or 0),
        "cumulative_quality_return_count": int(cumulative_status.get("quality_return_count") or 0),
        "cumulative_metric_count": int(cumulative_status.get("metric_count") or 0),
        "cumulative_horizons_minutes": cumulative_status.get("horizons_minutes", []),
        "cumulative_metrics": cumulative_metrics,
        "issues": issues,
    }


def _intraday_metric_table_markdown(metrics: object) -> list[str]:
    if not isinstance(metrics, list) or not metrics:
        return ["No intraday replay metric rows."]
    lines = [
        "| Side | Horizon min | Obs | Quality Obs | Hit | Avg Alpha | Max Symbol |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {side} | {horizon} | {obs} | {quality} | {hit} | {alpha} | {symbol_share} |".format(
                side=str(row.get("side") or ""),
                horizon=int(row.get("horizon_minutes") or 0),
                obs=int(row.get("observation_count") or 0),
                quality=int(row.get("quality_observation_count") or 0),
                hit=_pct(row.get("hit_rate")),
                alpha=_pct(row.get("avg_alpha")),
                symbol_share=_pct(row.get("max_symbol_sample_share")),
            )
        )
    return lines


def _intraday_replay_markdown(summary: dict[str, object]) -> str:
    lines = [
        f"- Today replay available: `{bool(summary.get('exists'))}`",
        f"- Today cutoffs / events / returns: `{summary.get('cutoff_count', 0)}` / `{summary.get('quality_event_count', 0)}` quality of `{summary.get('event_count', 0)}` / `{summary.get('quality_return_count', 0)}` quality of `{summary.get('return_count', 0)}`",
        f"- Today horizons: `{summary.get('horizons_minutes') or []}`",
        f"- Cumulative dates / events / returns: `{summary.get('cumulative_date_count', 0)}` / `{summary.get('cumulative_quality_event_count', 0)}` quality of `{summary.get('cumulative_event_count', 0)}` / `{summary.get('cumulative_quality_return_count', 0)}` quality of `{summary.get('cumulative_return_count', 0)}`",
        f"- Cumulative window: `{summary.get('cumulative_first_date') or 'n/a'}` to `{summary.get('cumulative_last_date') or 'n/a'}`; horizons `{summary.get('cumulative_horizons_minutes') or []}`",
    ]
    issues = summary.get("issues", [])
    if isinstance(issues, list) and issues:
        lines.append("- Issues: " + "; ".join(str(item) for item in issues))
    lines.extend(["", "Today metrics:"])
    lines.extend(_intraday_metric_table_markdown(summary.get("metrics", [])))
    lines.extend(["", "Cumulative metrics:"])
    lines.extend(_intraday_metric_table_markdown(summary.get("cumulative_metrics", [])))
    return "\n".join(lines) + "\n"


def _confidence_gap_markdown(summary: dict[str, object]) -> str:
    requirements = summary.get("requirements", {})
    if not isinstance(requirements, dict):
        requirements = {}
    blockers = summary.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = []
    replay = summary.get("cumulative_intraday_replay", {})
    if not isinstance(replay, dict):
        replay = {}
    lines = [
        f"- High-confidence ready: `{bool(summary.get('ready'))}`",
        "- Requirements: "
        + ", ".join(f"{key}={bool(value)}" for key, value in sorted(requirements.items())),
        f"- Official validation samples: `{summary.get('official_event_count', 0)}` events, `{summary.get('official_forward_return_count', 0)}` forward-return rows",
        f"- Shadow samples: `{summary.get('shadow_event_count', 0)}` events, `{summary.get('shadow_forward_return_count', 0)}` forward-return rows",
        f"- Exploration samples: `{summary.get('exploration_event_count', 0)}` events, `{summary.get('exploration_forward_return_count', 0)}` forward-return rows",
        f"- Current report eligible samples: `{summary.get('validation_eligible_count', 0)}` now; `{summary.get('validation_eligible_if_final_count', 0)}` if final",
        f"- Cumulative intraday replay: `{replay.get('date_count', 0)}` dates, `{replay.get('quality_event_count', 0)}` quality events, `{replay.get('quality_return_count', 0)}` quality returns",
    ]
    if blockers:
        lines.append("- Blockers: " + "; ".join(str(item) for item in blockers))
    rows = summary.get("side_gaps", [])
    if not isinstance(rows, list) or not rows:
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            "| Side | Validated | Obs Need | Days Need | Alpha Gap | Hit Gap | Recent Hit Gap | Wilson Gap | Concentration Excess |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {side} | {validated} | {obs} | {days} | {alpha} | {hit} | {recent} | {wilson} | {concentration} |".format(
                side=str(row.get("side") or ""),
                validated="yes" if row.get("validated") else "no",
                obs=int(row.get("observations_needed") or 0),
                days=int(row.get("signal_days_needed") or 0),
                alpha=_pct(row.get("alpha_gap")),
                hit=_pct(row.get("hit_rate_gap")),
                recent=_pct(row.get("recent_hit_rate_gap")),
                wilson=_pct(row.get("wilson_gap")),
                concentration=_pct(row.get("concentration_excess")),
            )
        )
    return "\n".join(lines) + "\n"


def _quality_markdown_table(data_quality: dict[str, object]) -> str:
    rows = data_quality.get("symbols", [])
    if not isinstance(rows, list) or not rows:
        return "No data-quality rows.\n"
    header = (
        "| Symbol | Eligible | Coverage | Trade Cov | Book Cov | Quote Cov | "
        "Trades | Raw Trades | Dup Rows | Dollar Vol | Dup Seq | Spread bps |\n"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        body.append(
            "| {symbol} | {eligible} | {coverage} | {trade_cov} | {book_cov} | {quote_cov} | {trades} | {raw_trades} | {dup_rows} | {dollar} | {dup} | {spread} |".format(
                symbol=row.get("symbol", ""),
                eligible="yes" if row.get("eligible") else "no",
                coverage=_pct(row.get("coverage_ratio_regular")),
                trade_cov=_pct(row.get("trade_coverage_ratio_regular")),
                book_cov=_pct(row.get("book_coverage_ratio_regular")),
                quote_cov=_pct(row.get("quote_coverage_ratio_regular")),
                trades=int(row.get("trade_count") or 0),
                raw_trades=int(row.get("raw_trade_count") or 0),
                dup_rows=int(row.get("duplicate_sequence_count") or 0),
                dollar=_money(row.get("dollar_volume")),
                dup=_pct(row.get("duplicate_sequence_rate")),
                spread=_bps(row.get("spread_bps")),
            )
        )
    return header + sep + "\n".join(body) + "\n"


def render_markdown_report(
    *,
    date: str,
    signals: pd.DataFrame,
    features: pd.DataFrame,
    raw_counts: dict[str, int],
    validation_gate: dict,
    data_quality: dict[str, object],
    intraday_replay: dict[str, object],
    confidence_gap: dict[str, object],
    top_n: int,
    min_score: float,
) -> str:
    view = _candidate_view(signals, top_n=top_n, min_score=min_score)
    coverage = _coverage_summary(features)
    validation_progress = _validation_progress(validation_gate)
    eligibility = _validation_eligibility_summary(
        signals,
        min_event_score=_validation_min_event_score(validation_gate),
    )
    high_count = int((signals.get("confidence", pd.Series(dtype=str)) == "high").sum()) if not signals.empty else 0
    state = str(validation_gate.get("state") or "warmup")
    lines = [
        f"# US Microstructure Flow Report - {date}",
        "",
        f"State: `{state}`",
        f"High-confidence candidates: `{high_count}`",
        "",
        "This report uses Futu OpenD trade prints, order-book snapshots, and quotes. "
        "It does not claim account-level institutional identity.",
        "",
        "## Validation",
        "",
        f"- Gate validated: `{bool(validation_gate.get('validated'))}`",
        f"- Gate reason: {validation_gate.get('reason', '')}",
        f"- Validation samples: `{validation_progress.get('event_count', 0)}` events, `{validation_progress.get('forward_return_count', 0)}` forward-return rows",
        f"- Shadow calibration samples: `{validation_progress.get('shadow_event_count', 0)}` events, `{validation_progress.get('shadow_forward_return_count', 0)}` forward-return rows; min score `{_score(validation_progress.get('shadow_min_event_score'))}`",
        f"- Exploration calibration samples: `{validation_progress.get('exploration_event_count', 0)}` events, `{validation_progress.get('exploration_forward_return_count', 0)}` forward-return rows; min score `{_score(validation_progress.get('exploration_min_event_score'))}`",
        f"- Promotion horizon: `{validation_progress.get('promotion_horizon', 0)}d`; benchmark: `{validation_progress.get('benchmark') or 'n/a'}`",
        f"- Symbols eligible for high-confidence reporting: `{data_quality.get('eligible_symbol_count', 0)}` / `{data_quality.get('symbol_count', 0)}`",
        f"- NAS raw uploads complete: `{bool(data_quality.get('nas_upload_complete'))}`; manifest rows: `{data_quality.get('manifest_count', 0)}`",
        f"- Median trade/book coverage: `{_pct(data_quality.get('median_trade_coverage_ratio_regular'))}` / `{_pct(data_quality.get('median_book_coverage_ratio_regular'))}`",
        f"- Duplicate sequence rows: `{data_quality.get('duplicate_sequence_count', 0)}` / `{data_quality.get('raw_trade_count', 0)}` (`{_pct(data_quality.get('duplicate_sequence_rate'))}`)",
        "",
        "## Confidence Readiness",
        "",
        _confidence_gap_markdown(confidence_gap),
        "",
        "## Validation Progress By Side",
        "",
        _validation_markdown_table(validation_progress),
        "",
        "## Validation Event Eligibility",
        "",
        _eligibility_markdown(eligibility),
        "",
        "## Intraday Replay Calibration",
        "",
        _intraday_replay_markdown(intraday_replay),
        "",
        "## Data Coverage",
        "",
        f"- Raw trade rows: `{raw_counts.get('trades', 0)}`",
        f"- Raw order-book rows: `{raw_counts.get('order_book', 0)}`",
        f"- Raw quote rows: `{raw_counts.get('quotes', 0)}`",
        f"- Symbols with features: `{coverage['symbol_count']}`",
        f"- Feature minutes: `{coverage['minute_count']}`",
        f"- Regular feature minutes: `{coverage['regular_minute_count']}`",
        f"- Regular trade/book/quote minutes: `{coverage['regular_trade_minutes']}` / `{coverage['regular_book_minutes']}` / `{coverage['regular_quote_minutes']}`",
        "",
        "## Candidates",
        "",
        _markdown_table(view),
        "",
        "## Data Quality By Symbol",
        "",
        _quality_markdown_table(data_quality),
    ]
    return "\n".join(lines)


def _html_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "<p>No candidates.</p>"
    table_rows = []
    for _, row in rows.iterrows():
        cls = "buy" if row.get("side") == "accumulation" else "sell"
        table_rows.append(
            "<tr class='{cls}'><td>{rank}</td><td>{symbol}</td><td>{side}</td><td>{score}</td>"
            "<td>{confidence}</td><td>{stage}</td><td>{dollar}</td><td>{net}</td>"
            "<td>{buy_ratio}</td><td>{vwap}</td><td>{spread}</td><td>{reason}</td></tr>".format(
                cls=cls,
                rank=int(row.get("rank") or 0),
                symbol=html.escape(str(row.get("symbol") or "")),
                side=html.escape(str(row.get("side") or "")),
                score=_score(row.get("side_score")),
                confidence=html.escape(str(row.get("confidence") or "")),
                stage=html.escape(str(row.get("stage") or "")),
                dollar=_money(row.get("dollar_volume")),
                net=_money(row.get("net_active_dollar")),
                buy_ratio=_pct(row.get("active_buy_ratio")),
                vwap=_bps(row.get("vwap_deviation_bps")),
                spread=_bps(row.get("spread_bps")),
                reason=html.escape(str(row.get("reason") or "")),
            )
        )
    return (
        "<table><tr><th>Rank</th><th>Symbol</th><th>Side</th><th>Score</th><th>Confidence</th>"
        "<th>Stage</th><th>Dollar Vol</th><th>Net Active</th><th>Buy Ratio</th>"
        "<th>VWAP bps</th><th>Spread bps</th><th>Reason</th></tr>"
        + "\n".join(table_rows)
        + "</table>"
    )


def _quality_html_table(data_quality: dict[str, object]) -> str:
    rows = data_quality.get("symbols", [])
    if not isinstance(rows, list) or not rows:
        return "<p>No data-quality rows.</p>"
    table_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cls = "buy" if row.get("eligible") else "sell"
        table_rows.append(
            "<tr class='{cls}'><td>{symbol}</td><td>{eligible}</td><td>{coverage}</td>"
            "<td>{trade_cov}</td><td>{book_cov}</td><td>{quote_cov}</td><td>{trades}</td>"
            "<td>{raw_trades}</td><td>{dup_rows}</td><td>{dollar}</td><td>{dup}</td><td>{spread}</td></tr>".format(
                cls=cls,
                symbol=html.escape(str(row.get("symbol") or "")),
                eligible="yes" if row.get("eligible") else "no",
                coverage=_pct(row.get("coverage_ratio_regular")),
                trade_cov=_pct(row.get("trade_coverage_ratio_regular")),
                book_cov=_pct(row.get("book_coverage_ratio_regular")),
                quote_cov=_pct(row.get("quote_coverage_ratio_regular")),
                trades=int(row.get("trade_count") or 0),
                raw_trades=int(row.get("raw_trade_count") or 0),
                dup_rows=int(row.get("duplicate_sequence_count") or 0),
                dollar=_money(row.get("dollar_volume")),
                dup=_pct(row.get("duplicate_sequence_rate")),
                spread=_bps(row.get("spread_bps")),
            )
        )
    return (
        "<table><tr><th>Symbol</th><th>Eligible</th><th>Coverage</th><th>Trade Cov</th>"
        "<th>Book Cov</th><th>Quote Cov</th><th>Trades</th><th>Raw Trades</th><th>Dup Rows</th><th>Dollar Vol</th>"
        "<th>Dup Seq</th><th>Spread bps</th></tr>"
        + "\n".join(table_rows)
        + "</table>"
    )


def _validation_html_table(progress: dict[str, object]) -> str:
    rows = progress.get("sides", [])
    if not isinstance(rows, list) or not rows:
        return "<p>No validation-side rows.</p>"
    table_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cls = "buy" if row.get("validated") else "sell"
        table_rows.append(
            "<tr class='{cls}'><td>{side}</td><td>{validated}</td><td>{reason}</td>"
            "<td>{obs}/{min_obs}</td><td>{days}/{min_days}</td><td>{alpha}/{min_alpha}</td>"
            "<td>{hit}/{min_hit}</td><td>{recent}/{min_recent}</td><td>{wilson}/{min_wilson}</td>"
            "<td>{max_symbol}/{max_allowed}</td></tr>".format(
                cls=cls,
                side=html.escape(str(row.get("side") or "")),
                validated="yes" if row.get("validated") else "no",
                reason=html.escape(str(row.get("reason") or "")),
                obs=int(row.get("observation_count") or 0),
                min_obs=int(row.get("min_observations") or 0),
                days=int(row.get("signal_day_count") or 0),
                min_days=int(row.get("min_signal_days") or 0),
                alpha=_pct(row.get("avg_alpha")),
                min_alpha=_pct(row.get("min_alpha")),
                hit=_pct(row.get("hit_rate")),
                min_hit=_pct(row.get("min_hit_rate")),
                recent=_pct(row.get("recent_hit_rate")),
                min_recent=_pct(row.get("min_recent_hit_rate")),
                wilson=_pct(row.get("wilson_lower")),
                min_wilson=_pct(row.get("min_wilson_lower")),
                max_symbol=_pct(row.get("max_symbol_sample_share")),
                max_allowed=_pct(row.get("max_allowed_symbol_sample_share")),
            )
        )
    return (
        "<table><tr><th>Side</th><th>Validated</th><th>Reason</th><th>Obs</th><th>Days</th>"
        "<th>Alpha</th><th>Hit</th><th>Recent Hit</th><th>Wilson</th><th>Max Symbol</th></tr>"
        + "\n".join(table_rows)
        + "</table>"
    )


def _eligibility_html(summary: dict[str, object]) -> str:
    blockers = summary.get("blocking_counts", {})
    if not isinstance(blockers, dict):
        blockers = {}
    blocker_text = ", ".join(f"{html.escape(str(key))}={int(value or 0)}" for key, value in sorted(blockers.items()))
    return (
        "<div class='gate'><strong>Validation event eligibility:</strong> "
        "eligible={eligible}; eligible_if_final={eligible_if_final}; "
        "score_pass={score_pass}; near_score={near_score}; max_score={max_score}; "
        "watch_or_high={watch}; data_quality_pass={quality}; final_rows={final}; "
        "blockers={blockers}</div>"
    ).format(
        eligible=int(summary.get("validation_eligible_count") or 0),
        eligible_if_final=int(summary.get("validation_eligible_if_final_count") or 0),
        score_pass=int(summary.get("score_pass_count") or 0),
        near_score=int(summary.get("near_score_count") or 0),
        max_score=_score(summary.get("max_side_score")),
        watch=int(summary.get("watch_or_high_count") or 0),
        quality=int(summary.get("data_quality_pass_count") or 0),
        final=int(summary.get("final_report_count") or 0),
        blockers=blocker_text,
    )


def _intraday_metric_table_html(metrics: object) -> str:
    if not isinstance(metrics, list) or not metrics:
        return "<p>No intraday replay metric rows.</p>"
    rows = []
    for row in metrics:
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr><td>{side}</td><td>{horizon}</td><td>{obs}</td><td>{quality}</td>"
            "<td>{hit}</td><td>{alpha}</td><td>{symbol_share}</td></tr>".format(
                side=html.escape(str(row.get("side") or "")),
                horizon=int(row.get("horizon_minutes") or 0),
                obs=int(row.get("observation_count") or 0),
                quality=int(row.get("quality_observation_count") or 0),
                hit=_pct(row.get("hit_rate")),
                alpha=_pct(row.get("avg_alpha")),
                symbol_share=_pct(row.get("max_symbol_sample_share")),
            )
        )
    if not rows:
        return "<p>No intraday replay metric rows.</p>"
    return (
        "<table><tr><th>Side</th><th>Horizon min</th><th>Obs</th><th>Quality Obs</th>"
        "<th>Hit</th><th>Avg Alpha</th><th>Max Symbol</th></tr>"
        + "\n".join(rows)
        + "</table>"
    )


def _confidence_gap_html(summary: dict[str, object]) -> str:
    requirements = summary.get("requirements", {})
    if not isinstance(requirements, dict):
        requirements = {}
    blockers = summary.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = []
    replay = summary.get("cumulative_intraday_replay", {})
    if not isinstance(replay, dict):
        replay = {}
    req_text = ", ".join(
        f"{html.escape(str(key))}={bool(value)}" for key, value in sorted(requirements.items())
    )
    blocker_text = "; ".join(html.escape(str(item)) for item in blockers) if blockers else "none"
    rows = []
    side_gaps = summary.get("side_gaps", [])
    if isinstance(side_gaps, list):
        for row in side_gaps:
            if not isinstance(row, dict):
                continue
            rows.append(
                "<tr><td>{side}</td><td>{validated}</td><td>{obs}</td><td>{days}</td>"
                "<td>{alpha}</td><td>{hit}</td><td>{recent}</td><td>{wilson}</td>"
                "<td>{concentration}</td></tr>".format(
                    side=html.escape(str(row.get("side") or "")),
                    validated="yes" if row.get("validated") else "no",
                    obs=int(row.get("observations_needed") or 0),
                    days=int(row.get("signal_days_needed") or 0),
                    alpha=_pct(row.get("alpha_gap")),
                    hit=_pct(row.get("hit_rate_gap")),
                    recent=_pct(row.get("recent_hit_rate_gap")),
                    wilson=_pct(row.get("wilson_gap")),
                    concentration=_pct(row.get("concentration_excess")),
                )
            )
    table = "<p>No confidence gap rows.</p>"
    if rows:
        table = (
            "<table><tr><th>Side</th><th>Validated</th><th>Obs Need</th><th>Days Need</th>"
            "<th>Alpha Gap</th><th>Hit Gap</th><th>Recent Hit Gap</th><th>Wilson Gap</th>"
            "<th>Concentration Excess</th></tr>"
            + "\n".join(rows)
            + "</table>"
        )
    return (
        "<div class='gate'><strong>Confidence readiness:</strong> ready={ready}; "
        "requirements={requirements}; blockers={blockers}</div>"
        "<div class='gate'><strong>Validation sample gap:</strong> official_events={official_events}; "
        "official_forward_returns={official_returns}; shadow_events={shadow_events}; "
        "shadow_forward_returns={shadow_returns}; exploration_events={exploration_events}; "
        "exploration_forward_returns={exploration_returns}; eligible_now={eligible}; eligible_if_final={eligible_if_final}; "
        "cumulative_replay_dates={replay_dates}; cumulative_quality_events={replay_events}; "
        "cumulative_quality_returns={replay_returns}</div>{table}"
    ).format(
        ready=bool(summary.get("ready")),
        requirements=req_text,
        blockers=blocker_text,
        official_events=int(summary.get("official_event_count") or 0),
        official_returns=int(summary.get("official_forward_return_count") or 0),
        shadow_events=int(summary.get("shadow_event_count") or 0),
        shadow_returns=int(summary.get("shadow_forward_return_count") or 0),
        exploration_events=int(summary.get("exploration_event_count") or 0),
        exploration_returns=int(summary.get("exploration_forward_return_count") or 0),
        eligible=int(summary.get("validation_eligible_count") or 0),
        eligible_if_final=int(summary.get("validation_eligible_if_final_count") or 0),
        replay_dates=int(replay.get("date_count") or 0),
        replay_events=int(replay.get("quality_event_count") or 0),
        replay_returns=int(replay.get("quality_return_count") or 0),
        table=table,
    )


def _intraday_replay_html(summary: dict[str, object]) -> str:
    today_table = _intraday_metric_table_html(summary.get("metrics", []))
    cumulative_table = _intraday_metric_table_html(summary.get("cumulative_metrics", []))
    issues = summary.get("issues", [])
    issue_text = ""
    if isinstance(issues, list) and issues:
        issue_text = "; issues=" + html.escape("; ".join(str(item) for item in issues))
    if summary.get("cumulative_first_date") or summary.get("cumulative_last_date"):
        cumulative_window = "{first} to {last}".format(
            first=html.escape(str(summary.get("cumulative_first_date") or "n/a")),
            last=html.escape(str(summary.get("cumulative_last_date") or "n/a")),
        )
    else:
        cumulative_window = "n/a"
    return (
        "<div class='gate'><strong>Intraday replay today:</strong> available={exists}; "
        "cutoffs={cutoffs}; quality_events={quality_events}/{events}; "
        "quality_returns={quality_returns}/{returns}; horizons={horizons}{issues}</div>"
        "<h3>Today Metrics</h3>{today_table}"
        "<div class='gate'><strong>Intraday replay cumulative:</strong> dates={cum_dates}; "
        "window={cum_window}; quality_events={cum_quality_events}/{cum_events}; "
        "quality_returns={cum_quality_returns}/{cum_returns}; horizons={cum_horizons}</div>"
        "<h3>Cumulative Metrics</h3>{cumulative_table}"
    ).format(
        exists=bool(summary.get("exists")),
        cutoffs=int(summary.get("cutoff_count") or 0),
        quality_events=int(summary.get("quality_event_count") or 0),
        events=int(summary.get("event_count") or 0),
        quality_returns=int(summary.get("quality_return_count") or 0),
        returns=int(summary.get("return_count") or 0),
        horizons=html.escape(str(summary.get("horizons_minutes") or [])),
        issues=issue_text,
        today_table=today_table,
        cum_dates=int(summary.get("cumulative_date_count") or 0),
        cum_window=cumulative_window,
        cum_quality_events=int(summary.get("cumulative_quality_event_count") or 0),
        cum_events=int(summary.get("cumulative_event_count") or 0),
        cum_quality_returns=int(summary.get("cumulative_quality_return_count") or 0),
        cum_returns=int(summary.get("cumulative_return_count") or 0),
        cum_horizons=html.escape(str(summary.get("cumulative_horizons_minutes") or [])),
        cumulative_table=cumulative_table,
    )


def render_html_report(
    *,
    date: str,
    signals: pd.DataFrame,
    features: pd.DataFrame,
    raw_counts: dict[str, int],
    validation_gate: dict,
    data_quality: dict[str, object],
    intraday_replay: dict[str, object],
    confidence_gap: dict[str, object],
    top_n: int,
    min_score: float,
) -> str:
    view = _candidate_view(signals, top_n=top_n, min_score=min_score)
    coverage = _coverage_summary(features)
    validation_progress = _validation_progress(validation_gate)
    eligibility = _validation_eligibility_summary(
        signals,
        min_event_score=_validation_min_event_score(validation_gate),
    )
    high_count = int((signals.get("confidence", pd.Series(dtype=str)) == "high").sum()) if not signals.empty else 0
    state = html.escape(str(validation_gate.get("state") or "warmup"))
    reason = html.escape(str(validation_gate.get("reason") or ""))
    return f"""
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 980px; margin: 0 auto; padding: 24px; color: #1f2933; }}
h1 {{ border-bottom: 2px solid #263238; padding-bottom: 8px; }}
.metric {{ display: inline-block; margin: 8px 20px 8px 0; }}
.value {{ font-size: 22px; font-weight: 700; }}
.label {{ color: #667085; font-size: 12px; }}
.gate {{ border-left: 4px solid #9aa5b1; background: #f5f7fa; padding: 10px 14px; margin: 16px 0; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #d9e2ec; padding: 7px; text-align: left; font-size: 13px; }}
th {{ background: #263238; color: #fff; }}
tr.buy {{ background: #edf7ed; }}
tr.sell {{ background: #fff1f2; }}
.muted {{ color: #667085; }}
</style>
</head>
<body>
<h1>US Microstructure Flow Report - {html.escape(date)}</h1>
<div class="metric"><div class="value">{state}</div><div class="label">Report State</div></div>
<div class="metric"><div class="value">{high_count}</div><div class="label">High-confidence Candidates</div></div>
<div class="metric"><div class="value">{coverage['symbol_count']}</div><div class="label">Symbols</div></div>
<div class="metric"><div class="value">{coverage['minute_count']}</div><div class="label">Feature Minutes</div></div>
<p class="muted">Uses Futu OpenD tick prints, order-book snapshots, and quotes. It does not claim account-level institutional identity.</p>
<div class="gate"><strong>Validation gate:</strong> validated={bool(validation_gate.get('validated'))}; {reason}</div>
<div class="gate"><strong>Validation samples:</strong> events={validation_progress.get('event_count', 0)}; forward_returns={validation_progress.get('forward_return_count', 0)}; promotion_horizon={validation_progress.get('promotion_horizon', 0)}d; benchmark={html.escape(str(validation_progress.get('benchmark') or 'n/a'))}</div>
<div class="gate"><strong>Shadow calibration:</strong> events={validation_progress.get('shadow_event_count', 0)}; forward_returns={validation_progress.get('shadow_forward_return_count', 0)}; min_score={_score(validation_progress.get('shadow_min_event_score'))}</div>
<div class="gate"><strong>Exploration calibration:</strong> events={validation_progress.get('exploration_event_count', 0)}; forward_returns={validation_progress.get('exploration_forward_return_count', 0)}; min_score={_score(validation_progress.get('exploration_min_event_score'))}</div>
<div class="gate"><strong>Data quality gate:</strong> eligible_symbols={data_quality.get('eligible_symbol_count', 0)}/{data_quality.get('symbol_count', 0)}; median trade/book coverage={_pct(data_quality.get('median_trade_coverage_ratio_regular'))}/{_pct(data_quality.get('median_book_coverage_ratio_regular'))}; nas_upload_complete={bool(data_quality.get('nas_upload_complete'))}; manifest_rows={data_quality.get('manifest_count', 0)}</div>
<div class="gate"><strong>Duplicate audit:</strong> duplicate_sequence_rows={data_quality.get('duplicate_sequence_count', 0)}/{data_quality.get('raw_trade_count', 0)} ({_pct(data_quality.get('duplicate_sequence_rate'))})</div>
<h2>Confidence Readiness</h2>
{_confidence_gap_html(confidence_gap)}
<h2>Validation Progress By Side</h2>
{_validation_html_table(validation_progress)}
<h2>Validation Event Eligibility</h2>
{_eligibility_html(eligibility)}
<h2>Intraday Replay Calibration</h2>
{_intraday_replay_html(intraday_replay)}
<h2>Data Coverage</h2>
<p>Raw trades={raw_counts.get('trades', 0)}, order_book={raw_counts.get('order_book', 0)}, quotes={raw_counts.get('quotes', 0)}. Regular trade/book/quote minutes={coverage['regular_trade_minutes']} / {coverage['regular_book_minutes']} / {coverage['regular_quote_minutes']}.</p>
<h2>Candidates</h2>
{_html_table(view)}
<h2>Data Quality By Symbol</h2>
{_quality_html_table(data_quality)}
</body>
</html>
"""


def _write_outputs(
    *,
    base_dir: Path,
    date: str,
    features: pd.DataFrame,
    signals: pd.DataFrame,
    markdown: str,
    html_report: str,
    status: dict,
    write_latest: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    feature_path = write_feature_table(features, base_dir, date=date)
    outputs["features"] = feature_path

    signal_dir = base_dir / "signals" / f"date={date}"
    signal_dir.mkdir(parents=True, exist_ok=True)
    signal_csv = signal_dir / "us_major_flow_signals.csv"
    signals.to_csv(signal_csv, index=False)
    outputs["signals"] = signal_csv
    if write_latest:
        latest_csv = base_dir / "signals" / "us_major_flow_signals_latest.csv"
        latest_csv.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(latest_csv, index=False)
        outputs["signals_latest"] = latest_csv

    quality_dir = base_dir / "quality" / f"date={date}"
    quality_dir.mkdir(parents=True, exist_ok=True)
    quality_csv = quality_dir / "us_microstructure_data_quality.csv"
    quality_rows = status.get("data_quality", {}).get("symbols", []) if isinstance(status.get("data_quality"), dict) else []
    quality_frame = pd.DataFrame(quality_rows if isinstance(quality_rows, list) else [])
    quality_frame.to_csv(quality_csv, index=False)
    outputs["data_quality"] = quality_csv
    if write_latest:
        quality_latest = base_dir / "quality" / "us_microstructure_data_quality_latest.csv"
        quality_latest.parent.mkdir(parents=True, exist_ok=True)
        quality_frame.to_csv(quality_latest, index=False)
        outputs["data_quality_latest"] = quality_latest

    report_dir = base_dir / "reports" / f"date={date}"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "us_microstructure_flow_report.md"
    html_path = report_dir / "us_microstructure_flow_report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")
    outputs["markdown"] = markdown_path
    outputs["html"] = html_path
    if write_latest:
        latest_html = base_dir / "reports" / "us_microstructure_flow_report_latest.html"
        latest_md = base_dir / "reports" / "us_microstructure_flow_report_latest.md"
        latest_html.parent.mkdir(parents=True, exist_ok=True)
        latest_html.write_text(html_report, encoding="utf-8")
        latest_md.write_text(markdown, encoding="utf-8")
        outputs["html_latest"] = latest_html
        outputs["markdown_latest"] = latest_md

    status_path = report_dir / "status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs["status"] = status_path
    if write_latest:
        latest_status = base_dir / "reports" / "us_microstructure_flow_status_latest.json"
        latest_status.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs["status_latest"] = latest_status
    return outputs


def _write_status_outputs(outputs: dict[str, Path], status: dict) -> None:
    text = json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    outputs["status"].write_text(text, encoding="utf-8")
    if "status_latest" in outputs:
        outputs["status_latest"].write_text(text, encoding="utf-8")


def _sync_outputs(paths: Iterable[Path], *, base_dir: Path, nas_host: str, nas_dir: str) -> list[dict[str, str]]:
    results = []
    if not nas_host or not nas_dir:
        return results
    for path in paths:
        status, remote_path, error = _copy_to_nas(path, base_dir, nas_host, nas_dir)
        results.append({"local_path": str(path), "nas_path": remote_path, "status": status, "error": error})
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build US microstructure major-flow report.")
    parser.add_argument("--date", default=os.environ.get("US_MICROSTRUCTURE_DATE", _default_date()))
    parser.add_argument("--base-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR)))
    parser.add_argument("--symbols", default=os.environ.get("US_MICROSTRUCTURE_REPORT_SYMBOLS", ""))
    parser.add_argument("--top-n", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_REPORT_TOP_N", "20")))
    parser.add_argument("--min-score", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_REPORT_MIN_SCORE", "50")))
    parser.add_argument("--book-levels", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_BOOK_LEVELS", "5")))
    parser.add_argument("--validation-gate", default=os.environ.get("US_MICROSTRUCTURE_VALIDATION_GATE", ""))
    parser.add_argument("--nas-host", default=os.environ.get("US_MICROSTRUCTURE_NAS_HOST", ""))
    parser.add_argument("--nas-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_DIR", DEFAULT_NAS_DIR))
    parser.add_argument("--no-nas-sync", action="store_true")
    parser.add_argument("--send-email", action="store_true")
    return parser.parse_args(argv)


def _subject(signals: pd.DataFrame, gate: dict[str, object]) -> str:
    high = signals[signals["confidence"] == "high"] if not signals.empty and "confidence" in signals else pd.DataFrame()
    if bool(gate.get("validated")) and not high.empty:
        buys = int((high["side"] == "accumulation").sum())
        sells = int((high["side"] == "distribution").sum())
        return f"US Micro Flow - {buys} accumulation / {sells} distribution"
    return "US Microstructure Flow - warmup, 0 validated"


def _email_delivery_payload(
    *,
    requested: bool,
    subject: str,
    attachment_paths: Iterable[Path],
    sent: bool | None = None,
    error: str = "",
) -> dict[str, object]:
    return {
        "requested": bool(requested),
        "sent": bool(sent) if sent is not None else None,
        "subject": subject if requested else "",
        "attachment_paths": [str(path) for path in attachment_paths] if requested else [],
        "error": str(error or ""),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir).expanduser()
    symbols = _parse_symbols(args.symbols)
    validation_gate_path = args.validation_gate or str(base_dir / "validation" / "active_gate.json")
    gate = load_validation_gate(validation_gate_path)
    manifest_quality = check_manifest(base_dir, date=args.date, latest_only=False)
    report_gate = _gate_for_report(gate, manifest_quality)

    inputs = read_microstructure_inputs(base_dir, date=args.date, symbols=symbols)
    features = compute_microstructure_features(
        inputs["trades"],
        inputs["order_book"],
        inputs["quotes"],
        config=MicrostructureFeatureConfig(book_levels=args.book_levels),
    )
    signal_cfg = MicrostructureSignalConfig()
    is_final_report = _is_final_report(args.date)
    signals = score_microstructure_signals(
        features,
        config=signal_cfg,
        validation_gate=report_gate,
        include_diagnostic=True,
    )
    signals = _apply_manifest_quality_to_signals(signals, manifest_quality)
    if not signals.empty:
        signals = signals.copy()
        signals["is_final_report"] = bool(is_final_report)
    signals = _apply_final_report_gate_to_signals(signals, is_final_report=bool(is_final_report))
    raw_counts = _raw_counts(inputs)
    data_quality = _data_quality_summary_with_manifest(features, signal_cfg, manifest_quality=manifest_quality)
    intraday_replay = _load_intraday_replay_summary(base_dir, args.date)
    validation_progress = _validation_progress(report_gate)
    validation_eligibility = _validation_eligibility_summary(
        signals,
        min_event_score=_validation_min_event_score(report_gate),
    )
    email_subject = _subject(signals, report_gate)
    confidence_gap = build_confidence_gap(
        report_gate,
        data_quality=data_quality,
        validation_eligibility=validation_eligibility,
        intraday_replay=intraday_replay,
        manifest_quality=manifest_quality,
        is_final_report=bool(is_final_report),
    )
    status = {
        "date": args.date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "is_final_report": bool(is_final_report),
        "latest_alias_updated": bool(is_final_report),
        "base_dir": str(base_dir),
        "raw_counts": raw_counts,
        "coverage": _coverage_summary(features),
        "data_quality": data_quality,
        "intraday_replay": intraday_replay,
        "manifest_quality": manifest_quality,
        "signal_count": int(len(signals)),
        "high_count": int((signals.get("confidence", pd.Series(dtype=str)) == "high").sum()) if not signals.empty else 0,
        "watch_count": int((signals.get("confidence", pd.Series(dtype=str)) == "watch").sum()) if not signals.empty else 0,
        "validation_gate": report_gate,
        "validation_progress": validation_progress,
        "validation_eligibility": validation_eligibility,
        "confidence_gap": confidence_gap,
        "email_delivery": _email_delivery_payload(
            requested=bool(args.send_email),
            subject=email_subject,
            attachment_paths=[],
        ),
    }
    markdown = render_markdown_report(
        date=args.date,
        signals=signals,
        features=features,
        raw_counts=raw_counts,
        validation_gate=report_gate,
        data_quality=data_quality,
        intraday_replay=intraday_replay,
        confidence_gap=confidence_gap,
        top_n=args.top_n,
        min_score=args.min_score,
    )
    html_report = render_html_report(
        date=args.date,
        signals=signals,
        features=features,
        raw_counts=raw_counts,
        validation_gate=report_gate,
        data_quality=data_quality,
        intraday_replay=intraday_replay,
        confidence_gap=confidence_gap,
        top_n=args.top_n,
        min_score=args.min_score,
    )
    outputs = _write_outputs(
        base_dir=base_dir,
        date=args.date,
        features=features,
        signals=signals,
        markdown=markdown,
        html_report=html_report,
        status=status,
        write_latest=bool(is_final_report),
    )
    nas_results = []
    if not args.no_nas_sync:
        nas_results = _sync_outputs(outputs.values(), base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)
    if nas_results:
        status["nas_sync"] = nas_results
        _write_status_outputs(outputs, status)
        status_paths = [outputs["status"]]
        if "status_latest" in outputs:
            status_paths.append(outputs["status_latest"])
        _sync_outputs(status_paths, base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)

    if args.send_email:
        from reporter.send_report import send_email

        attachment_paths = [outputs["signals"], outputs["data_quality"], outputs["status"]]
        status["email_delivery"] = _email_delivery_payload(
            requested=True,
            subject=email_subject,
            attachment_paths=attachment_paths,
            sent=False,
        )
        _write_status_outputs(outputs, status)
        try:
            email_sent = bool(
                send_email(
                    html_report,
                    email_subject,
                    report_filename=outputs["html"].name,
                    report_dir=outputs["html"].parent,
                    attachment_paths=attachment_paths,
                )
            )
            email_error = "" if email_sent else "send_email returned false"
        except Exception as exc:
            email_sent = False
            email_error = str(exc)
        status["email_delivery"] = _email_delivery_payload(
            requested=True,
            subject=email_subject,
            attachment_paths=attachment_paths,
            sent=email_sent,
            error=email_error,
        )
        _write_status_outputs(outputs, status)
        if not args.no_nas_sync:
            status_paths = [outputs["status"]]
            if "status_latest" in outputs:
                status_paths.append(outputs["status_latest"])
            _sync_outputs(status_paths, base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)
        if not email_sent:
            print(f"Email delivery failed: {email_error}")

    print(f"Wrote features: {outputs['features']}")
    print(f"Wrote signals: {outputs['signals']}")
    print(f"Wrote data quality: {outputs['data_quality']}")
    print(f"Wrote report: {outputs['html']}")
    print(f"Updated latest aliases: {status['latest_alias_updated']}")
    print(f"State={gate.get('state')} high={status['high_count']} watch={status['watch_count']}")
    email_delivery = status.get("email_delivery", {})
    if isinstance(email_delivery, dict) and email_delivery.get("requested") and not email_delivery.get("sent"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
