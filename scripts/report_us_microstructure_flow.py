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


def _symbol_first_letter(symbol: object) -> str:
    text = str(symbol or "").strip().upper()
    if text.startswith("US."):
        text = text[3:]
    for char in text:
        if char.isalpha():
            return char
    return "0"


def _first_letter_counts(symbols: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in symbols:
        letter = _symbol_first_letter(symbol)
        counts[letter] = counts.get(letter, 0) + 1
    return dict(sorted(counts.items()))


def _dominant_letter_summary(counts: object) -> dict[str, object]:
    if not isinstance(counts, dict) or not counts:
        return {"letter": "", "count": 0, "total": 0, "share": 0.0, "biased": False}
    parsed = {str(key): _count(value) for key, value in counts.items()}
    parsed = {key: value for key, value in parsed.items() if value > 0}
    total = sum(parsed.values())
    if total <= 0:
        return {"letter": "", "count": 0, "total": 0, "share": 0.0, "biased": False}
    letter, count = max(parsed.items(), key=lambda item: item[1])
    share = count / total
    return {
        "letter": letter,
        "count": int(count),
        "total": int(total),
        "share": float(share),
        "biased": bool(total >= 50 and share >= 0.60),
    }


def _read_symbol_counts_from_csv(path: Path, *, column: str = "symbol") -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, usecols=[column])
    except Exception:
        return {}
    if frame.empty or column not in frame.columns:
        return {}
    return _first_letter_counts(frame[column].dropna().astype(str).tolist())


def _letter_summary_text(summary: dict[str, object]) -> str:
    total = int(summary.get("total") or 0)
    if total <= 0:
        return "n/a"
    return "{letter} {count}/{total}（{share}）".format(
        letter=str(summary.get("letter") or ""),
        count=int(summary.get("count") or 0),
        total=total,
        share=_pct(summary.get("share")),
    )


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
    blocker_text = ", ".join(
        f"{_blocking_count_label_cn(key)}={value}" for key, value in sorted(blockers.items())
    )
    lines = [
        f"- 当前可进入验证样本账本：`{summary.get('validation_eligible_count', 0)}`",
        f"- 如果这是最终报告，可进入验证样本账本：`{summary.get('validation_eligible_if_final_count', 0)}`",
        f"- 分数达标 / 接近达标：`{summary.get('score_pass_count', 0)}` / `{summary.get('near_score_count', 0)}`；最高分 `{_score(summary.get('max_side_score'))}`",
        f"- 观察或高置信 / 数据质量通过 / 最终报告行数：`{summary.get('watch_or_high_count', 0)}` / `{summary.get('data_quality_pass_count', 0)}` / `{summary.get('final_report_count', 0)}`",
        f"- 未进入样本的原因计数：{blocker_text}",
    ]
    return "\n".join(lines) + "\n"


def _validation_markdown_table(progress: dict[str, object]) -> str:
    rows = progress.get("sides", [])
    if not isinstance(rows, list) or not rows:
        return "没有按方向拆分的验证记录。\n"
    header = (
        "| 方向 | 已验证 | 原因 | 样本数 | 天数 | Alpha | 命中率 | 近期命中率 | "
        "Wilson 下界 | 单标的集中度 |\n"
    )
    sep = "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        body.append(
            "| {side} | {validated} | {reason} | {obs}/{min_obs} | {days}/{min_days} | {alpha}/{min_alpha} | {hit}/{min_hit} | {recent}/{min_recent} | {wilson}/{min_wilson} | {max_symbol}/{max_allowed} |".format(
                side=_side_label_cn(row.get("side")),
                validated=_yes_no_cn(row.get("validated")),
                reason=_reason_cn(row.get("reason")).replace("|", "/"),
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
        return "没有候选标的。\n"
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
    header = "| 排名 | 标的 | 方向 | 分数 | 置信度 | 阶段 | 成交额 | 净主动资金 | 主动买入占比 | VWAP 偏离(bps) | 点差(bps) | 原因 |\n"
    sep = "|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---|\n"
    body = []
    for _, row in rows[columns].iterrows():
        body.append(
            "| {rank} | {symbol} | {side} | {score} | {confidence} | {stage} | {dollar} | {net} | {buy_ratio} | {vwap} | {spread} | {reason} |".format(
                rank=int(row["rank"]),
                symbol=row["symbol"],
                side=_side_label_cn(row["side"]),
                score=_score(row["side_score"]),
                confidence=_confidence_label_cn(row["confidence"]),
                stage=_stage_label_cn(row["stage"]),
                dollar=_money(row["dollar_volume"]),
                net=_money(row["net_active_dollar"]),
                buy_ratio=_pct(row["active_buy_ratio"]),
                vwap=_bps(row["vwap_deviation_bps"]),
                spread=_bps(row["spread_bps"]),
                reason=_reason_cn(row["reason"]).replace("|", "/"),
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


def _load_coarse_universe_summary(base_dir: Path, date: str) -> dict[str, object]:
    dated_status = base_dir / "universe" / f"date={date}" / "status.json"
    latest_status = base_dir / "universe" / "us_microstructure_universe_status_latest.json"
    status_path = dated_status if dated_status.exists() else latest_status
    dated_candidates = base_dir / "universe" / f"date={date}" / "us_microstructure_candidates.csv"
    latest_candidates = base_dir / "universe" / "us_microstructure_candidates_latest.csv"
    candidates_path = dated_candidates if dated_candidates.exists() else latest_candidates
    dated_collection_status = base_dir / "universe" / f"date={date}" / "collection_status.json"
    latest_collection_status = base_dir / "universe" / "us_microstructure_collection_universe_status_latest.json"
    collection_status_path = dated_collection_status if dated_collection_status.exists() else latest_collection_status
    dated_collection = base_dir / "universe" / f"date={date}" / "us_microstructure_collection_universe.csv"
    latest_collection = base_dir / "universe" / "us_microstructure_collection_universe_latest.csv"
    collection_path = dated_collection if dated_collection.exists() else latest_collection
    payload: dict[str, object] = {}
    collection_payload: dict[str, object] = {}
    issues: list[str] = []
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload = raw
            else:
                issues.append("coarse universe status is not a JSON object")
        except Exception as exc:
            issues.append(f"coarse universe status unreadable: {exc}")
    else:
        issues.append("coarse universe status missing")
    if collection_status_path.exists():
        try:
            raw = json.loads(collection_status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                collection_payload = raw
        except Exception as exc:
            issues.append(f"collection universe status unreadable: {exc}")
    candidate_counts = payload.get("candidate_first_letter_counts")
    if not isinstance(candidate_counts, dict) or not candidate_counts:
        candidate_rows = payload.get("candidates", [])
        if isinstance(candidate_rows, list):
            candidate_counts = _first_letter_counts(
                row.get("symbol") for row in candidate_rows if isinstance(row, dict)
            )
    if not isinstance(candidate_counts, dict) or not candidate_counts:
        candidate_counts = _read_symbol_counts_from_csv(candidates_path)
    collection_counts = _read_symbol_counts_from_csv(collection_path)
    candidate_dominant = _dominant_letter_summary(candidate_counts)
    collection_dominant = _dominant_letter_summary(collection_counts)
    if bool(candidate_dominant.get("biased")):
        issues.append(
            "candidate universe first-letter concentration: "
            f"{_letter_summary_text(candidate_dominant)}"
        )
    if bool(collection_dominant.get("biased")):
        issues.append(
            "collection universe first-letter concentration: "
            f"{_letter_summary_text(collection_dominant)}"
        )
    return {
        "exists": status_path.exists(),
        "status_path": str(status_path),
        "status": str(payload.get("status") or "missing"),
        "date": str(payload.get("date") or ""),
        "generated_at": str(payload.get("generated_at") or ""),
        "target_size": _count(payload.get("target_size")),
        "universe_count": _count(payload.get("universe_count")),
        "snapshot_symbol_count": _count(payload.get("snapshot_symbol_count")),
        "daily_symbol_count": _count(payload.get("daily_symbol_count")),
        "minute_symbol_count": _count(payload.get("minute_symbol_count")),
        "candidate_count": _count(payload.get("candidate_count")),
        "core_symbol_count": _count(payload.get("core_symbol_count")),
        "core_symbol_source": str(payload.get("core_symbol_source") or ""),
        "core_symbol_fallback_used": bool(payload.get("core_symbol_fallback_used")),
        "core_watchlist_us_symbol_count": _count(payload.get("core_watchlist_us_symbol_count")),
        "candidate_core_count": _count(payload.get("candidate_core_count")),
        "collection_exists": collection_status_path.exists(),
        "collection_status_path": str(collection_status_path),
        "candidates_path": str(candidates_path),
        "collection_path": str(collection_path),
        "collection_symbol_count": _count(collection_payload.get("collection_symbol_count")),
        "collection_followup_count": _count(collection_payload.get("followup_selected_count")),
        "collection_followup_days": _count(collection_payload.get("followup_days")),
        "collection_max_total_symbols": _count(collection_payload.get("max_total_symbols")),
        "snapshot_error_count": _count(payload.get("snapshot_error_count")),
        "daily_error_count": _count(payload.get("daily_error_count")),
        "minute_error_count": _count(payload.get("minute_error_count")),
        "candidate_first_letter_counts": candidate_counts if isinstance(candidate_counts, dict) else {},
        "candidate_dominant_letter": candidate_dominant,
        "collection_first_letter_counts": collection_counts if isinstance(collection_counts, dict) else {},
        "collection_dominant_letter": collection_dominant,
        "alphabet_bias_warning": bool(candidate_dominant.get("biased")) or bool(collection_dominant.get("biased")),
        "candidates": payload.get("candidates", []) if isinstance(payload.get("candidates"), list) else [],
        "issues": issues,
    }


def _coarse_universe_markdown(summary: dict[str, object]) -> str:
    lines = [
        f"- 粗筛结果可用：`{_yes_no_cn(summary.get('exists'))}`",
        f"- 状态 / 日期：`{summary.get('status')}` / `{summary.get('date') or 'n/a'}`",
        f"- 全市场股票数 / 快照覆盖 / 日线覆盖 / 分钟线覆盖：`{summary.get('universe_count', 0)}` / `{summary.get('snapshot_symbol_count', 0)}` / `{summary.get('daily_symbol_count', 0)}` / `{summary.get('minute_symbol_count', 0)}`",
        f"- 候选池：`{summary.get('candidate_count', 0)}`，目标 `{summary.get('target_size', 0)}`；核心标的保留 `{summary.get('candidate_core_count', 0)}` / `{summary.get('core_symbol_count', 0)}`",
        f"- 核心来源：`{_core_symbol_source_cn(summary.get('core_symbol_source'))}`；Futu 自选股美股数 `{summary.get('core_watchlist_us_symbol_count', 0)}`；启用静态兜底 `{_yes_no_cn(summary.get('core_symbol_fallback_used'))}`",
        f"- 实际采集池：`{summary.get('collection_symbol_count', 0)}`；滚动追踪票 `{summary.get('collection_followup_count', 0)}`；追踪窗口 `{summary.get('collection_followup_days', 0)}` 天",
        f"- 候选池首字母最高占比：`{_letter_summary_text(summary.get('candidate_dominant_letter', {}))}`；实际采集池首字母最高占比：`{_letter_summary_text(summary.get('collection_dominant_letter', {}))}`",
        f"- 快照 / 日线 / 分钟线错误数：`{summary.get('snapshot_error_count', 0)}` / `{summary.get('daily_error_count', 0)}` / `{summary.get('minute_error_count', 0)}`",
    ]
    if bool(summary.get("alphabet_bias_warning")):
        lines.append("- 注意：今日候选池或实际采集池存在明显首字母集中，不能作为全市场追主力结论，只能解读已采集股票。")
    issues = summary.get("issues", [])
    if isinstance(issues, list) and issues:
        lines.append("- 问题：" + "；".join(str(item) for item in issues))
    return "\n".join(lines) + "\n"


def _coarse_universe_html(summary: dict[str, object]) -> str:
    issues = summary.get("issues", [])
    issue_text = ""
    if isinstance(issues, list) and issues:
        issue_text = "；问题=" + html.escape("；".join(str(item) for item in issues))
    warning_text = ""
    if bool(summary.get("alphabet_bias_warning")):
        warning_text = "；注意=今日候选池或实际采集池存在明显首字母集中，不能作为全市场追主力结论"
    return (
        "<div class='gate'><strong>粗筛股票池：</strong>可用={exists}；状态={status}；"
        "日期={date}；全市场={universe}；快照覆盖={snapshot}；日线覆盖={daily}；分钟线覆盖={minute}；"
        "候选={candidates}/{target}；核心={candidate_core}/{core}；核心来源={core_source}；Futu自选美股={watchlist_core}；静态兜底={fallback_used}；"
        "实际采集={collection_count}；滚动追踪={followup_count}；追踪窗口={followup_days}天；"
        "候选首字母最高占比={candidate_letter}；采集首字母最高占比={collection_letter}；"
        "错误数（快照/日线/分钟线）={snapshot_errors}/{daily_errors}/{minute_errors}{warning}{issues}</div>"
    ).format(
        exists=_yes_no_cn(summary.get("exists")),
        status=html.escape(str(summary.get("status") or "missing")),
        date=html.escape(str(summary.get("date") or "n/a")),
        universe=int(summary.get("universe_count") or 0),
        snapshot=int(summary.get("snapshot_symbol_count") or 0),
        daily=int(summary.get("daily_symbol_count") or 0),
        minute=int(summary.get("minute_symbol_count") or 0),
        candidates=int(summary.get("candidate_count") or 0),
        target=int(summary.get("target_size") or 0),
        candidate_core=int(summary.get("candidate_core_count") or 0),
        core=int(summary.get("core_symbol_count") or 0),
        core_source=html.escape(_core_symbol_source_cn(summary.get("core_symbol_source"))),
        watchlist_core=int(summary.get("core_watchlist_us_symbol_count") or 0),
        fallback_used=_yes_no_cn(summary.get("core_symbol_fallback_used")),
        collection_count=int(summary.get("collection_symbol_count") or 0),
        followup_count=int(summary.get("collection_followup_count") or 0),
        followup_days=int(summary.get("collection_followup_days") or 0),
        candidate_letter=html.escape(_letter_summary_text(summary.get("candidate_dominant_letter", {}))),
        collection_letter=html.escape(_letter_summary_text(summary.get("collection_dominant_letter", {}))),
        snapshot_errors=int(summary.get("snapshot_error_count") or 0),
        daily_errors=int(summary.get("daily_error_count") or 0),
        minute_errors=int(summary.get("minute_error_count") or 0),
        warning=warning_text,
        issues=issue_text,
    )


def _intraday_metric_table_markdown(metrics: object) -> list[str]:
    if not isinstance(metrics, list) or not metrics:
        return ["没有日内回放指标。"]
    lines = [
        "| 方向 | 观察窗口(分钟) | 样本数 | 合格样本 | 命中率 | 平均 Alpha | 最大单标的占比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {side} | {horizon} | {obs} | {quality} | {hit} | {alpha} | {symbol_share} |".format(
                side=_side_label_cn(row.get("side")),
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
        f"- 今日回放可用：`{_yes_no_cn(summary.get('exists'))}`",
        f"- 今日切点 / 事件 / forward return：`{summary.get('cutoff_count', 0)}` / `{summary.get('quality_event_count', 0)}` 个合格事件（总 `{summary.get('event_count', 0)}`） / `{summary.get('quality_return_count', 0)}` 行合格收益（总 `{summary.get('return_count', 0)}`）",
        f"- 今日观察窗口：`{summary.get('horizons_minutes') or []}`",
        f"- 累计日期 / 事件 / forward return：`{summary.get('cumulative_date_count', 0)}` / `{summary.get('cumulative_quality_event_count', 0)}` 个合格事件（总 `{summary.get('cumulative_event_count', 0)}`） / `{summary.get('cumulative_quality_return_count', 0)}` 行合格收益（总 `{summary.get('cumulative_return_count', 0)}`）",
        f"- 累计窗口：`{summary.get('cumulative_first_date') or 'n/a'}` 到 `{summary.get('cumulative_last_date') or 'n/a'}`；观察窗口 `{summary.get('cumulative_horizons_minutes') or []}`",
    ]
    issues = summary.get("issues", [])
    if isinstance(issues, list) and issues:
        lines.append("- 问题：" + "；".join(str(item) for item in issues))
    lines.extend(["", "今日指标："])
    lines.extend(_intraday_metric_table_markdown(summary.get("metrics", [])))
    lines.extend(["", "累计指标："])
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
    req_text = ", ".join(
        f"{_requirement_label_cn(key)}={_yes_no_cn(value)}" for key, value in sorted(requirements.items())
    )
    lines = [
        f"- 高置信是否可发布：`{_yes_no_cn(summary.get('ready'))}`",
        f"- 发布条件：{req_text}",
        f"- 正式验证样本：`{summary.get('official_event_count', 0)}` 个事件，`{summary.get('official_forward_return_count', 0)}` 行 forward return",
        f"- 影子样本：`{summary.get('shadow_event_count', 0)}` 个事件，`{summary.get('shadow_forward_return_count', 0)}` 行 forward return",
        f"- 探索样本：`{summary.get('exploration_event_count', 0)}` 个事件，`{summary.get('exploration_forward_return_count', 0)}` 行 forward return",
        f"- 当前报告可进入验证样本：`{summary.get('validation_eligible_count', 0)}`；如果是最终报告则为 `{summary.get('validation_eligible_if_final_count', 0)}`",
        f"- 累计日内回放：`{replay.get('date_count', 0)}` 个日期，`{replay.get('quality_event_count', 0)}` 个合格事件，`{replay.get('quality_return_count', 0)}` 行合格收益",
    ]
    if blockers:
        lines.append("- 阻塞项：" + "；".join(_blocker_cn(item) for item in blockers))
    rows = summary.get("side_gaps", [])
    if not isinstance(rows, list) or not rows:
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            "| 方向 | 已验证 | 还缺样本 | 还缺天数 | Alpha 差距 | 命中率差距 | 近期命中率差距 | Wilson 差距 | 集中度超限 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {side} | {validated} | {obs} | {days} | {alpha} | {hit} | {recent} | {wilson} | {concentration} |".format(
                side=_side_label_cn(row.get("side")),
                validated=_yes_no_cn(row.get("validated")),
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
        return "没有数据质量明细。\n"
    header = (
        "| 标的 | 合格 | 覆盖率 | 成交覆盖 | 盘口覆盖 | 报价覆盖 | "
        "成交笔数 | 原始成交行 | 重复序列行 | 成交额 | 重复率 | 点差(bps) |\n"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        body.append(
            "| {symbol} | {eligible} | {coverage} | {trade_cov} | {book_cov} | {quote_cov} | {trades} | {raw_trades} | {dup_rows} | {dollar} | {dup} | {spread} |".format(
                symbol=row.get("symbol", ""),
                eligible=_yes_no_cn(row.get("eligible")),
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


def _side_label_cn(side: object) -> str:
    return {
        "accumulation": "吸筹",
        "distribution": "出货",
    }.get(str(side or "").lower(), "未知")


def _confidence_label_cn(confidence: object) -> str:
    return {
        "high": "高置信",
        "watch": "观察",
        "diagnostic": "诊断",
    }.get(str(confidence or "").lower(), str(confidence or "未知"))


def _state_label_cn(gate: dict[str, object]) -> str:
    if bool(gate.get("validated")):
        return "已验证"
    state = str(gate.get("state") or "warmup").lower()
    return {
        "warmup": "暖场验证中",
        "disabled": "验证未启用",
        "validated": "已验证",
    }.get(state, state or "未知")


def _blocker_cn(blocker: object) -> str:
    text = str(blocker or "")
    return {
        "validation gate is not promoted": "验证门槛还没有通过",
        "data-quality gate is not passing": "数据质量还没有达标",
        "full-session NAS raw uploads are incomplete": "NAS 全天原始数据上传还不完整",
        "report is not a final post-close report": "当前还不是收盘后的最终报告",
    }.get(text, text)


def _yes_no_cn(value: object) -> str:
    return "是" if bool(value) else "否"


def _core_symbol_source_cn(value: object) -> str:
    text = str(value or "").strip()
    return {
        "futu_watchlist": "Futu 自选股",
        "file_fallback": "静态文件兜底",
        "file": "静态文件",
        "explicit": "显式传入",
        "none": "未启用",
    }.get(text, text or "未知")


def _reason_cn(reason: object) -> str:
    text = str(reason or "")
    translations = {
        "forward validation sample not promoted yet": "forward 验证样本还没有晋级",
        "validation gate not configured": "验证门槛未配置",
        "validation gate not promoted": "验证门槛还没有通过",
        "missing 5d validation metrics": "缺少 5 日验证指标",
        "insufficient independent evidence": "独立证据不足",
        "positive active tape": "主动买盘偏强",
        "negative active tape": "主动卖盘偏强",
        "supportive bid/depth absorption": "买盘深度和承接较强",
        "weak bid/depth or ask replenishment": "买盘深度偏弱或卖盘补单明显",
        "price holds near/above VWAP with controlled impact": "价格守住或高于 VWAP，冲击可控",
        "price below VWAP with controlled selling pressure": "价格低于 VWAP，卖压冲击可控",
        "not final post-close report": "还不是收盘后的最终报告",
    }
    parts = [part.strip() for part in text.split(";") if part.strip()]
    if not parts:
        return ""
    return "；".join(translations.get(part, part) for part in parts)


def _stage_label_cn(stage: object) -> str:
    return {
        "stealth_accumulation": "隐蔽吸筹",
        "accumulation_watch": "吸筹观察",
        "accumulation_diagnostic": "吸筹诊断",
        "distribution_risk": "出货风险",
        "distribution_watch": "出货观察",
        "distribution_diagnostic": "出货诊断",
    }.get(str(stage or "").lower(), str(stage or "未知"))


def _requirement_label_cn(key: object) -> str:
    return {
        "validation_gate_validated": "验证门槛通过",
        "data_quality_gate_ready": "数据质量达标",
        "nas_uploads_complete": "NAS 原始数据完整",
        "final_report_complete": "收盘后最终报告",
    }.get(str(key or ""), str(key or ""))


def _blocking_count_label_cn(key: object) -> str:
    return {
        "missing_symbol": "缺少代码",
        "invalid_side": "方向无效",
        "score_below_min": "分数低于门槛",
        "not_watch_or_high": "不是观察或高置信",
        "data_quality_failed": "数据质量不通过",
        "not_final_report": "非最终报告",
    }.get(str(key or ""), str(key or ""))


def _side_counts(signals: pd.DataFrame, confidence: str | None = None) -> tuple[int, int, int]:
    if signals.empty or "side" not in signals.columns:
        return (0, 0, 0)
    frame = signals
    if confidence is not None:
        if "confidence" not in frame.columns:
            return (0, 0, 0)
        frame = frame[frame["confidence"].astype(str).str.lower() == str(confidence).lower()]
    sides = frame["side"].astype(str).str.lower() if not frame.empty else pd.Series(dtype=str)
    total = int(len(frame))
    accumulation = int((sides == "accumulation").sum())
    distribution = int((sides == "distribution").sum())
    return (total, accumulation, distribution)


def _top_signal_lines_cn(signals: pd.DataFrame, *, confidence: str, limit: int = 5) -> list[str]:
    if signals.empty or "confidence" not in signals.columns:
        return []
    frame = signals[signals["confidence"].astype(str).str.lower() == confidence.lower()].copy()
    if frame.empty:
        return []
    if "side_score" in frame.columns:
        frame["_score_sort"] = pd.to_numeric(frame["side_score"], errors="coerce").fillna(0.0)
        frame = frame.sort_values("_score_sort", ascending=False)
    lines = []
    for _, row in frame.head(limit).iterrows():
        symbol = str(row.get("symbol") or "").replace("US.", "")
        side = _side_label_cn(row.get("side"))
        confidence_label = _confidence_label_cn(row.get("confidence"))
        score = _score(row.get("side_score"))
        net_active = _money(row.get("net_active_dollar"))
        buy_ratio = _pct(row.get("active_buy_ratio"))
        lines.append(f"{symbol}：{side}，{confidence_label}，分数 {score}，净主动资金 {net_active}，主动买入占比 {buy_ratio}")
    return lines


def _alphabet_bias_warning_cn(coarse_universe: dict[str, object] | None) -> str:
    if not isinstance(coarse_universe, dict) or not bool(coarse_universe.get("alphabet_bias_warning")):
        return ""
    details = []
    candidate_text = _letter_summary_text(coarse_universe.get("candidate_dominant_letter", {}))
    collection_text = _letter_summary_text(coarse_universe.get("collection_dominant_letter", {}))
    if candidate_text != "n/a":
        details.append(f"候选池 {candidate_text}")
    if collection_text != "n/a":
        details.append(f"实际采集池 {collection_text}")
    detail_text = "；".join(details) if details else "股票池首字母过度集中"
    return f"采集池异常：{detail_text}。今日候选作废，不展示观察名单，不能作为全市场追主力结论。"


def _chinese_conclusion_lines(
    *,
    signals: pd.DataFrame,
    validation_gate: dict[str, object],
    data_quality: dict[str, object],
    validation_progress: dict[str, object],
    confidence_gap: dict[str, object],
    coarse_universe: dict[str, object] | None = None,
) -> list[str]:
    high_count, high_accumulation, high_distribution = _side_counts(signals, "high")
    watch_count, _, _ = _side_counts(signals, "watch")
    diagnostic_count, _, _ = _side_counts(signals, "diagnostic")
    state = _state_label_cn(validation_gate)
    alphabet_warning = _alphabet_bias_warning_cn(coarse_universe)
    if alphabet_warning:
        first_line = "结论：今日股票池采集异常，本日报作废，不发布主力进出结论。"
    elif high_count > 0:
        first_line = f"结论：今日有 {high_count} 个高置信追主力信号（吸筹 {high_accumulation} 个，出货 {high_distribution} 个）。"
    elif bool(confidence_gap.get("ready")):
        first_line = "结论：今日没有高置信主力进出信号。"
    else:
        first_line = "结论：今日不发布高置信主力进出结论，当前仍在验证或数据质量检查阶段。"

    requirements = confidence_gap.get("requirements", {})
    if not isinstance(requirements, dict):
        requirements = {}
    final_report = "是" if bool(requirements.get("final_report_complete")) else "否"
    readiness = "已满足" if bool(confidence_gap.get("ready")) else "未满足"
    nas_status = "完整" if bool(data_quality.get("nas_upload_complete")) else "不完整"
    blockers = confidence_gap.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = []

    lines = [
        first_line,
        f"当前状态：{state}；高置信 {high_count} 个，观察 {watch_count} 个，诊断 {diagnostic_count} 个。",
        f"报告口径：收盘后最终报告={final_report}；高置信发布条件={readiness}。",
        "数据质量：合格股票 {eligible}/{total}；NAS 原始数据上传{nas_status}；成交/挂单覆盖中位数 {trade}/{book}。".format(
            eligible=int(data_quality.get("eligible_symbol_count") or 0),
            total=int(data_quality.get("symbol_count") or 0),
            nas_status=nas_status,
            trade=_pct(data_quality.get("median_trade_coverage_ratio_regular")),
            book=_pct(data_quality.get("median_book_coverage_ratio_regular")),
        ),
        "验证样本：正式 {events} 个事件 / {returns} 行 forward return；影子校准 {shadow_events}/{shadow_returns}；探索校准 {explore_events}/{explore_returns}。".format(
            events=int(validation_progress.get("event_count") or 0),
            returns=int(validation_progress.get("forward_return_count") or 0),
            shadow_events=int(validation_progress.get("shadow_event_count") or 0),
            shadow_returns=int(validation_progress.get("shadow_forward_return_count") or 0),
            explore_events=int(validation_progress.get("exploration_event_count") or 0),
            explore_returns=int(validation_progress.get("exploration_forward_return_count") or 0),
        ),
    ]
    if blockers:
        lines.append("没有高置信的主要原因：" + "；".join(_blocker_cn(item) for item in blockers[:4]) + "。")
    if alphabet_warning:
        lines.append(alphabet_warning)

    top_high = _top_signal_lines_cn(signals, confidence="high")
    if alphabet_warning:
        lines.append("今日观察候选：不展示，已屏蔽偏置股票池产生的候选。")
    elif top_high:
        lines.append("高置信标的：" + "；".join(top_high) + "。")
    else:
        top_watch = _top_signal_lines_cn(signals, confidence="watch")
        if top_watch:
            lines.append("今日观察候选：" + "；".join(top_watch) + "。")
        else:
            lines.append("今日观察候选：暂无。")
    lines.append("口径说明：这里抓的是逐笔成交、盘口和报价共同指向的隐蔽吸筹/出货迹象，不等同于确认具体机构账户身份。")
    return lines


def _chinese_conclusion_markdown(
    *,
    signals: pd.DataFrame,
    validation_gate: dict[str, object],
    data_quality: dict[str, object],
    validation_progress: dict[str, object],
    confidence_gap: dict[str, object],
    coarse_universe: dict[str, object] | None = None,
) -> str:
    lines = _chinese_conclusion_lines(
        signals=signals,
        validation_gate=validation_gate,
        data_quality=data_quality,
        validation_progress=validation_progress,
        confidence_gap=confidence_gap,
        coarse_universe=coarse_universe,
    )
    return "## 今日结论\n\n" + "\n".join(f"- {line}" for line in lines) + "\n"


def _chinese_conclusion_html(
    *,
    signals: pd.DataFrame,
    validation_gate: dict[str, object],
    data_quality: dict[str, object],
    validation_progress: dict[str, object],
    confidence_gap: dict[str, object],
    coarse_universe: dict[str, object] | None = None,
) -> str:
    lines = _chinese_conclusion_lines(
        signals=signals,
        validation_gate=validation_gate,
        data_quality=data_quality,
        validation_progress=validation_progress,
        confidence_gap=confidence_gap,
        coarse_universe=coarse_universe,
    )
    if not lines:
        return ""
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines[1:])
    return (
        "<section class='cn-summary'>"
        "<h2>今日结论</h2>"
        f"<p><strong>{html.escape(lines[0])}</strong></p>"
        f"<ul>{items}</ul>"
        "</section>"
    )


def render_markdown_report(
    *,
    date: str,
    signals: pd.DataFrame,
    features: pd.DataFrame,
    raw_counts: dict[str, int],
    validation_gate: dict,
    data_quality: dict[str, object],
    coarse_universe: dict[str, object],
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
    state = _state_label_cn(validation_gate)
    lines = [
        f"# 追主力日报 - {date}",
        "",
        _chinese_conclusion_markdown(
            signals=signals,
            validation_gate=validation_gate,
            data_quality=data_quality,
            validation_progress=validation_progress,
            confidence_gap=confidence_gap,
            coarse_universe=coarse_universe,
        ),
        "",
        f"报告状态：`{state}`",
        f"高置信候选：`{high_count}`",
        "",
        "本报告使用 Futu OpenD 的逐笔成交、盘口快照和报价数据。"
        "它识别的是主力行为迹象，不声称能确认具体机构账户身份。",
        "",
        "## 验证状态",
        "",
        f"- 验证门槛是否通过：`{_yes_no_cn(validation_gate.get('validated'))}`",
        f"- 验证原因：{_reason_cn(validation_gate.get('reason', ''))}",
        f"- 正式验证样本：`{validation_progress.get('event_count', 0)}` 个事件，`{validation_progress.get('forward_return_count', 0)}` 行 forward return",
        f"- 影子校准样本：`{validation_progress.get('shadow_event_count', 0)}` 个事件，`{validation_progress.get('shadow_forward_return_count', 0)}` 行 forward return；最低分 `{_score(validation_progress.get('shadow_min_event_score'))}`",
        f"- 探索校准样本：`{validation_progress.get('exploration_event_count', 0)}` 个事件，`{validation_progress.get('exploration_forward_return_count', 0)}` 行 forward return；最低分 `{_score(validation_progress.get('exploration_min_event_score'))}`",
        f"- 晋级观察周期：`{validation_progress.get('promotion_horizon', 0)}d`；基准：`{validation_progress.get('benchmark') or 'n/a'}`",
        f"- 可用于高置信报告的合格股票：`{data_quality.get('eligible_symbol_count', 0)}` / `{data_quality.get('symbol_count', 0)}`",
        f"- NAS 原始数据上传完整：`{_yes_no_cn(data_quality.get('nas_upload_complete'))}`；manifest 行数：`{data_quality.get('manifest_count', 0)}`",
        f"- 成交/盘口覆盖率中位数：`{_pct(data_quality.get('median_trade_coverage_ratio_regular'))}` / `{_pct(data_quality.get('median_book_coverage_ratio_regular'))}`",
        f"- 重复序列行：`{data_quality.get('duplicate_sequence_count', 0)}` / `{data_quality.get('raw_trade_count', 0)}`（`{_pct(data_quality.get('duplicate_sequence_rate'))}`）",
        "",
        "## 高置信准备度",
        "",
        _confidence_gap_markdown(confidence_gap),
        "",
        "## 分方向验证进度",
        "",
        _validation_markdown_table(validation_progress),
        "",
        "## 验证样本入账资格",
        "",
        _eligibility_markdown(eligibility),
        "",
        "## 日内回放校准",
        "",
        _intraday_replay_markdown(intraday_replay),
        "",
        "## 数据覆盖",
        "",
        _coarse_universe_markdown(coarse_universe),
        "",
        f"- 原始逐笔成交行数：`{raw_counts.get('trades', 0)}`",
        f"- 原始盘口行数：`{raw_counts.get('order_book', 0)}`",
        f"- 原始报价行数：`{raw_counts.get('quotes', 0)}`",
        f"- 有特征的股票数：`{coverage['symbol_count']}`",
        f"- 特征分钟数：`{coverage['minute_count']}`",
        f"- 正常交易时段特征分钟数：`{coverage['regular_minute_count']}`",
        f"- 正常交易时段成交/盘口/报价分钟数：`{coverage['regular_trade_minutes']}` / `{coverage['regular_book_minutes']}` / `{coverage['regular_quote_minutes']}`",
        "",
        "## 候选标的",
        "",
        _markdown_table(view),
        "",
        "## 分标的数据质量",
        "",
        _quality_markdown_table(data_quality),
    ]
    return "\n".join(lines)


def _html_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "<p>没有候选标的。</p>"
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
                side=html.escape(_side_label_cn(row.get("side"))),
                score=_score(row.get("side_score")),
                confidence=html.escape(_confidence_label_cn(row.get("confidence"))),
                stage=html.escape(_stage_label_cn(row.get("stage"))),
                dollar=_money(row.get("dollar_volume")),
                net=_money(row.get("net_active_dollar")),
                buy_ratio=_pct(row.get("active_buy_ratio")),
                vwap=_bps(row.get("vwap_deviation_bps")),
                spread=_bps(row.get("spread_bps")),
                reason=html.escape(_reason_cn(row.get("reason"))),
            )
        )
    return (
        "<table><tr><th>排名</th><th>标的</th><th>方向</th><th>分数</th><th>置信度</th>"
        "<th>阶段</th><th>成交额</th><th>净主动资金</th><th>主动买入占比</th>"
        "<th>VWAP 偏离(bps)</th><th>点差(bps)</th><th>原因</th></tr>"
        + "\n".join(table_rows)
        + "</table>"
    )


def _quality_html_table(data_quality: dict[str, object]) -> str:
    rows = data_quality.get("symbols", [])
    if not isinstance(rows, list) or not rows:
        return "<p>没有数据质量明细。</p>"
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
                eligible=_yes_no_cn(row.get("eligible")),
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
        "<table><tr><th>标的</th><th>合格</th><th>覆盖率</th><th>成交覆盖</th>"
        "<th>盘口覆盖</th><th>报价覆盖</th><th>成交笔数</th><th>原始成交行</th><th>重复序列行</th><th>成交额</th>"
        "<th>重复率</th><th>点差(bps)</th></tr>"
        + "\n".join(table_rows)
        + "</table>"
    )


def _validation_html_table(progress: dict[str, object]) -> str:
    rows = progress.get("sides", [])
    if not isinstance(rows, list) or not rows:
        return "<p>没有按方向拆分的验证记录。</p>"
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
                side=html.escape(_side_label_cn(row.get("side"))),
                validated=_yes_no_cn(row.get("validated")),
                reason=html.escape(_reason_cn(row.get("reason"))),
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
        "<table><tr><th>方向</th><th>已验证</th><th>原因</th><th>样本数</th><th>天数</th>"
        "<th>Alpha</th><th>命中率</th><th>近期命中率</th><th>Wilson 下界</th><th>单标的集中度</th></tr>"
        + "\n".join(table_rows)
        + "</table>"
    )


def _eligibility_html(summary: dict[str, object]) -> str:
    blockers = summary.get("blocking_counts", {})
    if not isinstance(blockers, dict):
        blockers = {}
    blocker_text = "，".join(
        f"{html.escape(_blocking_count_label_cn(key))}={int(value or 0)}" for key, value in sorted(blockers.items())
    )
    return (
        "<div class='gate'><strong>验证样本入账资格：</strong>"
        "当前可入账={eligible}；如果是最终报告可入账={eligible_if_final}；"
        "分数达标={score_pass}；接近达标={near_score}；最高分={max_score}；"
        "观察或高置信={watch}；数据质量通过={quality}；最终报告行数={final}；"
        "未入账原因={blockers}</div>"
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
        return "<p>没有日内回放指标。</p>"
    rows = []
    for row in metrics:
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr><td>{side}</td><td>{horizon}</td><td>{obs}</td><td>{quality}</td>"
            "<td>{hit}</td><td>{alpha}</td><td>{symbol_share}</td></tr>".format(
                side=html.escape(_side_label_cn(row.get("side"))),
                horizon=int(row.get("horizon_minutes") or 0),
                obs=int(row.get("observation_count") or 0),
                quality=int(row.get("quality_observation_count") or 0),
                hit=_pct(row.get("hit_rate")),
                alpha=_pct(row.get("avg_alpha")),
                symbol_share=_pct(row.get("max_symbol_sample_share")),
            )
        )
    if not rows:
        return "<p>没有日内回放指标。</p>"
    return (
        "<table><tr><th>方向</th><th>观察窗口(分钟)</th><th>样本数</th><th>合格样本</th>"
        "<th>命中率</th><th>平均 Alpha</th><th>最大单标的占比</th></tr>"
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
    req_text = "，".join(
        f"{html.escape(_requirement_label_cn(key))}={_yes_no_cn(value)}" for key, value in sorted(requirements.items())
    )
    blocker_text = "；".join(html.escape(_blocker_cn(item)) for item in blockers) if blockers else "无"
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
                    side=html.escape(_side_label_cn(row.get("side"))),
                    validated=_yes_no_cn(row.get("validated")),
                    obs=int(row.get("observations_needed") or 0),
                    days=int(row.get("signal_days_needed") or 0),
                    alpha=_pct(row.get("alpha_gap")),
                    hit=_pct(row.get("hit_rate_gap")),
                    recent=_pct(row.get("recent_hit_rate_gap")),
                    wilson=_pct(row.get("wilson_gap")),
                    concentration=_pct(row.get("concentration_excess")),
                )
            )
    table = "<p>没有高置信差距明细。</p>"
    if rows:
        table = (
            "<table><tr><th>方向</th><th>已验证</th><th>还缺样本</th><th>还缺天数</th>"
            "<th>Alpha 差距</th><th>命中率差距</th><th>近期命中率差距</th><th>Wilson 差距</th>"
            "<th>集中度超限</th></tr>"
            + "\n".join(rows)
            + "</table>"
        )
    return (
        "<div class='gate'><strong>高置信准备度：</strong>可发布={ready}；"
        "条件={requirements}；阻塞项={blockers}</div>"
        "<div class='gate'><strong>验证样本差距：</strong>正式事件={official_events}；"
        "正式 forward return={official_returns}；影子事件={shadow_events}；"
        "影子 forward return={shadow_returns}；探索事件={exploration_events}；"
        "探索 forward return={exploration_returns}；当前可入账={eligible}；最终报告可入账={eligible_if_final}；"
        "累计回放日期={replay_dates}；累计合格事件={replay_events}；"
        "累计合格收益={replay_returns}</div>{table}"
    ).format(
        ready=_yes_no_cn(summary.get("ready")),
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
        issue_text = "；问题=" + html.escape("；".join(str(item) for item in issues))
    if summary.get("cumulative_first_date") or summary.get("cumulative_last_date"):
        cumulative_window = "{first} 到 {last}".format(
            first=html.escape(str(summary.get("cumulative_first_date") or "n/a")),
            last=html.escape(str(summary.get("cumulative_last_date") or "n/a")),
        )
    else:
        cumulative_window = "n/a"
    return (
        "<div class='gate'><strong>今日日内回放：</strong>可用={exists}；"
        "切点={cutoffs}；合格事件={quality_events}/{events}；"
        "合格收益={quality_returns}/{returns}；观察窗口={horizons}{issues}</div>"
        "<h3>今日指标</h3>{today_table}"
        "<div class='gate'><strong>累计日内回放：</strong>日期数={cum_dates}；"
        "窗口={cum_window}；合格事件={cum_quality_events}/{cum_events}；"
        "合格收益={cum_quality_returns}/{cum_returns}；观察窗口={cum_horizons}</div>"
        "<h3>累计指标</h3>{cumulative_table}"
    ).format(
        exists=_yes_no_cn(summary.get("exists")),
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
    coarse_universe: dict[str, object],
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
    state = html.escape(_state_label_cn(validation_gate))
    reason = html.escape(_reason_cn(validation_gate.get("reason") or ""))
    chinese_conclusion = _chinese_conclusion_html(
        signals=signals,
        validation_gate=validation_gate,
        data_quality=data_quality,
        validation_progress=validation_progress,
        confidence_gap=confidence_gap,
        coarse_universe=coarse_universe,
    )
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
.cn-summary {{ border-left: 5px solid #16794c; background: #f1f8f4; padding: 14px 18px; margin: 18px 0; }}
.cn-summary h2 {{ margin: 0 0 8px; }}
.cn-summary p {{ margin: 0 0 10px; }}
.cn-summary ul {{ margin: 0; padding-left: 20px; }}
.cn-summary li {{ margin: 6px 0; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #d9e2ec; padding: 7px; text-align: left; font-size: 13px; }}
th {{ background: #263238; color: #fff; }}
tr.buy {{ background: #edf7ed; }}
tr.sell {{ background: #fff1f2; }}
.muted {{ color: #667085; }}
</style>
</head>
<body>
<h1>追主力日报 - {html.escape(date)}</h1>
<div class="metric"><div class="value">{state}</div><div class="label">报告状态</div></div>
<div class="metric"><div class="value">{high_count}</div><div class="label">高置信信号</div></div>
<div class="metric"><div class="value">{coverage['symbol_count']}</div><div class="label">股票数</div></div>
<div class="metric"><div class="value">{coverage['minute_count']}</div><div class="label">特征分钟数</div></div>
{chinese_conclusion}
<p class="muted">本报告使用 Futu OpenD 的逐笔成交、盘口快照和报价数据。它识别的是主力行为迹象，不声称能确认具体机构账户身份。</p>
<div class="gate"><strong>验证门槛：</strong>已通过={_yes_no_cn(validation_gate.get('validated'))}；{reason}</div>
<div class="gate"><strong>正式验证样本：</strong>事件={validation_progress.get('event_count', 0)}；forward return={validation_progress.get('forward_return_count', 0)}；晋级观察周期={validation_progress.get('promotion_horizon', 0)}d；基准={html.escape(str(validation_progress.get('benchmark') or 'n/a'))}</div>
<div class="gate"><strong>影子校准：</strong>事件={validation_progress.get('shadow_event_count', 0)}；forward return={validation_progress.get('shadow_forward_return_count', 0)}；最低分={_score(validation_progress.get('shadow_min_event_score'))}</div>
<div class="gate"><strong>探索校准：</strong>事件={validation_progress.get('exploration_event_count', 0)}；forward return={validation_progress.get('exploration_forward_return_count', 0)}；最低分={_score(validation_progress.get('exploration_min_event_score'))}</div>
<div class="gate"><strong>数据质量门槛：</strong>合格股票={data_quality.get('eligible_symbol_count', 0)}/{data_quality.get('symbol_count', 0)}；成交/盘口覆盖中位数={_pct(data_quality.get('median_trade_coverage_ratio_regular'))}/{_pct(data_quality.get('median_book_coverage_ratio_regular'))}；NAS 上传完整={_yes_no_cn(data_quality.get('nas_upload_complete'))}；manifest 行数={data_quality.get('manifest_count', 0)}</div>
<div class="gate"><strong>重复序列审计：</strong>重复序列行={data_quality.get('duplicate_sequence_count', 0)}/{data_quality.get('raw_trade_count', 0)}（{_pct(data_quality.get('duplicate_sequence_rate'))}）</div>
<h2>高置信准备度</h2>
{_confidence_gap_html(confidence_gap)}
<h2>分方向验证进度</h2>
{_validation_html_table(validation_progress)}
<h2>验证样本入账资格</h2>
{_eligibility_html(eligibility)}
<h2>日内回放校准</h2>
{_intraday_replay_html(intraday_replay)}
<h2>数据覆盖</h2>
{_coarse_universe_html(coarse_universe)}
<p>原始逐笔成交={raw_counts.get('trades', 0)}，原始盘口={raw_counts.get('order_book', 0)}，原始报价={raw_counts.get('quotes', 0)}。正常交易时段成交/盘口/报价分钟数={coverage['regular_trade_minutes']} / {coverage['regular_book_minutes']} / {coverage['regular_quote_minutes']}。</p>
<h2>候选标的</h2>
{_html_table(view)}
<h2>分标的数据质量</h2>
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


def _subject(
    signals: pd.DataFrame,
    gate: dict[str, object],
    coarse_universe: dict[str, object] | None = None,
) -> str:
    if isinstance(coarse_universe, dict) and bool(coarse_universe.get("alphabet_bias_warning")):
        return "追主力日报 - 股票池异常，本日报作废"
    high_count, accumulation_count, distribution_count = _side_counts(signals, "high")
    if high_count > 0:
        return f"追主力日报 - 高置信 {high_count} 个（吸筹 {accumulation_count} / 出货 {distribution_count}）"
    return f"追主力日报 - {_state_label_cn(gate)}，暂无高置信信号"


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
    coarse_universe = _load_coarse_universe_summary(base_dir, args.date)
    intraday_replay = _load_intraday_replay_summary(base_dir, args.date)
    validation_progress = _validation_progress(report_gate)
    validation_eligibility = _validation_eligibility_summary(
        signals,
        min_event_score=_validation_min_event_score(report_gate),
    )
    email_subject = _subject(signals, report_gate, coarse_universe)
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
        "coarse_universe": coarse_universe,
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
        coarse_universe=coarse_universe,
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
        coarse_universe=coarse_universe,
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

        os.environ.setdefault("REPORT_DELIVERY_METHOD", "smtp")
        os.environ.setdefault("SMTP_RETRIES", "3")
        os.environ.setdefault("SENDMAIL_FALLBACK", "false")
        os.environ.setdefault("MAIL_APP_FALLBACK", "false")

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
