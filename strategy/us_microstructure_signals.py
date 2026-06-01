"""Signal scoring for US microstructure major-flow candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MicrostructureSignalConfig:
    """Conservative first-pass scoring thresholds."""

    min_trade_count: int = 1_000
    min_dollar_volume: float = 50_000_000.0
    min_data_coverage: float = 0.80
    watch_score: float = 70.0
    high_score: float = 85.0
    max_spread_bps: float = 20.0


def load_validation_gate(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"state": "warmup", "validated": False, "reason": "validation gate not configured"}
    target = Path(path).expanduser()
    if not target.exists():
        return {"state": "warmup", "validated": False, "reason": f"validation gate missing: {target}"}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"state": "disabled", "validated": False, "reason": f"validation gate unreadable: {exc}"}
    if not isinstance(payload, dict):
        return {"state": "disabled", "validated": False, "reason": "validation gate is not a JSON object"}
    validated = bool(payload.get("validated")) or str(payload.get("state") or payload.get("status") or "").lower() == "validated"
    state = "validated" if validated else str(payload.get("state") or payload.get("status") or "warmup").lower()
    payload["validated"] = validated
    payload["state"] = state
    payload.setdefault("reason", "validation gate active" if validated else "validation gate not promoted")
    return payload


def _score_between(value: float, low: float, high: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val):
        return 0.0
    if high == low:
        return 1.0 if val >= high else 0.0
    return min(1.0, max(0.0, (val - low) / (high - low)))


def _score_below(value: float, good: float, bad: float) -> float:
    try:
        val = abs(float(value))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val):
        return 0.0
    if bad <= good:
        return 1.0 if val <= good else 0.0
    return 1.0 - min(1.0, max(0.0, (val - good) / (bad - good)))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _last_valid(series: pd.Series, default: float = np.nan) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return default
    return float(cleaned.iloc[-1])


def _first_valid(series: pd.Series, default: float = np.nan) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return default
    return float(cleaned.iloc[0])


def _mean_valid(series: pd.Series, default: float = 0.0) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return default
    return float(cleaned.mean())


def _sum_valid(series: pd.Series, default: float = 0.0) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return default
    return float(cleaned.sum())


def _median_valid(series: pd.Series, default: float = 0.0) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return default
    return float(cleaned.median())


def _series(df: pd.DataFrame, column: str, default: Any = 0.0) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _reason(labels: list[tuple[str, bool]]) -> str:
    selected = [label for label, active in labels if active]
    return "; ".join(selected) if selected else "insufficient independent evidence"


def _summarize_symbol(symbol: str, part: pd.DataFrame, cfg: MicrostructureSignalConfig) -> dict[str, Any]:
    part = part.sort_values("minute").copy()
    trade_count = int(pd.to_numeric(_series(part, "trade_count"), errors="coerce").fillna(0).sum())
    raw_trade_count = (
        int(_sum_valid(part["raw_trade_count"], float(trade_count)))
        if "raw_trade_count" in part.columns
        else trade_count
    )
    duplicate_sequence_count = (
        int(_sum_valid(part["duplicate_sequence_count"], 0.0))
        if "duplicate_sequence_count" in part.columns
        else 0
    )
    dollar_volume = float(pd.to_numeric(_series(part, "dollar_volume"), errors="coerce").fillna(0).sum())
    active_buy = float(pd.to_numeric(_series(part, "active_buy_dollar"), errors="coerce").fillna(0).sum())
    active_sell = float(pd.to_numeric(_series(part, "active_sell_dollar"), errors="coerce").fillna(0).sum())
    active_total = active_buy + active_sell
    net_active = active_buy - active_sell
    net_ratio = net_active / active_total if active_total > 0 else 0.0
    buy_ratio = active_buy / active_total if active_total > 0 else 0.5
    sell_ratio = active_sell / active_total if active_total > 0 else 0.5
    coverage_minutes = int((_series(part, "has_trade_data", False).fillna(False) | _series(part, "has_book_data", False).fillna(False)).sum())
    coverage_ratio = _last_valid(_series(part, "coverage_ratio_regular", np.nan), 0.0)
    if coverage_ratio <= 0 and coverage_minutes:
        coverage_ratio = min(1.0, coverage_minutes / 390.0)
    trade_coverage_minutes = int(_series(part, "has_trade_data", False).fillna(False).sum())
    book_coverage_minutes = int(_series(part, "has_book_data", False).fillna(False).sum())
    quote_coverage_minutes = int(_series(part, "has_quote_data", False).fillna(False).sum())
    trade_coverage_ratio = _last_valid(_series(part, "trade_coverage_ratio_regular", np.nan), coverage_ratio)
    book_coverage_ratio = _last_valid(_series(part, "book_coverage_ratio_regular", np.nan), coverage_ratio)
    quote_coverage_ratio = _last_valid(_series(part, "quote_coverage_ratio_regular", np.nan), 0.0)

    first_price = _first_valid(_series(part, "reference_price", np.nan))
    last_price = _last_valid(_series(part, "reference_price", np.nan))
    price_change_bps = (last_price / first_price - 1.0) * 10_000 if first_price and math.isfinite(first_price) else 0.0
    vwap_deviation_bps = _last_valid(_series(part, "vwap_deviation_bps", np.nan), 0.0)
    median_impact = _median_valid(_series(part, "price_impact_bps_per_musd", np.nan), 0.0)
    median_spread = _median_valid(_series(part, "spread_bps", np.nan), cfg.max_spread_bps)
    depth_imbalance_1 = _mean_valid(_series(part, "depth_imbalance_1", np.nan), 0.0)
    depth_imbalance_5 = _mean_valid(_series(part, "depth_imbalance_5", np.nan), 0.0)
    bid_replenish = _mean_valid(_series(part, "bid_replenish_1", np.nan), 0.0)
    ask_replenish = _mean_valid(_series(part, "ask_replenish_1", np.nan), 0.0)
    dollar_z_max = _finite(pd.to_numeric(_series(part, "dollar_volume_z", 0.0), errors="coerce").max(), 0.0)
    odd_lot_ratio = _mean_valid(_series(part, "odd_lot_ratio", np.nan), 0.0)
    duplicate_rate = (
        duplicate_sequence_count / raw_trade_count
        if raw_trade_count > 0
        else _mean_valid(_series(part, "duplicate_sequence_rate", np.nan), 0.0)
    )

    liquidity_quality = min(
        _score_between(trade_count, cfg.min_trade_count * 0.25, cfg.min_trade_count),
        _score_between(dollar_volume, cfg.min_dollar_volume * 0.10, cfg.min_dollar_volume),
    )
    coverage_quality = _score_between(coverage_ratio, cfg.min_data_coverage * 0.5, cfg.min_data_coverage)
    spread_quality = _score_below(median_spread, cfg.max_spread_bps * 0.35, cfg.max_spread_bps)
    context_score = 0.40 * liquidity_quality + 0.35 * coverage_quality + 0.25 * spread_quality

    acc_tape = (
        0.45 * _score_between(net_ratio, 0.02, 0.25)
        + 0.35 * _score_between(buy_ratio, 0.52, 0.68)
        + 0.20 * _score_between(dollar_z_max, 0.5, 2.0)
    )
    acc_book = (
        0.45 * _score_between(depth_imbalance_1, -0.02, 0.25)
        + 0.30 * _score_between(depth_imbalance_5, -0.02, 0.20)
        + 0.15 * _score_between(bid_replenish, 0.0, 500.0)
        + 0.10 * spread_quality
    )
    acc_impact = (
        0.45 * _score_between(vwap_deviation_bps, -10.0, 35.0)
        + 0.25 * _score_between(price_change_bps, -20.0, 120.0)
        + 0.30 * _score_below(median_impact, 10.0, 100.0)
    )

    dist_tape = (
        0.45 * _score_between(-net_ratio, 0.02, 0.25)
        + 0.35 * _score_between(sell_ratio, 0.52, 0.68)
        + 0.20 * _score_between(dollar_z_max, 0.5, 2.0)
    )
    dist_book = (
        0.45 * _score_between(-depth_imbalance_1, -0.02, 0.25)
        + 0.30 * _score_between(-depth_imbalance_5, -0.02, 0.20)
        + 0.15 * _score_between(ask_replenish, 0.0, 500.0)
        + 0.10 * spread_quality
    )
    dist_impact = (
        0.45 * _score_between(-vwap_deviation_bps, -10.0, 35.0)
        + 0.25 * _score_between(-price_change_bps, -20.0, 120.0)
        + 0.30 * _score_below(median_impact, 10.0, 100.0)
    )

    accumulation_score = 100.0 * (0.30 * acc_tape + 0.30 * acc_book + 0.25 * acc_impact + 0.15 * context_score)
    distribution_score = 100.0 * (0.30 * dist_tape + 0.30 * dist_book + 0.25 * dist_impact + 0.15 * context_score)

    acc_blocks = {
        "tape": acc_tape >= 0.55,
        "book": acc_book >= 0.55,
        "impact": acc_impact >= 0.55,
    }
    dist_blocks = {
        "tape": dist_tape >= 0.55,
        "book": dist_book >= 0.55,
        "impact": dist_impact >= 0.55,
    }

    return {
        "symbol": symbol,
        "minute_count": int(len(part)),
        "coverage_minutes": coverage_minutes,
        "coverage_ratio_regular": coverage_ratio,
        "trade_coverage_minutes": trade_coverage_minutes,
        "book_coverage_minutes": book_coverage_minutes,
        "quote_coverage_minutes": quote_coverage_minutes,
        "trade_coverage_ratio_regular": trade_coverage_ratio,
        "book_coverage_ratio_regular": book_coverage_ratio,
        "quote_coverage_ratio_regular": quote_coverage_ratio,
        "trade_count": trade_count,
        "raw_trade_count": raw_trade_count,
        "duplicate_sequence_count": duplicate_sequence_count,
        "dollar_volume": dollar_volume,
        "active_buy_dollar": active_buy,
        "active_sell_dollar": active_sell,
        "net_active_dollar": net_active,
        "net_active_ratio": net_ratio,
        "active_buy_ratio": buy_ratio,
        "active_sell_ratio": sell_ratio,
        "price_change_bps": price_change_bps,
        "vwap_deviation_bps": vwap_deviation_bps,
        "price_impact_bps_per_musd": median_impact,
        "spread_bps": median_spread,
        "depth_imbalance_1": depth_imbalance_1,
        "depth_imbalance_5": depth_imbalance_5,
        "bid_replenish_1": bid_replenish,
        "ask_replenish_1": ask_replenish,
        "dollar_volume_z_max": dollar_z_max,
        "odd_lot_ratio": odd_lot_ratio,
        "duplicate_sequence_rate": duplicate_rate,
        "context_score": context_score * 100.0,
        "acc_tape_score": acc_tape * 100.0,
        "acc_book_score": acc_book * 100.0,
        "acc_impact_score": acc_impact * 100.0,
        "dist_tape_score": dist_tape * 100.0,
        "dist_book_score": dist_book * 100.0,
        "dist_impact_score": dist_impact * 100.0,
        "accumulation_score": accumulation_score,
        "distribution_score": distribution_score,
        "acc_evidence_blocks": sum(acc_blocks.values()),
        "dist_evidence_blocks": sum(dist_blocks.values()),
        "acc_reason": _reason(
            [
                ("positive active tape", acc_blocks["tape"]),
                ("supportive bid/depth absorption", acc_blocks["book"]),
                ("price holds near/above VWAP with controlled impact", acc_blocks["impact"]),
            ]
        ),
        "dist_reason": _reason(
            [
                ("negative active tape", dist_blocks["tape"]),
                ("weak bid/depth or ask replenishment", dist_blocks["book"]),
                ("price below VWAP with controlled selling pressure", dist_blocks["impact"]),
            ]
        ),
    }


def _attach_side(row: pd.Series, cfg: MicrostructureSignalConfig, gate: dict[str, Any]) -> dict[str, Any]:
    acc = _finite(row.get("accumulation_score"), 0.0)
    dist = _finite(row.get("distribution_score"), 0.0)
    if acc >= dist:
        side = "accumulation"
        side_score = acc
        evidence_blocks = int(row.get("acc_evidence_blocks") or 0)
        reason = str(row.get("acc_reason") or "")
    else:
        side = "distribution"
        side_score = dist
        evidence_blocks = int(row.get("dist_evidence_blocks") or 0)
        reason = str(row.get("dist_reason") or "")

    has_data_quality = (
        _finite(row.get("coverage_ratio_regular"), 0.0) >= cfg.min_data_coverage
        and _finite(row.get("trade_coverage_ratio_regular"), 0.0) >= cfg.min_data_coverage
        and _finite(row.get("book_coverage_ratio_regular"), 0.0) >= cfg.min_data_coverage
        and int(row.get("trade_count") or 0) >= cfg.min_trade_count
        and _finite(row.get("dollar_volume"), 0.0) >= cfg.min_dollar_volume
        and _finite(row.get("spread_bps"), cfg.max_spread_bps) <= cfg.max_spread_bps
        and _finite(row.get("duplicate_sequence_rate"), 0.0) < 0.01
    )
    gate_validated = bool(gate.get("validated"))
    validated_sides = gate.get("validated_sides")
    if isinstance(validated_sides, dict):
        side_validated = bool(validated_sides.get(side, False))
    else:
        side_validated = gate_validated
    if side_score >= cfg.high_score and evidence_blocks >= 2 and has_data_quality and side_validated:
        confidence = "high"
        report_state = "validated"
    elif side_score >= cfg.watch_score and evidence_blocks >= 2:
        confidence = "watch"
        report_state = "warmup" if not side_validated else "validated"
    else:
        confidence = "diagnostic"
        report_state = "warmup" if not side_validated else "validated"

    stage = {
        ("accumulation", "high"): "stealth_accumulation",
        ("accumulation", "watch"): "accumulation_watch",
        ("accumulation", "diagnostic"): "accumulation_diagnostic",
        ("distribution", "high"): "distribution_risk",
        ("distribution", "watch"): "distribution_watch",
        ("distribution", "diagnostic"): "distribution_diagnostic",
    }[(side, confidence)]

    payload = row.to_dict()
    payload.update(
        {
            "side": side,
            "side_score": side_score,
            "evidence_blocks": evidence_blocks,
            "reason": reason,
            "stage": stage,
            "confidence": confidence,
            "report_state": report_state,
            "validation_state": str(gate.get("state") or "warmup"),
            "validation_reason": str(gate.get("reason") or ""),
            "data_quality_pass": bool(has_data_quality),
        }
    )
    return payload


def score_microstructure_signals(
    features: pd.DataFrame,
    *,
    config: MicrostructureSignalConfig | None = None,
    validation_gate: dict[str, Any] | None = None,
    include_diagnostic: bool = True,
) -> pd.DataFrame:
    """Score symbol-level accumulation/distribution candidates."""

    cfg = config or MicrostructureSignalConfig()
    gate = validation_gate or {"state": "warmup", "validated": False, "reason": "validation gate not configured"}
    if features.empty:
        return pd.DataFrame()
    if "is_regular_session" in features.columns:
        features = features[features["is_regular_session"].fillna(False)].copy()
        if features.empty:
            return pd.DataFrame()
    rows = []
    for symbol, part in features.groupby("symbol", sort=True):
        if not str(symbol or "").strip():
            continue
        rows.append(_summarize_symbol(str(symbol), part, cfg))
    if not rows:
        return pd.DataFrame()
    scored = pd.DataFrame(rows)
    attached = pd.DataFrame([_attach_side(row, cfg, gate) for _, row in scored.iterrows()])
    if not include_diagnostic:
        attached = attached[attached["side_score"] >= cfg.watch_score].copy()
    attached = attached.sort_values(["confidence", "side_score"], ascending=[True, False]).reset_index(drop=True)
    confidence_order = {"high": 0, "watch": 1, "diagnostic": 2}
    attached["_confidence_order"] = attached["confidence"].map(confidence_order).fillna(9)
    attached = attached.sort_values(["_confidence_order", "side_score"], ascending=[True, False]).drop(columns="_confidence_order")
    attached["rank"] = np.arange(1, len(attached) + 1)
    return attached.reset_index(drop=True)
