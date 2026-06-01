"""Confidence-readiness summaries for US microstructure major-flow reports."""

from __future__ import annotations

from typing import Any


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _positive_gap(current: float, required: float) -> float:
    return max(0.0, float(required) - float(current))


def _side_gap(
    side: str,
    *,
    gate: dict[str, Any],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    side_metrics = _dict(gate.get("side_metrics"))
    metrics = _dict(side_metrics.get(side))
    validated_sides = _dict(gate.get("validated_sides"))
    side_reasons = _dict(gate.get("side_reasons"))

    observation_count = _int(metrics.get("observation_count"))
    signal_day_count = _int(metrics.get("signal_day_count"))
    avg_alpha = _float(metrics.get("avg_alpha"))
    hit_rate = _float(metrics.get("hit_rate"))
    recent_hit_rate = _float(metrics.get("recent_hit_rate"))
    wilson_lower = _float(metrics.get("wilson_lower"))
    symbol_share = _float(metrics.get("max_symbol_sample_share"), 0.0 if observation_count <= 0 else 1.0)

    min_observations = _int(criteria.get("min_observations_per_side"))
    min_signal_days = _int(criteria.get("min_signal_days_per_side"))
    min_alpha = _float(criteria.get("min_alpha"))
    min_hit_rate = _float(criteria.get("min_hit_rate"))
    min_recent_hit_rate = _float(criteria.get("min_recent_hit_rate"))
    min_wilson_lower = _float(criteria.get("min_wilson_lower"))
    max_symbol_share = _float(criteria.get("max_symbol_sample_share"), 1.0)

    return {
        "side": side,
        "validated": _bool(validated_sides.get(side)),
        "reason": str(side_reasons.get(side) or ""),
        "observation_count": observation_count,
        "min_observations": min_observations,
        "observations_needed": max(0, min_observations - observation_count),
        "signal_day_count": signal_day_count,
        "min_signal_days": min_signal_days,
        "signal_days_needed": max(0, min_signal_days - signal_day_count),
        "avg_alpha": avg_alpha,
        "min_alpha": min_alpha,
        "alpha_gap": _positive_gap(avg_alpha, min_alpha),
        "hit_rate": hit_rate,
        "min_hit_rate": min_hit_rate,
        "hit_rate_gap": _positive_gap(hit_rate, min_hit_rate),
        "recent_hit_rate": recent_hit_rate,
        "min_recent_hit_rate": min_recent_hit_rate,
        "recent_hit_rate_gap": _positive_gap(recent_hit_rate, min_recent_hit_rate),
        "wilson_lower": wilson_lower,
        "min_wilson_lower": min_wilson_lower,
        "wilson_gap": _positive_gap(wilson_lower, min_wilson_lower),
        "max_symbol_sample_share": symbol_share,
        "max_allowed_symbol_sample_share": max_symbol_share,
        "concentration_excess": max(0.0, symbol_share - max_symbol_share),
    }


def build_confidence_gap(
    validation_gate: dict[str, Any] | None,
    *,
    data_quality: dict[str, Any] | None = None,
    validation_eligibility: dict[str, Any] | None = None,
    intraday_replay: dict[str, Any] | None = None,
    manifest_quality: dict[str, Any] | None = None,
    is_final_report: bool = True,
) -> dict[str, Any]:
    """Build a single auditable summary of why high confidence is or is not ready."""

    gate = _dict(validation_gate)
    quality = _dict(data_quality)
    eligibility = _dict(validation_eligibility)
    replay = _dict(intraday_replay)
    manifest = _dict(manifest_quality)
    criteria = _dict(gate.get("criteria"))

    validation_ready = _bool(gate.get("validated"))
    data_quality_ready = _bool(quality.get("high_confidence_data_quality_ok"))
    nas_ready = _bool(manifest.get("ok")) if manifest else _bool(quality.get("nas_upload_complete"))
    final_ready = bool(is_final_report)
    requirements = {
        "validation_gate_validated": validation_ready,
        "data_quality_gate_ready": data_quality_ready,
        "nas_uploads_complete": nas_ready,
        "final_report_complete": final_ready,
    }

    blockers: list[str] = []
    if not validation_ready:
        blockers.append("validation gate is not promoted")
    if not data_quality_ready:
        blockers.append("data-quality gate is not passing")
    if not nas_ready:
        blockers.append("full-session NAS raw uploads are incomplete")
    if not final_ready:
        blockers.append("report is not a final post-close report")

    side_gaps = [
        _side_gap("accumulation", gate=gate, criteria=criteria),
        _side_gap("distribution", gate=gate, criteria=criteria),
    ]

    return {
        "ready": all(requirements.values()),
        "state": str(gate.get("state") or "warmup"),
        "reason": str(gate.get("reason") or ""),
        "requirements": requirements,
        "blockers": blockers,
        "side_gaps": side_gaps,
        "official_event_count": _int(gate.get("event_count")),
        "official_forward_return_count": _int(gate.get("forward_return_count")),
        "shadow_event_count": _int(gate.get("shadow_event_count")),
        "shadow_forward_return_count": _int(gate.get("shadow_forward_return_count")),
        "validation_eligible_count": _int(eligibility.get("validation_eligible_count")),
        "validation_eligible_if_final_count": _int(eligibility.get("validation_eligible_if_final_count")),
        "score_pass_count": _int(eligibility.get("score_pass_count")),
        "data_quality_pass_count": _int(eligibility.get("data_quality_pass_count")),
        "final_report_count": _int(eligibility.get("final_report_count")),
        "cumulative_intraday_replay": {
            "date_count": _int(replay.get("cumulative_date_count")),
            "event_count": _int(replay.get("cumulative_event_count")),
            "quality_event_count": _int(replay.get("cumulative_quality_event_count")),
            "return_count": _int(replay.get("cumulative_return_count")),
            "quality_return_count": _int(replay.get("cumulative_quality_return_count")),
            "metric_count": _int(replay.get("cumulative_metric_count")),
            "first_date": str(replay.get("cumulative_first_date") or ""),
            "last_date": str(replay.get("cumulative_last_date") or ""),
        },
    }
