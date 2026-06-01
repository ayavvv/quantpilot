"""Offline validation and filter search for daily-bar major-force signals."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationCriteria:
    """Minimum out-of-sample evidence required before daily reporting."""

    min_train_dates: int = 24
    min_test_dates: int = 12
    min_train_alpha: float = 0.005
    min_test_alpha: float = 0.003
    min_train_hit_rate: float = 0.5
    min_train_win_rate_days: float = 0.5
    min_test_hit_rate: float = 0.53
    min_test_win_rate_days: float = 0.55
    min_recent_dates: int = 12
    min_recent_alpha: float = 0.003
    min_recent_hit_rate: float = 0.53
    min_recent_win_rate_days: float = 0.55
    split_ratio: float = 0.67
    recent_ratio: float = 0.25
    train_candidate_limit_per_side: int = 20


def _num(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _load_eval_rows(eval_dir: Path) -> pd.DataFrame:
    picks_path = eval_dir / "picks.csv"
    daily_path = eval_dir / "daily.csv"
    if not picks_path.exists():
        raise FileNotFoundError(f"missing picks.csv: {picks_path}")
    if not daily_path.exists():
        raise FileNotFoundError(f"missing daily.csv: {daily_path}")
    picks = pd.read_csv(picks_path)
    daily = pd.read_csv(daily_path)
    if picks.empty or daily.empty:
        return pd.DataFrame()

    universe = (
        daily.sort_values(["date", "signal_side", "horizon", "top_n"])
        .drop_duplicates(["date", "signal_side", "horizon"])
        [["date", "signal_side", "horizon", "universe_return"]]
        .rename(columns={"date": "eval_date"})
    )
    rows = []
    for horizon in sorted(pd.to_numeric(daily["horizon"], errors="coerce").dropna().astype(int).unique()):
        ret_col = f"fwd_return_{horizon}d"
        if ret_col not in picks.columns:
            continue
        part = picks[picks[ret_col].notna()].copy()
        if part.empty:
            continue
        part["horizon"] = horizon
        part["fwd_return"] = pd.to_numeric(part[ret_col], errors="coerce")
        part = part.merge(
            universe[universe["horizon"] == horizon],
            on=["eval_date", "signal_side", "horizon"],
            how="left",
        )
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    string_cols = {"code", "date", "eval_date", "signal_side", "stage", "reason"}
    for col in result.columns:
        if col not in string_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _filter_rows(rows: pd.DataFrame, rule: dict[str, object]) -> pd.DataFrame:
    side = str(rule.get("side", "")).lower()
    result = rows[(rows["signal_side"].astype(str).str.lower() == side) & (rows["horizon"] == int(rule["horizon"]))].copy()
    result = result[result["rank"] <= int(rule["rank_n"])]
    result = result[result["side_score"] >= float(rule["min_score"])]
    stages = rule.get("stages")
    if isinstance(stages, list) and stages and "stage" in result.columns:
        result = result[result["stage"].astype(str).isin({str(value) for value in stages})]
    if _num(rule.get("min_amount_ratio_5_20")) > 0:
        result = result[result["amount_ratio_5_20"] >= float(rule["min_amount_ratio_5_20"])]

    if side == "buy":
        if "min_cmf_20" in rule:
            result = result[result["cmf_20"] >= float(rule["min_cmf_20"])]
        if "min_close_location_10" in rule:
            result = result[result["close_location_10"] >= float(rule["min_close_location_10"])]
        if "min_breakout_20" in rule:
            result = result[result["breakout_20"] >= float(rule["min_breakout_20"])]
    elif side == "sell":
        if "max_cmf_20" in rule:
            result = result[result["cmf_20"] <= float(rule["max_cmf_20"])]
        if "max_close_location_10" in rule:
            result = result[result["close_location_10"] <= float(rule["max_close_location_10"])]
        if "max_breakout_20" in rule:
            result = result[result["breakout_20"] <= float(rule["max_breakout_20"])]

    skip_keys = {"min_score"}
    for key, value in rule.items():
        if key in skip_keys:
            continue
        if key.startswith("min_"):
            field = key[4:]
            if field in result.columns:
                result = result[pd.to_numeric(result[field], errors="coerce") >= float(value)]
        elif key.startswith("max_"):
            field = key[4:]
            if field in result.columns:
                result = result[pd.to_numeric(result[field], errors="coerce") <= float(value)]
    return result


def _evaluate_filtered(rows: pd.DataFrame, side: str) -> dict[str, float | int]:
    if rows.empty:
        return {
            "date_count": 0,
            "avg_selected_count": 0.0,
            "avg_return": 0.0,
            "avg_universe_return": 0.0,
            "avg_alpha": 0.0,
            "avg_hit_rate": 0.0,
            "win_rate_days": 0.0,
        }
    grouped = rows.groupby("eval_date", sort=True)
    selected_return = grouped["fwd_return"].mean()
    universe_return = grouped["universe_return"].first().reindex(selected_return.index)
    if side == "sell":
        alpha = universe_return - selected_return
        hit_rate = float((rows["fwd_return"] < 0).mean())
        win_rate_days = float((selected_return < 0).mean())
    else:
        alpha = selected_return - universe_return
        hit_rate = float((rows["fwd_return"] > 0).mean())
        win_rate_days = float((selected_return > 0).mean())
    return {
        "date_count": int(selected_return.shape[0]),
        "avg_selected_count": float(grouped.size().mean()),
        "avg_return": float(selected_return.mean()),
        "avg_universe_return": float(universe_return.mean()),
        "avg_alpha": float(alpha.mean()),
        "avg_hit_rate": hit_rate,
        "win_rate_days": win_rate_days,
    }


def _split_dates(rows: pd.DataFrame, split_ratio: float) -> tuple[set[str], set[str]]:
    dates = sorted(str(value) for value in rows["eval_date"].dropna().unique())
    if not dates:
        return set(), set()
    split_pos = max(1, min(len(dates) - 1, int(len(dates) * split_ratio)))
    return set(dates[:split_pos]), set(dates[split_pos:])


def _tail_dates(rows: pd.DataFrame, ratio: float) -> set[str]:
    dates = sorted(str(value) for value in rows["eval_date"].dropna().unique())
    if not dates:
        return set()
    count = max(1, int(len(dates) * ratio))
    return set(dates[-count:])


def _train_score(metrics: dict[str, float | int]) -> float:
    return (
        float(metrics["avg_alpha"])
        + float(metrics["avg_hit_rate"]) / 100.0
        + float(metrics["win_rate_days"]) / 100.0
        + min(float(metrics["date_count"]), 100.0) / 100_000.0
    )


def _rule_key(rule: dict[str, object]) -> str:
    return json.dumps(rule, sort_keys=True, separators=(",", ":"))


def _extra_filter_available(extra: dict[str, object], columns: set[str]) -> bool:
    for key in extra:
        if key == "stages":
            if "stage" not in columns:
                return False
        elif key.startswith(("min_", "max_")):
            if key[4:] not in columns:
                return False
    return True


BUY_STAGE_FILTER_SETS: tuple[dict[str, object], ...] = (
    {"stages": ["stealth_accumulation"]},
    {"stages": ["stealth_accumulation", "accumulation_candidate"]},
)

BUY_MARKET_FILTER_SETS: tuple[dict[str, object], ...] = (
    {"min_market_above_ma20_rate": 0.45, "min_market_positive_rate_20": 0.50},
    {"min_market_drawdown_20": -0.005},
    {"min_market_return_20": -0.01, "max_market_return_20": 0.03},
)

BUY_STOCK_FILTER_SETS: tuple[dict[str, object], ...] = (
    {"min_cmf_20": 0.08},
    {"min_cmf_20": 0.12},
    {"min_cmf_20": 0.08, "max_price_change_20": 0.08},
    {"min_cmf_20": 0.08, "max_amount_ratio_5_20": 2.0},
    {"min_close_location_10": 0.55},
    {"min_close_location_10": 0.50, "max_close_location_10": 0.85, "max_price_change_20": 0.08},
    {"min_breakout_20": 0.0},
    {"max_price_change_20": 0.08},
    {"min_price_change_20": -0.08, "max_price_change_20": 0.08},
    {"max_today_chg_pct": 3.0},
    {"max_amount_ratio_5_20": 2.0},
    {"max_volatility_20": 0.026},
    {"min_positive_flow_days_20": 0.55},
)

BUY_FOCUSED_MARKET_FILTER_SETS: tuple[dict[str, object], ...] = (
    {"min_market_drawdown_20": -0.005},
    {"min_market_return_20": -0.01, "max_market_return_20": 0.03},
    {"min_market_above_ma20_rate": 0.45, "min_market_positive_rate_20": 0.50},
)

BUY_FOCUSED_STOCK_FILTER_SETS: tuple[dict[str, object], ...] = (
    {"min_cmf_20": 0.08, "max_price_change_20": 0.08},
    {"min_cmf_20": 0.08, "max_amount_ratio_5_20": 2.0},
    {"max_price_change_20": 0.08},
    {"min_close_location_10": 0.50, "max_close_location_10": 0.85, "max_price_change_20": 0.08},
)

SELL_FILTER_SETS: tuple[dict[str, object], ...] = (
    {},
    {"stages": ["distribution_risk"]},
    {"max_market_return_20": 0.02, "max_market_above_ma20_rate": 0.55},
)


def _merge_filters(*extras: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for extra in extras:
        merged.update(extra)
    return merged


def _rule_has_valid_bounds(rule: dict[str, object]) -> bool:
    fields = {key[4:] for key in rule if key.startswith(("min_", "max_")) and key != "min_score"}
    for field in fields:
        min_key = f"min_{field}"
        max_key = f"max_{field}"
        if min_key in rule and max_key in rule and float(rule[min_key]) > float(rule[max_key]):
            return False
    return True


def _buy_extra_filter_sets() -> Iterable[dict[str, object]]:
    yield {}
    yield from BUY_STAGE_FILTER_SETS
    yield from BUY_MARKET_FILTER_SETS
    yield from BUY_STOCK_FILTER_SETS
    for market_filter in BUY_FOCUSED_MARKET_FILTER_SETS:
        for stock_filter in BUY_FOCUSED_STOCK_FILTER_SETS:
            yield _merge_filters(market_filter, stock_filter)


def _rule_variants(rule: dict[str, object], side: str, columns: set[str]) -> Iterable[dict[str, object]]:
    if side == "buy":
        extras = _buy_extra_filter_sets()
    else:
        extras = SELL_FILTER_SETS
    for extra in extras:
        if not _extra_filter_available(extra, columns):
            continue
        candidate = dict(rule)
        candidate.update(extra)
        if not _rule_has_valid_bounds(candidate):
            continue
        yield candidate


def _rule_candidates(rows: pd.DataFrame) -> Iterable[dict[str, object]]:
    horizons = sorted(pd.to_numeric(rows["horizon"], errors="coerce").dropna().astype(int).unique())
    columns = set(rows.columns)
    seen: set[str] = set()
    for horizon in horizons:
        for side in ["buy", "sell"]:
            for rank_n in [5, 10, 20, 50, 100, 200]:
                min_scores = [80, 88, 92]
                for min_score in min_scores:
                    for min_amount_ratio in [0.0, 1.2]:
                        base = {
                            "side": side,
                            "horizon": int(horizon),
                            "rank_n": rank_n,
                            "min_score": float(min_score),
                            "min_amount_ratio_5_20": float(min_amount_ratio),
                        }
                        if side == "buy":
                            for candidate in _rule_variants(base, side, columns):
                                key = _rule_key(candidate)
                                if key not in seen:
                                    seen.add(key)
                                    yield candidate
                        else:
                            for max_cmf in [None, -0.12, -0.2]:
                                for max_loc in [None, 0.45, 0.35]:
                                    for max_breakout in [None, -0.03]:
                                        rule = dict(base)
                                        if max_cmf is not None:
                                            rule["max_cmf_20"] = float(max_cmf)
                                        if max_loc is not None:
                                            rule["max_close_location_10"] = float(max_loc)
                                        if max_breakout is not None:
                                            rule["max_breakout_20"] = float(max_breakout)
                                        for candidate in _rule_variants(rule, side, columns):
                                            key = _rule_key(candidate)
                                            if key not in seen:
                                                seen.add(key)
                                                yield candidate


def validate_major_force_eval(
    eval_dir: str | Path,
    *,
    criteria: ValidationCriteria | None = None,
    max_rules_per_side: int = 1,
) -> dict[str, object]:
    """Search for validated daily-bar major-force reporting rules."""

    cfg = criteria or ValidationCriteria()
    rows = _load_eval_rows(Path(eval_dir).expanduser())
    if rows.empty:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "validated": False,
            "criteria": asdict(cfg),
            "rules": [],
            "best_rules": [],
            "message": "No eval rows available.",
        }

    train_dates, test_dates = _split_dates(rows, cfg.split_ratio)
    recent_dates = _tail_dates(rows, cfg.recent_ratio)
    train_rows = rows[rows["eval_date"].astype(str).isin(train_dates)]
    test_rows = rows[rows["eval_date"].astype(str).isin(test_dates)]
    recent_rows = rows[rows["eval_date"].astype(str).isin(recent_dates)]
    train_passed: list[dict[str, object]] = []
    candidate_rule_count = 0
    candidate_rule_count_by_side = {"buy": 0, "sell": 0}
    train_passed_count_by_side = {"buy": 0, "sell": 0}

    for rule in _rule_candidates(rows):
        candidate_rule_count += 1
        side = str(rule["side"])
        if side in candidate_rule_count_by_side:
            candidate_rule_count_by_side[side] += 1
        train_metrics = _evaluate_filtered(_filter_rows(train_rows, rule), side)
        if train_metrics["date_count"] < cfg.min_train_dates:
            continue
        if train_metrics["avg_alpha"] < cfg.min_train_alpha:
            continue
        if train_metrics["avg_hit_rate"] < cfg.min_train_hit_rate:
            continue
        if train_metrics["win_rate_days"] < cfg.min_train_win_rate_days:
            continue
        test_metrics = _evaluate_filtered(_filter_rows(test_rows, rule), side)
        recent_metrics = _evaluate_filtered(_filter_rows(recent_rows, rule), side)
        record = {
            **rule,
            "train": train_metrics,
            "test": test_metrics,
            "recent": recent_metrics,
            "score": _train_score(train_metrics),
            "test_score": float(test_metrics["avg_alpha"]) + float(test_metrics["avg_hit_rate"]) / 100.0,
        }
        train_passed.append(record)
        if side in train_passed_count_by_side:
            train_passed_count_by_side[side] += 1

    best_rows = sorted(
        train_passed,
        key=lambda item: (
            -float(item["score"]),
            str(item.get("side")),
            -float(item["train"]["avg_alpha"]),
            -float(item["train"]["avg_hit_rate"]),
            -int(item["train"]["date_count"]),
        ),
    )
    best_rows_by_side = {
        side: [
            item
            for item in sorted(
                (row for row in train_passed if str(row.get("side")) == side),
                key=lambda item: (
                    -float(item["score"]),
                    -float(item["train"]["avg_alpha"]),
                    -float(item["train"]["avg_hit_rate"]),
                    -int(item["train"]["date_count"]),
                ),
            )[:10]
        ]
        for side in ["buy", "sell"]
    }
    chosen: list[dict[str, object]] = []
    for side in ["buy", "sell"]:
        side_train_rules = [
            item
            for item in train_passed
            if str(item.get("side")) == side
        ]
        side_train_rules = sorted(
            side_train_rules,
            key=lambda item: (
                -float(item["score"]),
                -float(item["train"]["avg_alpha"]),
                -float(item["train"]["avg_hit_rate"]),
                -int(item["train"]["date_count"]),
            ),
        )[: max(1, cfg.train_candidate_limit_per_side)]
        side_validated = []
        for item in side_train_rules:
            test_metrics = item["test"]
            if (
                test_metrics["date_count"] >= cfg.min_test_dates
                and test_metrics["avg_alpha"] >= cfg.min_test_alpha
                and test_metrics["avg_hit_rate"] >= cfg.min_test_hit_rate
                and test_metrics["win_rate_days"] >= cfg.min_test_win_rate_days
                and item["recent"]["date_count"] >= cfg.min_recent_dates
                and item["recent"]["avg_alpha"] >= cfg.min_recent_alpha
                and item["recent"]["avg_hit_rate"] >= cfg.min_recent_hit_rate
                and item["recent"]["win_rate_days"] >= cfg.min_recent_win_rate_days
            ):
                side_validated.append({**item, "status": "validated"})
        chosen.extend(side_validated[:max_rules_per_side])

    message = (
        f"Validated {len(chosen)} rule(s) with train/test split "
        f"{len(train_dates)}/{len(test_dates)} dates; "
        f"recent robustness window {len(recent_dates)} dates; "
        f"searched {candidate_rule_count} candidate rule(s); "
        f"train-passed buy/sell "
        f"{train_passed_count_by_side['buy']}/{train_passed_count_by_side['sell']}; "
        f"tested top {cfg.train_candidate_limit_per_side} train-selected candidate(s) per side."
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "validated": bool(chosen),
        "criteria": asdict(cfg),
        "train_date_count": len(train_dates),
        "test_date_count": len(test_dates),
        "recent_date_count": len(recent_dates),
        "candidate_rule_count": candidate_rule_count,
        "candidate_rule_count_by_side": candidate_rule_count_by_side,
        "train_passed_count": len(train_passed),
        "train_passed_count_by_side": train_passed_count_by_side,
        "rules": chosen,
        "best_rules": best_rows[:20],
        "best_rules_by_side": best_rows_by_side,
        "message": message,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate major-force eval artifacts and export a reporting gate.")
    parser.add_argument("--eval-dir", default=os.environ.get("MAJOR_FORCE_EVAL_DIR", "~/quantpilot_data/output/major_force_eval"))
    parser.add_argument(
        "--output-json",
        default=os.environ.get("MAJOR_FORCE_VALIDATION_JSON", "~/quantpilot_data/output/major_force_validation.json"),
    )
    parser.add_argument("--min-train-dates", type=int, default=24)
    parser.add_argument("--min-test-dates", type=int, default=12)
    parser.add_argument("--min-train-alpha", type=float, default=0.005)
    parser.add_argument("--min-test-alpha", type=float, default=0.003)
    parser.add_argument("--min-train-hit-rate", type=float, default=0.5)
    parser.add_argument("--min-train-win-rate-days", type=float, default=0.5)
    parser.add_argument("--min-test-hit-rate", type=float, default=0.53)
    parser.add_argument("--min-test-win-rate-days", type=float, default=0.55)
    parser.add_argument("--min-recent-dates", type=int, default=12)
    parser.add_argument("--min-recent-alpha", type=float, default=0.003)
    parser.add_argument("--min-recent-hit-rate", type=float, default=0.53)
    parser.add_argument("--min-recent-win-rate-days", type=float, default=0.55)
    parser.add_argument("--split-ratio", type=float, default=0.67)
    parser.add_argument("--recent-ratio", type=float, default=0.25)
    parser.add_argument("--train-candidate-limit-per-side", type=int, default=20)
    parser.add_argument("--max-rules-per-side", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    criteria = ValidationCriteria(
        min_train_dates=max(1, args.min_train_dates),
        min_test_dates=max(1, args.min_test_dates),
        min_train_alpha=args.min_train_alpha,
        min_test_alpha=args.min_test_alpha,
        min_train_hit_rate=args.min_train_hit_rate,
        min_train_win_rate_days=args.min_train_win_rate_days,
        min_test_hit_rate=args.min_test_hit_rate,
        min_test_win_rate_days=args.min_test_win_rate_days,
        min_recent_dates=max(1, args.min_recent_dates),
        min_recent_alpha=args.min_recent_alpha,
        min_recent_hit_rate=args.min_recent_hit_rate,
        min_recent_win_rate_days=args.min_recent_win_rate_days,
        split_ratio=max(0.1, min(0.9, args.split_ratio)),
        recent_ratio=max(0.05, min(0.5, args.recent_ratio)),
        train_candidate_limit_per_side=max(1, args.train_candidate_limit_per_side),
    )
    payload = validate_major_force_eval(
        args.eval_dir,
        criteria=criteria,
        max_rules_per_side=max(1, args.max_rules_per_side),
    )
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote validation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
