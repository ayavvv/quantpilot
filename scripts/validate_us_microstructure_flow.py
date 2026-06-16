"""Update the forward-validation ledger for US microstructure flow signals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from scripts.collect_us_microstructure import _copy_to_nas
from strategy.us_microstructure_validation import (
    ForwardValidationConfig,
    build_active_gate,
    build_rule_metrics,
    compute_forward_returns,
    discover_signal_files,
    load_price_history_from_csv,
    load_price_history_from_qlib,
    load_exploration_signal_events,
    load_shadow_signal_events,
    load_signal_events,
    merge_price_history,
    write_validation_outputs,
)


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
DEFAULT_BASE_DIR = Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DATA_DIR / "us_microstructure")))
DEFAULT_QLIB_DIR = Path(os.environ.get("QLIB_DATA_DIR", str(DATA_DIR / "qlib_data")))
DEFAULT_NAS_DIR = "/volume1/docker/quantpilot/us_microstructure"


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate US microstructure major-flow signals.")
    parser.add_argument("--base-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR)))
    parser.add_argument("--start-date", default=os.environ.get("US_MICROSTRUCTURE_VALIDATION_START", ""))
    parser.add_argument("--end-date", default=os.environ.get("US_MICROSTRUCTURE_VALIDATION_END", ""))
    parser.add_argument("--qlib-dir", default=os.environ.get("QLIB_DATA_DIR", str(DEFAULT_QLIB_DIR)))
    parser.add_argument("--price-csv", default=os.environ.get("US_MICROSTRUCTURE_PRICE_CSV", ""))
    parser.add_argument("--benchmark", default=os.environ.get("US_MICROSTRUCTURE_BENCHMARK", "US.SPY"))
    parser.add_argument("--horizons", default=os.environ.get("US_MICROSTRUCTURE_VALIDATION_HORIZONS", "1,3,5"))
    parser.add_argument("--entry-lag-days", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_ENTRY_LAG_DAYS", "1")))
    parser.add_argument("--min-event-score", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_VALIDATION_MIN_SCORE", "70")))
    parser.add_argument(
        "--shadow-min-event-score",
        type=float,
        default=float(os.environ.get("US_MICROSTRUCTURE_SHADOW_VALIDATION_MIN_SCORE", "65")),
    )
    parser.add_argument(
        "--exploration-min-event-score",
        type=float,
        default=float(os.environ.get("US_MICROSTRUCTURE_EXPLORATION_VALIDATION_MIN_SCORE", "50")),
    )
    parser.add_argument(
        "--min-signal-days-per-side",
        type=int,
        default=int(os.environ.get("US_MICROSTRUCTURE_MIN_SIGNAL_DAYS_PER_SIDE", "10")),
    )
    parser.add_argument(
        "--min-observations-per-side",
        type=int,
        default=int(os.environ.get("US_MICROSTRUCTURE_MIN_OBSERVATIONS_PER_SIDE", "40")),
    )
    parser.add_argument("--min-alpha", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_MIN_ALPHA", "0.0075")))
    parser.add_argument("--min-hit-rate", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_MIN_HIT_RATE", "0.58")))
    parser.add_argument(
        "--min-recent-hit-rate",
        type=float,
        default=float(os.environ.get("US_MICROSTRUCTURE_MIN_RECENT_HIT_RATE", "0.55")),
    )
    parser.add_argument("--nas-host", default=os.environ.get("US_MICROSTRUCTURE_NAS_HOST", ""))
    parser.add_argument("--nas-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_DIR", DEFAULT_NAS_DIR))
    parser.add_argument("--no-nas-sync", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir).expanduser()
    default_price_csv = base_dir / "validation" / "prices" / "us_daily_prices.csv"
    price_csv = str(args.price_csv or "")
    if not price_csv and default_price_csv.exists():
        price_csv = str(default_price_csv)
    cfg = ForwardValidationConfig(
        horizons=_parse_int_tuple(args.horizons),
        benchmark=args.benchmark,
        entry_lag_days=args.entry_lag_days,
        min_event_score=args.min_event_score,
        min_signal_days_per_side=args.min_signal_days_per_side,
        min_observations_per_side=args.min_observations_per_side,
        min_alpha=args.min_alpha,
        min_hit_rate=args.min_hit_rate,
        min_recent_hit_rate=args.min_recent_hit_rate,
    )
    signal_files = discover_signal_files(base_dir, start_date=args.start_date, end_date=args.end_date)
    events = load_signal_events(signal_files, min_event_score=cfg.min_event_score)
    shadow_cfg = ForwardValidationConfig(
        horizons=cfg.horizons,
        benchmark=cfg.benchmark,
        entry_lag_days=cfg.entry_lag_days,
        min_event_score=args.shadow_min_event_score,
        promotion_horizon=cfg.promotion_horizon,
        min_signal_days_per_side=cfg.min_signal_days_per_side,
        min_observations_per_side=cfg.min_observations_per_side,
        min_alpha=cfg.min_alpha,
        min_hit_rate=cfg.min_hit_rate,
        min_recent_hit_rate=cfg.min_recent_hit_rate,
        recent_signal_days=cfg.recent_signal_days,
        min_wilson_lower=cfg.min_wilson_lower,
        max_symbol_sample_share=cfg.max_symbol_sample_share,
    )
    shadow_events = load_shadow_signal_events(signal_files, min_event_score=shadow_cfg.min_event_score)
    exploration_cfg = ForwardValidationConfig(
        horizons=cfg.horizons,
        benchmark=cfg.benchmark,
        entry_lag_days=cfg.entry_lag_days,
        min_event_score=args.exploration_min_event_score,
        promotion_horizon=cfg.promotion_horizon,
        min_signal_days_per_side=cfg.min_signal_days_per_side,
        min_observations_per_side=cfg.min_observations_per_side,
        min_alpha=cfg.min_alpha,
        min_hit_rate=cfg.min_hit_rate,
        min_recent_hit_rate=cfg.min_recent_hit_rate,
        recent_signal_days=cfg.recent_signal_days,
        min_wilson_lower=cfg.min_wilson_lower,
        max_symbol_sample_share=cfg.max_symbol_sample_share,
    )
    exploration_events = load_exploration_signal_events(
        signal_files,
        min_event_score=exploration_cfg.min_event_score,
    )
    symbols = {cfg.benchmark}
    if not events.empty:
        symbols.update(events["symbol"].tolist())
    if not shadow_events.empty:
        symbols.update(shadow_events["symbol"].tolist())
    if not exploration_events.empty:
        symbols.update(exploration_events["symbol"].tolist())
    symbols = sorted(symbols)

    csv_prices = load_price_history_from_csv(price_csv) if price_csv else {}
    qlib_prices = load_price_history_from_qlib(args.qlib_dir, symbols)
    prices = merge_price_history(qlib_prices, csv_prices)
    forward_returns = compute_forward_returns(events, prices, config=cfg)
    metrics = build_rule_metrics(forward_returns, config=cfg)
    shadow_forward_returns = compute_forward_returns(shadow_events, prices, config=shadow_cfg)
    shadow_metrics = build_rule_metrics(shadow_forward_returns, config=shadow_cfg)
    exploration_forward_returns = compute_forward_returns(exploration_events, prices, config=exploration_cfg)
    exploration_metrics = build_rule_metrics(exploration_forward_returns, config=exploration_cfg)
    gate = build_active_gate(metrics, config=cfg)
    gate["signal_file_count"] = len(signal_files)
    gate["event_count"] = int(len(events))
    gate["forward_return_count"] = int(len(forward_returns))
    gate["shadow_min_event_score"] = float(shadow_cfg.min_event_score)
    gate["shadow_event_count"] = int(len(shadow_events))
    gate["shadow_forward_return_count"] = int(len(shadow_forward_returns))
    gate["exploration_min_event_score"] = float(exploration_cfg.min_event_score)
    gate["exploration_event_count"] = int(len(exploration_events))
    gate["exploration_forward_return_count"] = int(len(exploration_forward_returns))
    gate["price_symbol_count"] = int(len(prices))
    gate["price_sources"] = {
        "qlib_dir": str(Path(args.qlib_dir).expanduser()),
        "price_csv": str(Path(price_csv).expanduser()) if price_csv else "",
    }

    outputs = write_validation_outputs(
        base_dir,
        events=events,
        forward_returns=forward_returns,
        metrics=metrics,
        gate=gate,
    )
    validation_dir = base_dir / "validation"
    outputs["shadow_signal_events"] = validation_dir / "shadow_signal_events.parquet"
    outputs["shadow_forward_returns"] = validation_dir / "shadow_forward_returns.parquet"
    outputs["shadow_rule_metrics_csv"] = validation_dir / "shadow_rule_metrics.csv"
    outputs["exploration_signal_events"] = validation_dir / "exploration_signal_events.parquet"
    outputs["exploration_forward_returns"] = validation_dir / "exploration_forward_returns.parquet"
    outputs["exploration_rule_metrics_csv"] = validation_dir / "exploration_rule_metrics.csv"
    shadow_events.to_parquet(outputs["shadow_signal_events"], index=False)
    shadow_forward_returns.to_parquet(outputs["shadow_forward_returns"], index=False)
    shadow_metrics.to_csv(outputs["shadow_rule_metrics_csv"], index=False)
    exploration_events.to_parquet(outputs["exploration_signal_events"], index=False)
    exploration_forward_returns.to_parquet(outputs["exploration_forward_returns"], index=False)
    exploration_metrics.to_csv(outputs["exploration_rule_metrics_csv"], index=False)
    if not args.no_nas_sync:
        sync_results = _sync_outputs(list(outputs.values()), base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)
        gate["nas_sync"] = sync_results
        outputs["active_gate"].write_text(
            json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.nas_host and args.nas_dir:
            _copy_to_nas(outputs["active_gate"], base_dir, args.nas_host, args.nas_dir)

    print(f"Validated signal files: {len(signal_files)}")
    print(f"Events: {len(events)}")
    print(f"Forward returns: {len(forward_returns)}")
    print(f"Shadow events: {len(shadow_events)}")
    print(f"Shadow forward returns: {len(shadow_forward_returns)}")
    print(f"Exploration events: {len(exploration_events)}")
    print(f"Exploration forward returns: {len(exploration_forward_returns)}")
    print(f"Gate: state={gate.get('state')} validated={gate.get('validated')} reason={gate.get('reason')}")
    print(f"Wrote gate: {outputs['active_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
