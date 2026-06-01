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

import pandas as pd

from scripts.collect_us_microstructure import _copy_to_nas
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


def _parse_symbols(value: str) -> list[str]:
    return normalize_us_symbols(item for item in str(value or "").split(",") if item.strip())


def _default_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


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


def _raw_counts(inputs: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {kind: int(len(frame)) for kind, frame in inputs.items()}


def _coverage_summary(features: pd.DataFrame) -> dict[str, object]:
    if features.empty:
        return {
            "symbol_count": 0,
            "minute_count": 0,
            "trade_minutes": 0,
            "book_minutes": 0,
            "quote_minutes": 0,
        }
    return {
        "symbol_count": int(features["symbol"].nunique()),
        "minute_count": int(len(features)),
        "trade_minutes": int(features.get("has_trade_data", pd.Series(dtype=bool)).fillna(False).sum()),
        "book_minutes": int(features.get("has_book_data", pd.Series(dtype=bool)).fillna(False).sum()),
        "quote_minutes": int(features.get("has_quote_data", pd.Series(dtype=bool)).fillna(False).sum()),
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


def _data_quality_summary(features: pd.DataFrame, cfg: MicrostructureSignalConfig) -> dict[str, object]:
    if features.empty:
        return {
            "symbol_count": 0,
            "eligible_symbol_count": 0,
            "high_confidence_data_quality_ok": False,
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
        coverage = _last_numeric(part, "coverage_ratio_regular")
        trade_coverage = _last_numeric(part, "trade_coverage_ratio_regular", coverage)
        book_coverage = _last_numeric(part, "book_coverage_ratio_regular", coverage)
        quote_coverage = _last_numeric(part, "quote_coverage_ratio_regular")
        trade_count = int(pd.to_numeric(part.get("trade_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        dollar_volume = float(pd.to_numeric(part.get("dollar_volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        duplicate_rate = _median_numeric(part, "duplicate_sequence_rate")
        spread_bps = _median_numeric(part, "spread_bps", cfg.max_spread_bps)
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
                "dollar_volume": dollar_volume,
                "duplicate_sequence_rate": duplicate_rate,
                "spread_bps": spread_bps,
            }
        )

    ratios = [float(row["coverage_ratio_regular"]) for row in rows]
    trade_ratios = [float(row["trade_coverage_ratio_regular"]) for row in rows]
    book_ratios = [float(row["book_coverage_ratio_regular"]) for row in rows]
    eligible_count = sum(1 for row in rows if row["eligible"])
    return {
        "symbol_count": len(rows),
        "eligible_symbol_count": int(eligible_count),
        "high_confidence_data_quality_ok": eligible_count > 0,
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


def _candidate_view(signals: pd.DataFrame, *, top_n: int, min_score: float) -> pd.DataFrame:
    if signals.empty:
        return signals
    view = signals[signals["side_score"] >= float(min_score)].copy()
    if view.empty:
        view = signals.head(top_n).copy()
    return view.head(top_n)


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


def render_markdown_report(
    *,
    date: str,
    signals: pd.DataFrame,
    features: pd.DataFrame,
    raw_counts: dict[str, int],
    validation_gate: dict,
    data_quality: dict[str, object],
    top_n: int,
    min_score: float,
) -> str:
    view = _candidate_view(signals, top_n=top_n, min_score=min_score)
    coverage = _coverage_summary(features)
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
        f"- Symbols eligible for high-confidence reporting: `{data_quality.get('eligible_symbol_count', 0)}` / `{data_quality.get('symbol_count', 0)}`",
        f"- Median trade/book coverage: `{_pct(data_quality.get('median_trade_coverage_ratio_regular'))}` / `{_pct(data_quality.get('median_book_coverage_ratio_regular'))}`",
        "",
        "## Data Coverage",
        "",
        f"- Raw trade rows: `{raw_counts.get('trades', 0)}`",
        f"- Raw order-book rows: `{raw_counts.get('order_book', 0)}`",
        f"- Raw quote rows: `{raw_counts.get('quotes', 0)}`",
        f"- Symbols with features: `{coverage['symbol_count']}`",
        f"- Feature minutes: `{coverage['minute_count']}`",
        f"- Trade/book/quote minutes: `{coverage['trade_minutes']}` / `{coverage['book_minutes']}` / `{coverage['quote_minutes']}`",
        "",
        "## Candidates",
        "",
        _markdown_table(view),
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


def render_html_report(
    *,
    date: str,
    signals: pd.DataFrame,
    features: pd.DataFrame,
    raw_counts: dict[str, int],
    validation_gate: dict,
    data_quality: dict[str, object],
    top_n: int,
    min_score: float,
) -> str:
    view = _candidate_view(signals, top_n=top_n, min_score=min_score)
    coverage = _coverage_summary(features)
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
<div class="gate"><strong>Data quality gate:</strong> eligible_symbols={data_quality.get('eligible_symbol_count', 0)}/{data_quality.get('symbol_count', 0)}; median trade/book coverage={_pct(data_quality.get('median_trade_coverage_ratio_regular'))}/{_pct(data_quality.get('median_book_coverage_ratio_regular'))}</div>
<h2>Data Coverage</h2>
<p>Raw trades={raw_counts.get('trades', 0)}, order_book={raw_counts.get('order_book', 0)}, quotes={raw_counts.get('quotes', 0)}. Trade/book/quote minutes={coverage['trade_minutes']} / {coverage['book_minutes']} / {coverage['quote_minutes']}.</p>
<h2>Candidates</h2>
{_html_table(view)}
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
) -> dict[str, Path]:
    feature_path = write_feature_table(features, base_dir, date=date)
    signal_dir = base_dir / "signals" / f"date={date}"
    signal_dir.mkdir(parents=True, exist_ok=True)
    signal_csv = signal_dir / "us_major_flow_signals.csv"
    signals.to_csv(signal_csv, index=False)
    latest_csv = base_dir / "signals" / "us_major_flow_signals_latest.csv"
    latest_csv.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(latest_csv, index=False)

    report_dir = base_dir / "reports" / f"date={date}"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "us_microstructure_flow_report.md"
    html_path = report_dir / "us_microstructure_flow_report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")
    latest_html = base_dir / "reports" / "us_microstructure_flow_report_latest.html"
    latest_md = base_dir / "reports" / "us_microstructure_flow_report_latest.md"
    latest_html.parent.mkdir(parents=True, exist_ok=True)
    latest_html.write_text(html_report, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")

    status_path = report_dir / "status.json"
    latest_status = base_dir / "reports" / "us_microstructure_flow_status_latest.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_status.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "features": feature_path,
        "signals": signal_csv,
        "signals_latest": latest_csv,
        "markdown": markdown_path,
        "html": html_path,
        "html_latest": latest_html,
        "markdown_latest": latest_md,
        "status": status_path,
        "status_latest": latest_status,
    }


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir).expanduser()
    symbols = _parse_symbols(args.symbols)
    validation_gate_path = args.validation_gate or str(base_dir / "validation" / "active_gate.json")
    gate = load_validation_gate(validation_gate_path)

    inputs = read_microstructure_inputs(base_dir, date=args.date, symbols=symbols)
    features = compute_microstructure_features(
        inputs["trades"],
        inputs["order_book"],
        inputs["quotes"],
        config=MicrostructureFeatureConfig(book_levels=args.book_levels),
    )
    signal_cfg = MicrostructureSignalConfig()
    signals = score_microstructure_signals(
        features,
        config=signal_cfg,
        validation_gate=gate,
        include_diagnostic=True,
    )
    raw_counts = _raw_counts(inputs)
    data_quality = _data_quality_summary(features, signal_cfg)
    status = {
        "date": args.date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir),
        "raw_counts": raw_counts,
        "coverage": _coverage_summary(features),
        "data_quality": data_quality,
        "signal_count": int(len(signals)),
        "high_count": int((signals.get("confidence", pd.Series(dtype=str)) == "high").sum()) if not signals.empty else 0,
        "watch_count": int((signals.get("confidence", pd.Series(dtype=str)) == "watch").sum()) if not signals.empty else 0,
        "validation_gate": gate,
    }
    markdown = render_markdown_report(
        date=args.date,
        signals=signals,
        features=features,
        raw_counts=raw_counts,
        validation_gate=gate,
        data_quality=data_quality,
        top_n=args.top_n,
        min_score=args.min_score,
    )
    html_report = render_html_report(
        date=args.date,
        signals=signals,
        features=features,
        raw_counts=raw_counts,
        validation_gate=gate,
        data_quality=data_quality,
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
    )
    nas_results = []
    if not args.no_nas_sync:
        nas_results = _sync_outputs(outputs.values(), base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)
    if nas_results:
        status["nas_sync"] = nas_results
        outputs["status"].write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs["status_latest"].write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.send_email:
        from reporter.send_report import send_email

        send_email(
            html_report,
            _subject(signals, gate),
            report_filename=outputs["html"].name,
            report_dir=outputs["html"].parent,
            attachment_paths=[outputs["signals"], outputs["status"]],
        )

    print(f"Wrote features: {outputs['features']}")
    print(f"Wrote signals: {outputs['signals']}")
    print(f"Wrote report: {outputs['html']}")
    print(f"State={gate.get('state')} high={status['high_count']} watch={status['watch_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
