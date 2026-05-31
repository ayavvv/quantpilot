"""Market-wide major-money digest helpers.

The digest deliberately treats "major money" as a vendor/proxy field.  It can
summarise A-share Eastmoney fund-flow ranks and Futu capital-flow snapshots for
US/HK/A-share markets, while keeping coverage explicit in the output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


MARKET_CURRENCIES = {
    "A": "CNY",
    "SH": "CNY",
    "SZ": "CNY",
    "HK": "HKD",
    "US": "USD",
}


@dataclass(frozen=True)
class FlowThresholds:
    entry_amount: float
    exit_amount: float
    watch_entry_amount: float
    watch_exit_amount: float


DEFAULT_THRESHOLDS = {
    "A": FlowThresholds(
        entry_amount=50_000_000.0,
        exit_amount=-50_000_000.0,
        watch_entry_amount=10_000_000.0,
        watch_exit_amount=-10_000_000.0,
    ),
    "HK": FlowThresholds(
        entry_amount=50_000_000.0,
        exit_amount=-50_000_000.0,
        watch_entry_amount=10_000_000.0,
        watch_exit_amount=-10_000_000.0,
    ),
    "US": FlowThresholds(
        entry_amount=20_000_000.0,
        exit_amount=-20_000_000.0,
        watch_entry_amount=5_000_000.0,
        watch_exit_amount=-5_000_000.0,
    ),
}


def normalize_market(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"CN", "ASHARE", "A_SHARE", "A-SHARE", "CHINA"}:
        return "A"
    return normalized


def market_currency(market: str) -> str:
    normalized = normalize_market(market)
    return MARKET_CURRENCIES.get(normalized, MARKET_CURRENCIES.get(normalized[:2], ""))


def thresholds_for_market(market: str) -> FlowThresholds:
    normalized = normalize_market(market)
    if normalized in {"SH", "SZ"}:
        normalized = "A"
    return DEFAULT_THRESHOLDS.get(normalized, DEFAULT_THRESHOLDS["A"])


def _number(value: Any) -> float | None:
    if value in (None, "", "N/A", "-", "nan"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def _string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _first_present(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if value is not None and not pd.isna(value):
                return value
    return None


def _date_from_values(values: list[Any], fallback: str = "") -> str:
    candidates: list[str] = []
    for value in values:
        text = _string(value).strip()
        if not text:
            continue
        if len(text) >= 10 and text[:4].isdigit():
            candidates.append(text[:10])
    return sorted(candidates)[-1] if candidates else fallback


def normalize_flow_frame(
    df: pd.DataFrame,
    *,
    market: str,
    source: str,
    snapshot_date: str = "",
) -> pd.DataFrame:
    """Normalize a vendor-specific flow frame into the digest row contract."""

    normalized_market = normalize_market(market)
    currency = market_currency(normalized_market)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        status = _string(_first_present(row, ["capital_flow_status", "fund_flow_status"])).strip()
        if not status:
            status = "ok"
        main_flow = _number(
            _first_present(
                row,
                [
                    "latest_main_in_flow",
                    "main_net_inflow",
                    "main_in_flow",
                    "distribution_net_main",
                    "net_main",
                ],
            )
        )
        super_flow = _number(
            _first_present(row, ["latest_super_in_flow", "super_net_inflow", "super_in_flow", "distribution_net_super"])
        )
        big_flow = _number(
            _first_present(row, ["latest_big_in_flow", "big_net_inflow", "big_in_flow", "distribution_net_big"])
        )
        flow_date = _date_from_values(
            [
                _first_present(row, ["capital_flow_latest_date", "date", "flow_date"]),
                _first_present(row, ["update_time", "distribution_update_time"]),
            ],
            fallback=snapshot_date,
        )
        records.append(
            {
                "market": normalized_market,
                "currency": currency,
                "source": source,
                "code": _string(row.get("code")).strip(),
                "name": _string(row.get("name")).strip(),
                "flow_date": flow_date,
                "status": status,
                "main_flow": main_flow,
                "super_flow": super_flow,
                "big_flow": big_flow,
                "main_3d_sum": _number(row.get("main_3d_sum")),
                "main_5d_sum": _number(row.get("main_5d_sum")),
                "main_10d_sum": _number(row.get("main_10d_sum")),
                "main_positive_5d": _number(row.get("main_positive_5d")),
                "rank": _number(_first_present(row, ["fund_flow_rank", "rank", "model_rank"])),
                "latest_price": _number(row.get("latest_price")),
                "change_pct": _number(row.get("change_pct")),
                "exchange_type": _string(row.get("exchange_type")).strip(),
                "error": _string(_first_present(row, ["capital_flow_error", "fund_flow_error"])).strip(),
            }
        )
    return pd.DataFrame(records)


def classify_flow(value: float | None, thresholds: FlowThresholds) -> str:
    if value is None:
        return "missing"
    if value >= thresholds.entry_amount:
        return "major_entry"
    if value <= thresholds.exit_amount:
        return "major_exit"
    if value >= thresholds.watch_entry_amount:
        return "watch_entry"
    if value <= thresholds.watch_exit_amount:
        return "watch_exit"
    return "neutral"


def classify_flow_frame(df: pd.DataFrame, *, thresholds: FlowThresholds) -> pd.DataFrame:
    if df.empty:
        result = df.copy()
        result["flow_label"] = []
        return result
    result = df.copy()
    result["flow_label"] = result["main_flow"].apply(lambda value: classify_flow(value, thresholds))
    return result


def _latest_date(series: pd.Series) -> str:
    values = sorted(value for value in series.dropna().astype(str).str[:10].unique().tolist() if value)
    return values[-1] if values else ""


def _top_rows(df: pd.DataFrame, *, ascending: bool, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    ranked = df.sort_values("main_flow", ascending=ascending, kind="stable").head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        rows.append(
            {
                "code": _string(row.get("code")),
                "name": _string(row.get("name")),
                "main_flow": _number(row.get("main_flow")),
                "super_flow": _number(row.get("super_flow")),
                "big_flow": _number(row.get("big_flow")),
                "flow_label": _string(row.get("flow_label")),
                "flow_date": _string(row.get("flow_date")),
                "rank": _number(row.get("rank")),
                "change_pct": _number(row.get("change_pct")),
                "exchange_type": _string(row.get("exchange_type")),
            }
        )
    return rows


def _exchange_type_counts(df: pd.DataFrame) -> dict[str, int]:
    if "exchange_type" not in df.columns:
        return {}
    values = df["exchange_type"].fillna("").astype(str).str.strip()
    counts = values[values != ""].value_counts().sort_index()
    return {str(exchange): int(count) for exchange, count in counts.items()}


def unavailable_market_summary(market: str, *, reason: str) -> dict[str, Any]:
    normalized_market = normalize_market(market)
    return {
        "market": normalized_market,
        "source": "",
        "currency": market_currency(normalized_market),
        "available": False,
        "message": reason,
        "flow_date": "",
        "total_rows": 0,
        "ok_rows": 0,
        "error_rows": 0,
        "missing_rows": 0,
        "exchange_types": {},
        "entry_count": 0,
        "entry_amount": 0.0,
        "exit_count": 0,
        "exit_amount": 0.0,
        "watch_entry_count": 0,
        "watch_exit_count": 0,
        "net_amount": 0.0,
        "thresholds": thresholds_for_market(normalized_market).__dict__,
        "top_entries": [],
        "top_exits": [],
    }


def build_market_summary(
    df: pd.DataFrame,
    *,
    market: str,
    source: str,
    top_n: int = 10,
    thresholds: FlowThresholds | None = None,
    snapshot_date: str = "",
) -> dict[str, Any]:
    normalized_market = normalize_market(market)
    thresholds = thresholds or thresholds_for_market(normalized_market)
    rows = normalize_flow_frame(df, market=normalized_market, source=source, snapshot_date=snapshot_date)
    rows = classify_flow_frame(rows, thresholds=thresholds)

    if rows.empty:
        return unavailable_market_summary(normalized_market, reason=f"No rows loaded for {normalized_market}.")

    ok_mask = rows["status"].fillna("").astype(str).str.lower().eq("ok")
    ok_rows = rows[ok_mask].copy()
    numeric_flow = pd.to_numeric(ok_rows["main_flow"], errors="coerce")
    entry_rows = ok_rows[ok_rows["flow_label"] == "major_entry"]
    exit_rows = ok_rows[ok_rows["flow_label"] == "major_exit"]
    watch_entry_rows = ok_rows[ok_rows["flow_label"] == "watch_entry"]
    watch_exit_rows = ok_rows[ok_rows["flow_label"] == "watch_exit"]

    ok_count = int(len(ok_rows))
    available = ok_count > 0
    message = "ok" if available else f"Loaded {len(rows)} row(s), but none had usable major-money flow."

    return {
        "market": normalized_market,
        "source": source,
        "currency": market_currency(normalized_market),
        "available": available,
        "message": message,
        "flow_date": _latest_date(rows["flow_date"]) if "flow_date" in rows.columns else "",
        "total_rows": int(len(rows)),
        "ok_rows": ok_count,
        "error_rows": int((~ok_mask).sum()),
        "missing_rows": int(ok_rows["main_flow"].isna().sum()),
        "exchange_types": _exchange_type_counts(rows),
        "entry_count": int(len(entry_rows)),
        "entry_amount": float(pd.to_numeric(entry_rows["main_flow"], errors="coerce").dropna().sum()),
        "exit_count": int(len(exit_rows)),
        "exit_amount": float(abs(pd.to_numeric(exit_rows["main_flow"], errors="coerce").dropna().sum())),
        "watch_entry_count": int(len(watch_entry_rows)),
        "watch_exit_count": int(len(watch_exit_rows)),
        "net_amount": float(numeric_flow.dropna().sum()),
        "thresholds": thresholds.__dict__,
        "top_entries": _top_rows(entry_rows, ascending=False, limit=top_n),
        "top_exits": _top_rows(exit_rows, ascending=True, limit=top_n),
    }


def build_digest(
    market_summaries: list[dict[str, Any]],
    *,
    expected_markets: list[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    seen = {normalize_market(item.get("market", "")) for item in market_summaries}
    summaries = list(market_summaries)
    for market in expected_markets or []:
        normalized = normalize_market(market)
        if normalized and normalized not in seen:
            summaries.append(
                unavailable_market_summary(
                    normalized,
                    reason=f"No market-wide major-money artifact found for {normalized}.",
                )
            )

    amount_by_currency: dict[str, dict[str, float]] = {}
    for summary in summaries:
        currency = str(summary.get("currency") or "N/A")
        bucket = amount_by_currency.setdefault(currency, {"entry_amount": 0.0, "exit_amount": 0.0, "net_amount": 0.0})
        if summary.get("available"):
            bucket["entry_amount"] += float(summary.get("entry_amount") or 0.0)
            bucket["exit_amount"] += float(summary.get("exit_amount") or 0.0)
            bucket["net_amount"] += float(summary.get("net_amount") or 0.0)

    available = [summary for summary in summaries if summary.get("available")]
    flow_dates = sorted(str(summary.get("flow_date")) for summary in available if summary.get("flow_date"))
    return {
        "generated_at": generated,
        "flow_date": flow_dates[-1] if flow_dates else "",
        "market_count": len(summaries),
        "available_market_count": len(available),
        "entry_count": int(sum(int(summary.get("entry_count") or 0) for summary in available)),
        "exit_count": int(sum(int(summary.get("exit_count") or 0) for summary in available)),
        "amount_by_currency": amount_by_currency,
        "markets": summaries,
    }


def digest_rows(digest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for market in digest.get("markets", []):
        rows.append(
            {
                "market": market.get("market", ""),
                "source": market.get("source", ""),
                "currency": market.get("currency", ""),
                "available": bool(market.get("available")),
                "flow_date": market.get("flow_date", ""),
                "total_rows": market.get("total_rows", 0),
                "ok_rows": market.get("ok_rows", 0),
                "exchange_types": json.dumps(market.get("exchange_types") or {}, ensure_ascii=False, sort_keys=True),
                "entry_count": market.get("entry_count", 0),
                "entry_amount": market.get("entry_amount", 0.0),
                "exit_count": market.get("exit_count", 0),
                "exit_amount": market.get("exit_amount", 0.0),
                "watch_entry_count": market.get("watch_entry_count", 0),
                "watch_exit_count": market.get("watch_exit_count", 0),
                "net_amount": market.get("net_amount", 0.0),
                "message": market.get("message", ""),
            }
        )
    return pd.DataFrame(rows)


def load_source_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser())
