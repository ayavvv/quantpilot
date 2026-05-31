"""Eastmoney A-share fund-flow helpers.

The public Eastmoney endpoints back AKShare's fund-flow interfaces. They are a
practical fallback when Futu OpenD is unavailable, but should still be treated as
derived vendor data rather than raw tick-by-tick transactions.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


EASTMONEY_UT = "b2884a393a59ad64002292a3e90d46a5"
RANK_ENDPOINT = "https://push2.eastmoney.com/api/qt/clist/get"
HISTORY_ENDPOINT = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
DATACENTER_ENDPOINT = "https://datacenter.eastmoney.com/securities/api/data/get"
A_SHARE_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
DATACENTER_EXTRA_COLS = (
    'f2|02|SECURITY_CODE|NEW_PRICE|(NEW_PRICE<>"-"),'
    "f3|02|SECURITY_CODE|CHANGE_RATE,"
    "MAIN_NETINFLOW|02|SECURITY_CODE|MAIN_NETINFLOW"
)

RANK_FIELDS = {
    "f12": "raw_code",
    "f14": "name",
    "f2": "latest_price",
    "f3": "change_pct",
    "f62": "main_net_inflow",
    "f184": "main_net_inflow_pct",
    "f66": "super_net_inflow",
    "f69": "super_net_inflow_pct",
    "f72": "big_net_inflow",
    "f75": "big_net_inflow_pct",
    "f78": "mid_net_inflow",
    "f81": "mid_net_inflow_pct",
    "f84": "small_net_inflow",
    "f87": "small_net_inflow_pct",
    "f124": "update_timestamp",
}

HISTORY_COLUMNS = [
    "date",
    "main_net_inflow",
    "super_net_inflow",
    "big_net_inflow",
    "mid_net_inflow",
    "small_net_inflow",
    "main_net_inflow_pct",
    "super_net_inflow_pct",
    "big_net_inflow_pct",
    "mid_net_inflow_pct",
    "small_net_inflow_pct",
    "close",
    "change_pct",
]


def _http_json(url: str, params: dict[str, Any], timeout: float = 10.0, retries: int = 3) -> dict:
    query = urllib.parse.urlencode(params)
    last_error = None
    for attempt in range(max(retries, 1)):
        try:
            request = urllib.request.Request(
                f"{url}?{query}",
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json,text/plain,*/*",
                    "Referer": "https://data.eastmoney.com/zjlx/detail.html",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8")
                if "(" in text and text.rstrip().endswith((")", ");")):
                    text = text.split("(", 1)[1].rsplit(")", 1)[0].rstrip(";")
                return json.loads(text)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise last_error


def _to_float(value: Any) -> float | None:
    if value in (None, "-", ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def _to_prefixed_code(raw_code: str) -> str:
    code = str(raw_code).zfill(6)
    if code.startswith(("6", "9")):
        return f"SH.{code}"
    if code.startswith(("0", "2", "3")):
        return f"SZ.{code}"
    if code.startswith(("4", "8")):
        return f"BJ.{code}"
    return code


def _to_prefixed_code_from_secucode(secucode: Any, raw_code: Any) -> str:
    value = str(secucode or "").upper()
    code = str(raw_code or value.split(".", 1)[0]).zfill(6)
    if value.endswith(".SH"):
        return f"SH.{code}"
    if value.endswith(".SZ"):
        return f"SZ.{code}"
    if value.endswith(".BJ"):
        return f"BJ.{code}"
    return _to_prefixed_code(code)


def _to_secid(code: str) -> str:
    normalized = str(code).upper().strip()
    if normalized.startswith("SH."):
        return f"1.{normalized.split('.', 1)[1]}"
    if normalized.startswith(("SZ.", "BJ.")):
        return f"0.{normalized.split('.', 1)[1]}"
    raw = normalized[-6:]
    market = "1" if raw.startswith(("6", "9")) else "0"
    return f"{market}.{raw}"


def _timestamp_to_datetime(value: Any) -> str:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return ""
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def _parse_rank_rows(rows: list[dict[str, Any]], start_rank: int = 1) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows, start=start_rank):
        record = {"fund_flow_rank": idx, "fund_flow_source": "eastmoney_push2"}
        for raw_field, normalized in RANK_FIELDS.items():
            value = row.get(raw_field)
            if normalized == "raw_code":
                record["raw_code"] = str(value).zfill(6)
                record["code"] = _to_prefixed_code(record["raw_code"])
            elif normalized == "name":
                record[normalized] = value
            elif normalized == "update_timestamp":
                record[normalized] = value
                record["update_time"] = _timestamp_to_datetime(value)
            else:
                record[normalized] = _to_float(value)
        records.append(record)
    return records


def _fetch_push2_fund_flow_rank(limit: int = 5000, timeout: float = 10.0, page_size: int = 100) -> pd.DataFrame:
    """Fetch latest A-share individual fund-flow ranking from the richer push2 endpoint."""

    max_rows = max(limit, 1)
    pz = min(max(page_size, 1), 100)
    max_pages = math.ceil(max_rows / pz)
    records: list[dict[str, Any]] = []
    total = None
    for page in range(1, max_pages + 1):
        params = {
            "pn": str(page),
            "pz": str(pz),
            "po": "1",
            "np": "1",
            "fltt": "2",
            "invt": "2",
            "fid": "f62",
            "fs": A_SHARE_FS,
            "fields": ",".join(RANK_FIELDS.keys()),
            "ut": EASTMONEY_UT,
        }
        payload = _http_json(RANK_ENDPOINT, params, timeout=timeout)
        data = payload.get("data") or {}
        total = data.get("total") if total is None else total
        rows = data.get("diff") or []
        if not rows:
            break
        records.extend(_parse_rank_rows(rows, start_rank=len(records) + 1))
        if len(records) >= max_rows:
            break
        if total is not None and len(records) >= int(total):
            break
        time.sleep(0.1)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records[:max_rows])


def _parse_datacenter_rows(rows: list[dict[str, Any]], start_rank: int = 1) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows, start=start_rank):
        raw_code = str(row.get("SECURITY_CODE") or "").zfill(6)
        records.append(
            {
                "fund_flow_rank": idx,
                "fund_flow_source": "eastmoney_datacenter",
                "raw_code": raw_code,
                "code": _to_prefixed_code_from_secucode(row.get("SECUCODE"), raw_code),
                "name": row.get("SECURITY_NAME_ABBR"),
                "latest_price": _to_float(row.get("NEW_PRICE")),
                "change_pct": _to_float(row.get("CHANGE_RATE")),
                "main_net_inflow": _to_float(row.get("MAIN_NETINFLOW")),
                "main_net_inflow_pct": None,
                "super_net_inflow": None,
                "super_net_inflow_pct": None,
                "big_net_inflow": None,
                "big_net_inflow_pct": None,
                "mid_net_inflow": None,
                "mid_net_inflow_pct": None,
                "small_net_inflow": None,
                "small_net_inflow_pct": None,
                "update_timestamp": None,
                "update_time": "",
            }
        )
    return records


def fetch_datacenter_fund_flow_rank(limit: int = 5000, timeout: float = 10.0, page_size: int = 500) -> pd.DataFrame:
    """Fetch latest A-share main-fund-flow ranking from Eastmoney's data-center API."""

    max_rows = max(limit, 1)
    pz = min(max(page_size, 1), 500)
    max_pages = math.ceil(max_rows / pz)
    records: list[dict[str, Any]] = []
    total = None
    for page in range(1, max_pages + 1):
        params = {
            "type": "RPT_FUNDFLOW_SECUCODE",
            "sty": "ALL",
            "source": "SECURITIES",
            "client": "WAP",
            "p": str(page),
            "ps": str(pz),
            "sr": "-1",
            "st": "MAIN_NETINFLOW",
            "extraCols": DATACENTER_EXTRA_COLS,
        }
        payload = _http_json(DATACENTER_ENDPOINT, params, timeout=timeout)
        result = payload.get("result") or {}
        total = result.get("count") if total is None else total
        rows = result.get("data") or []
        if not rows:
            break
        records.extend(_parse_datacenter_rows(rows, start_rank=len(records) + 1))
        if len(records) >= max_rows:
            break
        if total is not None and len(records) >= int(total):
            break
        time.sleep(0.1)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records[:max_rows])


def fetch_fund_flow_rank(
    limit: int = 5000,
    timeout: float = 10.0,
    page_size: int = 100,
    source: str = "auto",
) -> pd.DataFrame:
    """Fetch latest A-share individual fund-flow ranking from Eastmoney."""

    normalized_source = source.lower()
    if normalized_source == "push2":
        return _fetch_push2_fund_flow_rank(limit=limit, timeout=timeout, page_size=page_size)
    if normalized_source == "datacenter":
        return fetch_datacenter_fund_flow_rank(limit=limit, timeout=timeout)
    if normalized_source != "auto":
        raise ValueError("source must be one of: auto, push2, datacenter")

    try:
        return _fetch_push2_fund_flow_rank(limit=limit, timeout=timeout, page_size=page_size)
    except Exception as push2_error:
        fallback = fetch_datacenter_fund_flow_rank(limit=limit, timeout=timeout)
        if not fallback.empty:
            return fallback
        raise push2_error


def fetch_individual_fund_flow(code: str, limit: int = 100, timeout: float = 10.0) -> pd.DataFrame:
    """Fetch recent daily fund-flow history for one A-share symbol."""

    params = {
        "lmt": str(max(limit, 1)),
        "klt": "101",
        "secid": _to_secid(code),
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
        "ut": EASTMONEY_UT,
    }
    payload = _http_json(HISTORY_ENDPOINT, params, timeout=timeout)
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    rows = []
    for line in klines:
        values = str(line).split(",")
        if len(values) < len(HISTORY_COLUMNS):
            continue
        row = {"code": _to_prefixed_code(data.get("code", code[-6:])), "name": data.get("name", "")}
        for column, value in zip(HISTORY_COLUMNS, values):
            row[column] = value if column == "date" else _to_float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def save_fund_flow_rank(
    path: str | Path,
    limit: int = 5000,
    timeout: float = 10.0,
    source: str = "auto",
) -> pd.DataFrame:
    df = fetch_fund_flow_rank(limit=limit, timeout=timeout, source=source)
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df
