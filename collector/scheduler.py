"""Scheduler - cron job orchestration for data collection."""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import pandas as pd
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from collector.config import settings
from collector.futu_client import FutuClient
from collector.db_engine import DBEngine
from collector.yf_client import YFinanceClient
from collector.baostock_client import BaostockClient
from strategy.stock_filter import (
    A_SHARE_ST_METADATA as A_SHARE_ST_METADATA_NAME,
    build_a_share_stock_basic_metadata,
)


class DataCollectorScheduler:
    """Data collection scheduler."""

    A_SHARE_SYNC_STATUS_METADATA = "a_share_sync_status"
    A_SHARE_SYNC_SUMMARY_METADATA = "a_share_sync_summary"
    A_SHARE_ST_METADATA = A_SHARE_ST_METADATA_NAME
    A_SHARE_PREFIXES = ("SH.", "SZ.")
    DEFAULT_ALLOWED_A_SHARE_FAILURES = 0
    DEFAULT_MIN_A_SHARE_TARGET_HIT_RATIO = 0.995
    DEFAULT_MAX_NON_BLOCKING_A_SHARE_GAPS = 20
    TOLERABLE_A_SHARE_GAP_REASONS = frozenset({"empty_data", "target_not_reached", "converted_empty"})
    TOLERABLE_A_SHARE_QUERY_STATUSES = frozenset({"ok", "empty_data", "converted_empty"})

    def __init__(self):
        """Initialize scheduler."""
        self.scheduler = BlockingScheduler(timezone='Asia/Shanghai')
        self.client: FutuClient = None
        self.bs_client: BaostockClient = None
        self.db_engine: DBEngine = None
        self.qlib_writer = None

    def _init_qlib_writer(self):
        """Initialize Qlib direct writer if QLIB_DATA_DIR is configured."""
        if self.qlib_writer is not None:
            return
        qlib_dir = os.environ.get("QLIB_DATA_DIR", "")
        if qlib_dir:
            try:
                from converter.incremental import QlibDirectWriter
                self.qlib_writer = QlibDirectWriter(qlib_dir)
                logger.info(f"Qlib direct writer initialized: {qlib_dir}")
            except ImportError:
                logger.warning("converter.incremental not available, falling back to parquet")

    def sync_code_data(self, code: str):
        """
        Sync data for a single stock code.

        Args:
            code: Stock code
        """
        logger.info(f"Starting sync for {code}")

        try:
            # 1. Daily K-line (K_DAY)
            logger.info(f"Syncing {code} daily K-line...")
            self.sync_kline(code, "K_DAY")
            self.db_engine.log_job("success", f"Synced {code} daily K-line", code, "K_DAY")

            # 2. 1-minute K-line (K_1M) - sharded by year
            logger.info(f"Syncing {code} 1-min K-line...")
            self.sync_kline_1m(code)
            self.db_engine.log_job("success", f"Synced {code} 1-min K-line", code, "K_1M")

            logger.info(f"Completed sync for {code}")

        except Exception as e:
            error_msg = f"Sync failed for {code}: {e}"
            logger.error(error_msg)
            self.db_engine.log_job("error", error_msg, code, None)

    def _expected_bars_in_range(self, start: str, end: str, ktype: str) -> int:
        """Estimate expected K-line bar count in a date range."""
        start_d = datetime.strptime(start, "%Y-%m-%d")
        end_d = datetime.strptime(end, "%Y-%m-%d")
        days = (end_d - start_d).days + 1
        if ktype == "K_DAY":
            return max(1, int(days * 250 / 365))  # ~250 trading days/year
        return max(1, days)

    def sync_kline(self, code: str, ktype: str, start: str = None, end: str = None):
        """
        Sync K-line data: check DB by range, only fetch missing intervals from Futu.
        """
        # Qlib direct write for daily K-line
        if self.qlib_writer and ktype == "K_DAY":
            self._sync_kline_to_qlib(code, start, end)
            return

        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        if start is None:
            max_date = self.db_engine.get_kline_max_date(code, ktype)
            if max_date is not None:
                if max_date >= end:
                    logger.info(f"{code} {ktype} already up to date (max={max_date}), skipping")
                    return
                next_day = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                start = next_day
                logger.info(f"{code} {ktype} incremental fetch: {start} ~ {end}")
            elif ktype == "K_DAY":
                years_back = 5 if code.startswith(("HK.8", "SH.LIST")) else 10
                start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
                logger.info(f"{code} K_DAY first fetch, range: {start} ~ {end}")

        start_d = datetime.strptime(start, "%Y-%m-%d")
        end_d = datetime.strptime(end, "%Y-%m-%d")
        # Split by year for daily K-line with long ranges
        if ktype == "K_DAY" and (end_d - start_d).days > 400:
            years = range(start_d.year, end_d.year + 1)
            for y in years:
                chunk_start = f"{y}-01-01" if y > start_d.year else start
                chunk_end = f"{y}-12-31" if y < end_d.year else end
                if datetime.strptime(chunk_start, "%Y-%m-%d") > end_d:
                    continue
                if datetime.strptime(chunk_end, "%Y-%m-%d") < start_d:
                    continue
                chunk_start = max(chunk_start, start)
                chunk_end = min(chunk_end, end)
                existing = self.db_engine.get_kline_count_in_range(code, ktype, chunk_start, chunk_end)
                expected = self._expected_bars_in_range(chunk_start, chunk_end, ktype)
                if existing >= max(1, int(expected * 0.95)):
                    logger.info(f"{code} {ktype} range {chunk_start}~{chunk_end} has {existing} records, skipping")
                    continue
                data = self.client.get_history_kline(
                    code=code, start=chunk_start, end=chunk_end, ktype=ktype, autype="qfq"
                )
                if data:
                    self.db_engine.append_kline(pd.DataFrame(data), code, ktype)
        else:
            existing = self.db_engine.get_kline_count_in_range(code, ktype, start, end)
            expected = self._expected_bars_in_range(start, end, ktype)
            if existing >= max(1, int(expected * 0.95)):
                logger.info(f"{code} {ktype} range {start}~{end} has {existing} records, skipping")
                return
            data = self.client.get_history_kline(
                code=code, start=start, end=end, ktype=ktype, autype="qfq"
            )
            if not data:
                logger.warning(f"{code} {ktype} no data")
                return
            self.db_engine.append_kline(pd.DataFrame(data), code, ktype)

    def _sync_kline_to_qlib(self, code: str, start: str = None, end: str = None):
        """Sync daily K-line via Futu and write directly to Qlib bin format."""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        if start is None:
            max_date = self.qlib_writer.get_stock_last_date(code)
            if max_date is not None:
                if max_date >= end:
                    return
                start = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                years_back = 5 if code.startswith("HK.8") else 10
                start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
                logger.info(f"{code} K_DAY first fetch (qlib): {start} ~ {end}")

        data = self.client.get_history_kline(
            code=code, start=start, end=end, ktype="K_DAY", autype="qfq"
        )
        if data:
            n = self.qlib_writer.write_stock_records(code, data)
            if n > 0:
                self.db_engine.log_job("success", f"{code} +{n} days (qlib)", code, "K_DAY")

    def sync_kline_1m(self, code: str):
        """
        Sync 1-minute K-line data (sharded by year).
        """
        current_year = datetime.now().year
        years_to_sync = [current_year - 2, current_year - 1, current_year]

        for year in years_to_sync:
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"
            if year == current_year:
                year_end = datetime.now().strftime("%Y-%m-%d")

            # Historical years: skip if nearly full; current year: always incremental
            if year < current_year:
                existing = self.db_engine.get_kline_count_in_range(
                    code, "K_1M", year_start, year_end, year=year
                )
                days_in_range = (datetime.strptime(year_end, "%Y-%m-%d") - datetime.strptime(year_start, "%Y-%m-%d")).days + 1
                expected_1m = max(1, int(days_in_range * 250 * 240 / 365))
                if existing >= int(expected_1m * 0.95):
                    logger.info(f"{code} {year} 1-min K-line has {existing} records, skipping")
                    continue

            max_date = self.db_engine.get_kline_max_date(code, "K_1M", year=year)
            if max_date is not None and max_date >= year_end:
                logger.info(f"{code} {year} 1-min K-line up to date (max={max_date}), skipping")
                continue

            start = year_start
            if max_date is not None:
                start = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(f"{code} {year} 1-min K-line incremental: {start} ~ {year_end}")
            else:
                logger.info(f"Syncing {code} {year} 1-min K-line...")

            try:
                data = self.client.get_history_kline(
                    code=code,
                    start=start,
                    end=year_end,
                    ktype="K_1M",
                    autype="qfq"
                )

                if not data:
                    logger.warning(f"{code} {year} 1-min K-line no data")
                    continue

                df = pd.DataFrame(data)
                self.db_engine.append_kline(df, code, "K_1M", year=year)

            except Exception as e:
                logger.error(f"Sync {code} {year} 1-min K-line failed: {e}")
                continue

    FUNDAMENTAL_FIELDS = [
        "pb_ratio", "dividend_ttm", "net_profit_ttm",
        "return_on_equity", "net_profit_growth_rate",
    ]

    def sync_fundamentals(self, codes: List[str]):
        """Daily fundamental snapshot collection (PB, dividend, EPS, ROE)."""
        logger.info(f"Collecting fundamental snapshots for {len(codes)} stocks...")
        try:
            records = self.client.get_fundamentals(codes)
            if not records:
                logger.warning("Fundamental snapshot empty")
                return

            if self.qlib_writer:
                # Group by code, write each stock's fundamentals as Qlib features
                from collections import defaultdict
                by_code = defaultdict(list)
                for r in records:
                    by_code[r["code"]].append(r)
                total = 0
                for code, code_records in by_code.items():
                    n = self.qlib_writer.write_feature_records(
                        code, code_records, self.FUNDAMENTAL_FIELDS
                    )
                    total += n
                logger.info(f"Fundamentals → qlib: {len(by_code)} stocks, {total} new dates")
            else:
                df = pd.DataFrame(records)
                out_path = self.db_engine.data_path / "fundamentals" / "daily_snapshot.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    existing = pd.read_parquet(out_path)
                    combined = pd.concat([existing, df]).drop_duplicates(
                        subset=["code", "date"], keep="last"
                    ).sort_values(["code", "date"]).reset_index(drop=True)
                else:
                    combined = df
                combined.to_parquet(out_path, index=False)
                logger.info(f"Fundamental snapshot saved: {len(df)} new, {len(combined)} total")
        except Exception as e:
            logger.error(f"Fundamental snapshot collection failed: {e}")

    def sync_industry_map(self):
        """Collect industry plate mapping (refresh weekly).

        When qlib_writer is available, encodes industry as numeric ID and writes
        industry_id.day.bin for each stock (constant feature across all dates).
        Also saves the ID↔name mapping to metadata/industry_map.json.
        """
        logger.info("Collecting industry plate mapping...")
        try:
            records = self.client.get_industry_map("HK")
            if not records:
                logger.warning("Industry plate mapping empty")
                return

            if self.qlib_writer:
                # Build industry name → numeric ID mapping
                # Load existing mapping to keep IDs stable across updates
                existing_map = self.qlib_writer.load_metadata("industry_map") or {}
                name_to_id = {v: int(k) for k, v in existing_map.items()} if existing_map else {}
                next_id = max(name_to_id.values(), default=0) + 1

                # Assign IDs to new industries
                for r in records:
                    industry = r.get("industry", "")
                    if industry and industry not in name_to_id:
                        name_to_id[industry] = next_id
                        next_id += 1

                # Save mapping: {id_str: industry_name}
                id_map = {str(v): k for k, v in name_to_id.items()}
                self.qlib_writer.save_metadata("industry_map", id_map)

                # Build code → industry_id (use first/primary industry per stock)
                code_industry = {}
                for r in records:
                    code = r.get("code", "")
                    industry = r.get("industry", "")
                    if code and industry and code not in code_industry:
                        code_industry[code] = name_to_id[industry]

                # Write constant feature for each stock
                written = 0
                for code, industry_id in code_industry.items():
                    if self.qlib_writer.write_constant_feature(code, "industry_id", float(industry_id)):
                        written += 1

                logger.info(
                    f"Industry → qlib: {len(name_to_id)} industries, "
                    f"{written}/{len(code_industry)} stocks written"
                )
            else:
                df = pd.DataFrame(records)
                out_path = self.db_engine.data_path / "metadata" / "industry_map.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_path, index=False)
                logger.info(f"Industry mapping saved: {len(df)} records")
        except Exception as e:
            logger.error(f"Industry mapping collection failed: {e}")

    SHORT_SELL_FIELDS = [
        "short_sell_qty", "short_sell_amount", "short_sell_ratio",
    ]

    def sync_short_sell(self):
        """Daily HK short selling data collection."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Skip check: for Qlib, check bin; for parquet, check file
        if not self.qlib_writer:
            out_path = self.db_engine.data_path / "short_sell" / "daily.parquet"
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                if "date" in existing.columns and today in existing["date"].values:
                    logger.info(f"Short sell data {today} already collected, skipping")
                    return

        logger.info("Collecting HK short sell data...")
        try:
            records = self.client.get_short_sell_list("HK")
            if not records:
                logger.warning("Short sell data empty (may require subscription)")
                return

            if self.qlib_writer:
                from collections import defaultdict
                by_code = defaultdict(list)
                for r in records:
                    by_code[r["code"]].append(r)
                total = 0
                for code, code_records in by_code.items():
                    n = self.qlib_writer.write_feature_records(
                        code, code_records, self.SHORT_SELL_FIELDS
                    )
                    total += n
                logger.info(f"Short sell → qlib: {len(by_code)} stocks, {total} new dates")
            else:
                df = pd.DataFrame(records)
                out_path = self.db_engine.data_path / "short_sell" / "daily.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    existing = pd.read_parquet(out_path)
                    combined = pd.concat([existing, df]).drop_duplicates(
                        subset=["code", "date"], keep="last"
                    ).sort_values(["date", "code"]).reset_index(drop=True)
                else:
                    combined = df
                combined.to_parquet(out_path, index=False)
                logger.info(f"Short sell data saved: {len(df)} records")
        except Exception as e:
            logger.error(f"Short sell data collection failed: {e}")

    def sync_a_share_kline(self, code: str, target_end_date: str | None = None) -> dict:
        """
        Sync A-share daily K-line using Baostock.
        Writes directly to Qlib bin format (no parquet intermediate).
        Returns a diagnostic result dict including success and failure reason.
        """
        end = target_end_date or datetime.now().strftime("%Y-%m-%d")

        # Check last collected date from Qlib bin
        if self.qlib_writer:
            max_date = self.qlib_writer.get_stock_last_date(code)
        else:
            max_date = self.db_engine.get_kline_max_date(code, "K_DAY")

        if max_date is not None:
            if max_date >= end:
                return {
                    "ok": True,
                    "reason": "already_up_to_date",
                    "code": code,
                    "target_end_date": end,
                    "latest_date": max_date,
                }
            start = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365 * 15)).strftime("%Y-%m-%d")
            logger.info(f"Baostock {code} K_DAY first fetch: {start} ~ {end}")

        if start > end:
            return {
                "ok": True,
                "reason": "already_up_to_date",
                "code": code,
                "target_end_date": end,
                "latest_date": max_date,
            }

        data = self.bs_client.get_history_kline(code, start=start, end=end, ktype="K_DAY")
        query_status = self.bs_client.get_last_history_kline_status() or {}
        if not data:
            raw_status = str(query_status.get("status") or "")
            reason = "empty_data" if raw_status in {"", "ok"} else raw_status
            logger.warning(f"Baostock {code} returned no usable data for target={end}, reason={reason}")
            return {
                "ok": False,
                "reason": reason,
                "code": code,
                "target_end_date": end,
                "latest_date": max_date,
                "query_status": query_status,
            }

        # Write directly to Qlib bin
        if self.qlib_writer:
            n = self.qlib_writer.write_stock_records(code, data)
            if n > 0:
                self.db_engine.log_job("success", f"Baostock {code} +{n} days (qlib)", code, "K_DAY")
            refreshed_max_date = self.qlib_writer.get_stock_last_date(code)
        else:
            self.db_engine.append_kline(pd.DataFrame(data), code, "K_DAY")
            self.db_engine.log_job("success", f"Baostock {code} +{len(data)} records", code, "K_DAY")
            refreshed_max_date = self.db_engine.get_kline_max_date(code, "K_DAY")

        ok = bool(refreshed_max_date) and refreshed_max_date >= end
        return {
            "ok": ok,
            "reason": "ok" if ok else "target_not_reached",
            "code": code,
            "target_end_date": end,
            "latest_date": refreshed_max_date,
            "query_status": query_status,
            "rows": len(data),
        }

    def sync_a_share_kline_via_futu(self, code: str, target_end_date: str | None = None) -> dict:
        """
        Sync A-share daily K-line using Futu as a fallback when Baostock is unavailable.

        This keeps the same result shape as sync_a_share_kline() so completion
        evaluation can treat Baostock and Futu runs uniformly.
        """
        end = target_end_date or datetime.now().strftime("%Y-%m-%d")

        if self.qlib_writer:
            max_date = self.qlib_writer.get_stock_last_date(code)
        else:
            max_date = self.db_engine.get_kline_max_date(code, "K_DAY")

        if max_date is not None:
            if max_date >= end:
                return {
                    "ok": True,
                    "reason": "already_up_to_date",
                    "code": code,
                    "target_end_date": end,
                    "latest_date": max_date,
                    "query_status": {"source": "futu", "status": "already_up_to_date"},
                }
            start = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365 * 15)).strftime("%Y-%m-%d")
            logger.info(f"Futu fallback {code} K_DAY first fetch: {start} ~ {end}")

        if start > end:
            return {
                "ok": True,
                "reason": "already_up_to_date",
                "code": code,
                "target_end_date": end,
                "latest_date": max_date,
                "query_status": {"source": "futu", "status": "already_up_to_date"},
            }

        try:
            data = self.client.get_history_kline(
                code=code, start=start, end=end, ktype="K_DAY", autype="qfq"
            )
        except Exception as e:
            logger.warning(f"Futu fallback {code} failed for target={end}: {e}")
            return {
                "ok": False,
                "reason": "query_failed",
                "code": code,
                "target_end_date": end,
                "latest_date": max_date,
                "query_status": {"source": "futu", "status": "query_failed", "error": str(e)},
                "error": str(e),
            }

        if not data:
            logger.warning(f"Futu fallback {code} returned no usable data for target={end}")
            return {
                "ok": False,
                "reason": "empty_data",
                "code": code,
                "target_end_date": end,
                "latest_date": max_date,
                "query_status": {"source": "futu", "status": "empty_data"},
            }

        if self.qlib_writer:
            n = self.qlib_writer.write_stock_records(code, data)
            if n > 0:
                self.db_engine.log_job("success", f"Futu fallback {code} +{n} days (qlib)", code, "K_DAY")
            refreshed_max_date = self.qlib_writer.get_stock_last_date(code)
        else:
            self.db_engine.append_kline(pd.DataFrame(data), code, "K_DAY")
            self.db_engine.log_job("success", f"Futu fallback {code} +{len(data)} records", code, "K_DAY")
            refreshed_max_date = self.db_engine.get_kline_max_date(code, "K_DAY")

        ok = bool(refreshed_max_date) and refreshed_max_date >= end
        return {
            "ok": ok,
            "reason": "ok" if ok else "target_not_reached",
            "code": code,
            "target_end_date": end,
            "latest_date": refreshed_max_date,
            "query_status": {"source": "futu", "status": "ok", "rows": len(data)},
            "rows": len(data),
        }

    def _latest_a_share_date(self) -> str | None:
        """Read the latest A-share end date from Qlib instruments metadata."""
        self._init_qlib_writer()
        if self.qlib_writer is not None:
            latest = None
            for code, instrument_range in getattr(self.qlib_writer, "instruments", {}).items():
                if not code.startswith(self.A_SHARE_PREFIXES):
                    continue
                if not instrument_range or len(instrument_range) < 2:
                    continue
                end_date = instrument_range[1]
                if latest is None or end_date > latest:
                    latest = end_date
            if latest:
                return latest

        qlib_dir_raw = os.environ.get("QLIB_DATA_DIR", "")
        if not qlib_dir_raw:
            return None
        qlib_dir = Path(qlib_dir_raw)

        inst_path = qlib_dir / "instruments" / "all.txt"
        if not inst_path.exists():
            return None

        latest = None
        for line in inst_path.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            code, _, end_date = parts[:3]
            if not code.startswith(self.A_SHARE_PREFIXES):
                continue
            if latest is None or end_date > latest:
                latest = end_date
        return latest

    def _load_a_share_sync_status(self) -> dict | None:
        """Load A-share collection completion metadata."""
        self._init_qlib_writer()
        if not self.qlib_writer:
            logger.warning("Qlib direct writer unavailable; A-share completion metadata is disabled")
            return None

        status = self.qlib_writer.load_metadata(self.A_SHARE_SYNC_STATUS_METADATA)
        return status if isinstance(status, dict) else None

    def _latest_completed_a_share_date(self) -> str | None:
        """Return the latest fully completed A-share trading date."""
        status = self._load_a_share_sync_status() or {}
        completed = status.get("last_completed_trade_date")
        return completed if isinstance(completed, str) and completed else None

    @staticmethod
    def _latest_weekday_on_or_before(date_value: str) -> str:
        """Return a conservative weekday fallback when Baostock calendar is unavailable."""
        day = datetime.strptime(date_value, "%Y-%m-%d")
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return day.strftime("%Y-%m-%d")

    def _a_share_codes_from_qlib(self) -> list[str]:
        """Use the existing Qlib universe as an A-share fallback when Baostock listing is unavailable."""
        self._init_qlib_writer()
        if self.qlib_writer is not None:
            codes = sorted(
                code for code in getattr(self.qlib_writer, "instruments", {})
                if code.startswith(self.A_SHARE_PREFIXES)
            )
            if codes:
                return codes

        qlib_dir_raw = os.environ.get("QLIB_DATA_DIR", "")
        if not qlib_dir_raw:
            return []
        inst_path = Path(qlib_dir_raw) / "instruments" / "all.txt"
        if not inst_path.exists():
            return []

        codes = []
        for line in inst_path.read_text().splitlines():
            parts = line.strip().split("\t")
            if not parts:
                continue
            code = parts[0]
            if code.startswith(self.A_SHARE_PREFIXES):
                codes.append(code)
        return sorted(set(codes))

    def _ensure_futu_connected(self) -> bool:
        """Ensure a Futu quote context is available for HK and A-share fallback collection."""
        if self.client and self.client.ctx:
            return True
        try:
            self.client = FutuClient(settings.futu_host, settings.futu_port)
            return self.client.connect()
        except Exception as e:
            logger.error(f"Futu connection failed: {e}")
            return False

    def _allowed_a_share_failures(self) -> int:
        raw = os.environ.get("ALLOWED_A_SHARE_FAILURES")
        if raw is None or raw == "":
            return self.DEFAULT_ALLOWED_A_SHARE_FAILURES
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning("Invalid ALLOWED_A_SHARE_FAILURES=%r, using default=%d", raw, self.DEFAULT_ALLOWED_A_SHARE_FAILURES)
            return self.DEFAULT_ALLOWED_A_SHARE_FAILURES

    def _min_a_share_target_hit_ratio(self) -> float:
        raw = os.environ.get("A_SHARE_MIN_TARGET_HIT_RATIO")
        if raw is None or raw == "":
            return self.DEFAULT_MIN_A_SHARE_TARGET_HIT_RATIO
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "Invalid A_SHARE_MIN_TARGET_HIT_RATIO=%r, using default=%s",
                raw,
                self.DEFAULT_MIN_A_SHARE_TARGET_HIT_RATIO,
            )
            return self.DEFAULT_MIN_A_SHARE_TARGET_HIT_RATIO
        return min(max(value, 0.0), 1.0)

    def _max_non_blocking_a_share_gaps(self) -> int:
        raw = os.environ.get("A_SHARE_MAX_NON_BLOCKING_GAPS")
        if raw is None or raw == "":
            return self.DEFAULT_MAX_NON_BLOCKING_A_SHARE_GAPS
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning(
                "Invalid A_SHARE_MAX_NON_BLOCKING_GAPS=%r, using default=%d",
                raw,
                self.DEFAULT_MAX_NON_BLOCKING_A_SHARE_GAPS,
            )
            return self.DEFAULT_MAX_NON_BLOCKING_A_SHARE_GAPS

    def _is_tolerable_a_share_gap(self, result: dict, target_date: str) -> bool:
        """Treat non-trading symbols as gaps, not blocking collector failures."""
        if result.get("ok"):
            return False

        reason = str(result.get("reason") or "")
        if reason not in self.TOLERABLE_A_SHARE_GAP_REASONS:
            return False

        if result.get("error"):
            return False

        query_status = result.get("query_status") or {}
        if (query_status.get("error") or ""):
            return False

        status = str(query_status.get("status") or "")
        if status and status not in self.TOLERABLE_A_SHARE_QUERY_STATUSES:
            return False

        latest_date = result.get("latest_date")
        if latest_date and latest_date >= target_date:
            return False

        return True

    def _a_share_target_hit_ratio(self, total_codes: int, failed_a_share_runs: list[dict]) -> float:
        if total_codes <= 0:
            return 0.0
        target_hit_count = max(total_codes - len(failed_a_share_runs), 0)
        return target_hit_count / total_codes

    def _is_a_share_market_ready(
        self,
        *,
        total_codes: int,
        failed_a_share_runs: list[dict],
        latest_a_share_date: str | None,
        target_date: str,
        non_blocking_gap_count: int,
    ) -> bool:
        """Determine whether the market-level A-share dataset is ready for downstream nightly jobs."""
        if not latest_a_share_date or latest_a_share_date < target_date:
            return False

        target_hit_ratio = self._a_share_target_hit_ratio(total_codes, failed_a_share_runs)
        return (
            target_hit_ratio >= self._min_a_share_target_hit_ratio()
            or non_blocking_gap_count <= self._max_non_blocking_a_share_gaps()
        )

    def _evaluate_a_share_sync_completion(
        self,
        *,
        total_codes: int,
        failed_a_share_runs: list[dict],
        latest_a_share_date: str | None,
        target_a_share_date: str,
        allowed_failures: int,
    ) -> dict:
        """Classify A-share gaps and decide whether completion metadata can advance."""
        failure_preview = [
            {
                "code": item.get("code"),
                "reason": item.get("reason"),
                "latest_date": item.get("latest_date"),
                "error": item.get("error") or (item.get("query_status") or {}).get("error"),
            }
            for item in failed_a_share_runs[:10]
        ]
        non_blocking_gap_runs = [
            item for item in failed_a_share_runs
            if self._is_tolerable_a_share_gap(item, target_a_share_date)
        ]
        blocking_failures = [
            item for item in failed_a_share_runs
            if item not in non_blocking_gap_runs
        ]
        blocking_failure_preview = [
            {
                "code": item.get("code"),
                "reason": item.get("reason"),
                "latest_date": item.get("latest_date"),
                "error": item.get("error") or (item.get("query_status") or {}).get("error"),
            }
            for item in blocking_failures[:10]
        ]
        non_blocking_gap_preview = [
            {
                "code": item.get("code"),
                "reason": item.get("reason"),
                "latest_date": item.get("latest_date"),
                "error": item.get("error") or (item.get("query_status") or {}).get("error"),
            }
            for item in non_blocking_gap_runs[:10]
        ]
        target_hit_ratio = self._a_share_target_hit_ratio(total_codes, failed_a_share_runs)
        market_ready = self._is_a_share_market_ready(
            total_codes=total_codes,
            failed_a_share_runs=failed_a_share_runs,
            latest_a_share_date=latest_a_share_date,
            target_date=target_a_share_date,
            non_blocking_gap_count=len(non_blocking_gap_runs),
        )
        completed_a_share_target = None
        summary_status = "complete"
        if failed_a_share_runs:
            if market_ready and len(blocking_failures) <= allowed_failures:
                completed_a_share_target = target_a_share_date
                if blocking_failures:
                    summary_status = "complete_with_tolerated_failures"
                elif non_blocking_gap_runs:
                    summary_status = "complete_with_non_blocking_gaps"
            else:
                summary_status = "incomplete"
        else:
            completed_a_share_target = target_a_share_date

        return {
            "status": summary_status,
            "completed_a_share_target": completed_a_share_target,
            "target_hit_ratio": target_hit_ratio,
            "target_hit_count": max(total_codes - len(failed_a_share_runs), 0),
            "min_target_hit_ratio": self._min_a_share_target_hit_ratio(),
            "max_non_blocking_gap_count": self._max_non_blocking_a_share_gaps(),
            "market_ready": market_ready,
            "failure_preview": failure_preview,
            "blocking_failures": blocking_failures,
            "blocking_failure_preview": blocking_failure_preview,
            "non_blocking_gap_runs": non_blocking_gap_runs,
            "non_blocking_gap_preview": non_blocking_gap_preview,
        }

    def _save_a_share_sync_summary(self, summary: dict):
        """Persist A-share run summary metadata for diagnostics."""
        self._init_qlib_writer()
        if not self.qlib_writer:
            raise RuntimeError("QLIB_DATA_DIR is required to persist A-share summary metadata")
        self.qlib_writer.save_metadata(self.A_SHARE_SYNC_SUMMARY_METADATA, summary)

    def _save_a_share_stock_basic_metadata(self, stock_basic: pd.DataFrame):
        """Persist current A-share names and ST/*ST flags for model universe filtering."""
        self._init_qlib_writer()
        if not self.qlib_writer:
            logger.warning("Qlib direct writer unavailable; A-share ST metadata is disabled")
            return
        if stock_basic is None or stock_basic.empty:
            logger.warning("A-share stock basic metadata empty; skipping ST metadata save")
            return

        metadata = build_a_share_stock_basic_metadata(
            stock_basic.to_dict("records"),
            source="baostock.query_stock_basic",
        )
        self.qlib_writer.save_metadata(self.A_SHARE_ST_METADATA, metadata)
        logger.info(
            "A-share ST metadata updated: total={} st_count={}",
            metadata.get("total", 0),
            metadata.get("st_count", 0),
        )

    def _mark_a_share_sync_completed(self, target_date: str, total_codes: int, started_at: datetime):
        """Persist the latest fully completed A-share collection target date."""
        self._init_qlib_writer()
        if not self.qlib_writer:
            raise RuntimeError("QLIB_DATA_DIR is required to persist A-share completion metadata")

        data = {
            "last_completed_trade_date": target_date,
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "total_codes": total_codes,
        }
        self.qlib_writer.save_metadata(self.A_SHARE_SYNC_STATUS_METADATA, data)

    def sync_ticker(self, code: str):
        """
        Sync tick data (current day only, skip if already collected).

        Args:
            code: Stock code
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self.db_engine.ticker_file_exists(code, today):
            logger.info(f"{code} tick data already collected today, skipping")
            return

        start = f"{today} 09:30:00"
        end = f"{today} 16:00:00"

        try:
            data = self.client.get_rt_ticker(
                code=code,
                start=start,
                end=end
            )

            if not data:
                logger.warning(f"{code} {today} tick data empty")
                return

            df = pd.DataFrame(data)
            self.db_engine.append_ticker(df, code, today)

        except Exception as e:
            logger.error(f"Sync {code} {today} tick data failed: {e}")
            raise

    def run_daily_job(self):
        """
        Execute daily data sync job.
        A-shares (SH.*/SZ.*) use Baostock, HK stocks use Futu.
        Writes directly to Qlib bin format when QLIB_DATA_DIR is set.
        """
        job_start_time = datetime.now()
        logger.info(f"Starting daily data sync job ({job_start_time.strftime('%Y-%m-%d %H:%M:%S')})")

        try:
            self.db_engine = DBEngine(settings.data_path)
            self.bs_client = BaostockClient(rate_limit=0.3)

            self._init_qlib_writer()
            today = job_start_time.strftime("%Y-%m-%d")
            a_share_source = "baostock"
            force_futu_a_share = os.environ.get("A_SHARE_FORCE_FUTU", "false").lower() == "true"
            enable_futu_a_share_fallback = (
                os.environ.get("A_SHARE_ENABLE_FUTU_FALLBACK", "false").lower() == "true"
            )
            target_override = os.environ.get("A_SHARE_TARGET_DATE_OVERRIDE", "").strip()
            if target_override:
                target_a_share_date = target_override
                logger.warning("A-share target date override: {}", target_a_share_date)
            elif force_futu_a_share:
                target_a_share_date = self._latest_weekday_on_or_before(today)
                logger.warning("A-share Futu fallback forced, weekday target={}", target_a_share_date)
            else:
                try:
                    target_a_share_date = self.bs_client.latest_trade_date(on_or_before=today)
                except Exception as e:
                    target_a_share_date = self._latest_weekday_on_or_before(today)
                    if enable_futu_a_share_fallback:
                        a_share_source = "futu"
                        logger.error(
                            "Baostock A-share calendar failed: {}; falling back to weekday target={} and Futu K-line",
                            e,
                            target_a_share_date,
                        )
                    else:
                        logger.error(
                            "Baostock A-share calendar failed: {}; Futu full-market fallback disabled "
                            "(set A_SHARE_ENABLE_FUTU_FALLBACK=true to enable), weekday target={}",
                            e,
                            target_a_share_date,
                        )
            if force_futu_a_share:
                a_share_source = "futu"
            try:
                if not target_a_share_date:
                    raise ValueError("empty target")
                datetime.strptime(target_a_share_date, "%Y-%m-%d")
            except (TypeError, ValueError):
                raise RuntimeError(f"Invalid A-share target date: {target_a_share_date}")
            completed_a_share_date = self._latest_completed_a_share_date()
            latest_a_share_date = self._latest_a_share_date()
            logger.info(
                "A-share target date: {} | latest completed: {} | latest observed: {} | source={}",
                target_a_share_date or "N/A",
                completed_a_share_date or "N/A",
                latest_a_share_date or "N/A",
                a_share_source,
            )
            completed_a_share_target = None

            # 1. Get target stock pool

            # A-shares: via baostock
            a_share_codes = []
            if force_futu_a_share:
                logger.warning("Skipping Baostock A-share list because A_SHARE_FORCE_FUTU=true")
            else:
                try:
                    stock_basic = self.bs_client.get_a_share_basic()
                    a_share_codes = stock_basic["code"].astype(str).tolist() if not stock_basic.empty else []
                    self._save_a_share_stock_basic_metadata(stock_basic)
                    logger.info(f"Baostock A-share targets: {len(a_share_codes)}")
                except Exception as e:
                    logger.error(f"Baostock A-share list failed: {e}")
                    if enable_futu_a_share_fallback:
                        a_share_source = "futu"

            if not a_share_codes:
                if a_share_source == "futu":
                    a_share_codes = self._a_share_codes_from_qlib()
                if a_share_codes and a_share_source == "futu":
                    a_share_source = "futu"
                    logger.warning(
                        "Using Qlib A-share universe as fallback targets: {} stocks",
                        len(a_share_codes),
                    )
                elif not a_share_codes:
                    logger.error("No A-share targets available; skipping A-share collection")

            # HK stocks: via Futu index constituents
            hk_codes = []
            futu_ok = False
            hk_indexes = [idx for idx in settings.index_list if idx.startswith("HK.")]
            if hk_indexes:
                if self._ensure_futu_connected():
                    futu_ok = True
                    for index_code in hk_indexes:
                        try:
                            constituents = self.client.get_index_constituents(index_code)
                            hk_codes.extend(constituents)
                            logger.info(f"Futu index {index_code}: {len(constituents)} stocks")
                        except Exception as e:
                            logger.error(f"Failed to get constituents for {index_code}: {e}")
                    hk_codes = sorted(set(hk_codes))
                    logger.info(f"Futu HK targets: {len(hk_codes)}")

            if a_share_source == "futu" and not futu_ok:
                futu_ok = self._ensure_futu_connected()

            # Extra codes
            extra_codes = [c.strip() for c in settings.extra_codes if c.strip()]

            # 2. A-share daily K-line (Baostock, or Futu fallback)
            if a_share_codes:
                if not target_a_share_date:
                    logger.warning("Cannot determine latest A-share trading date, skipping A-share collection")
                elif a_share_source == "futu" and not futu_ok:
                    logger.error("Futu fallback unavailable, skipping A-share collection")
                elif completed_a_share_date and completed_a_share_date >= target_a_share_date:
                    logger.info(
                        "A-share already up to date: completed={} target={}, skipping",
                        completed_a_share_date,
                        target_a_share_date,
                    )
                else:
                    logger.info(
                        f"=== {a_share_source} A-share collection: {len(a_share_codes)} stocks, "
                        f"target={target_a_share_date} ==="
                    )
                    failed_a_share_runs = []
                    for idx, code in enumerate(a_share_codes, 1):
                        try:
                            if a_share_source == "futu":
                                result = self.sync_a_share_kline_via_futu(
                                    code, target_end_date=target_a_share_date
                                )
                            else:
                                result = self.sync_a_share_kline(
                                    code, target_end_date=target_a_share_date
                                )
                            if not result.get("ok"):
                                failed_a_share_runs.append(result)
                            if idx % 50 == 0:
                                elapsed = (datetime.now() - job_start_time).total_seconds()
                                logger.info(
                                    f"A-share progress: {idx}/{len(a_share_codes)} "
                                    f"({idx*100//len(a_share_codes)}%) | "
                                    f"elapsed: {elapsed/60:.1f} min | last_code={code} | "
                                    f"target={target_a_share_date} | pending_failures={len(failed_a_share_runs)}"
                                )
                        except Exception as e:
                            failed_a_share_runs.append(
                                {
                                    "ok": False,
                                    "reason": "exception",
                                    "code": code,
                                    "target_end_date": target_a_share_date,
                                    "error": str(e),
                                }
                            )
                            logger.error(f"[{idx}/{len(a_share_codes)}] {a_share_source} {code} failed: {e}")
                            continue

                    latest_a_share_date = self._latest_a_share_date()
                    allowed_failures = self._allowed_a_share_failures()
                    completion = self._evaluate_a_share_sync_completion(
                        total_codes=len(a_share_codes),
                        failed_a_share_runs=failed_a_share_runs,
                        latest_a_share_date=latest_a_share_date,
                        target_a_share_date=target_a_share_date,
                        allowed_failures=allowed_failures,
                    )
                    summary_status = completion["status"]
                    completed_a_share_target = completion["completed_a_share_target"]
                    blocking_failures = completion["blocking_failures"]
                    blocking_failure_preview = completion["blocking_failure_preview"]
                    non_blocking_gap_runs = completion["non_blocking_gap_runs"]
                    non_blocking_gap_preview = completion["non_blocking_gap_preview"]
                    failure_preview = completion["failure_preview"]

                    if completed_a_share_target:
                        logger.warning(
                            "A-share completion metadata updated: target={} latest={} blocking_failures={} non_blocking_gaps={} allowed_failures={} target_hit_ratio={:.4f} blocking_preview={} gap_preview={}",
                            target_a_share_date,
                            latest_a_share_date,
                            len(blocking_failures),
                            len(non_blocking_gap_runs),
                            allowed_failures,
                            completion["target_hit_ratio"],
                            ",".join(item["code"] for item in blocking_failure_preview if item.get("code")) or "-",
                            ",".join(item["code"] for item in non_blocking_gap_preview if item.get("code")) or "-",
                        )
                    else:
                        logger.warning(
                            "A-share completion metadata not updated; target={} latest={} blocking_failures={} non_blocking_gaps={} allowed_failures={} target_hit_ratio={:.4f} min_target_hit_ratio={:.4f} max_non_blocking_gaps={} blocking_preview={} gap_preview={}",
                            target_a_share_date,
                            latest_a_share_date or "N/A",
                            len(blocking_failures),
                            len(non_blocking_gap_runs),
                            allowed_failures,
                            completion["target_hit_ratio"],
                            completion["min_target_hit_ratio"],
                            completion["max_non_blocking_gap_count"],
                            ",".join(item["code"] for item in blocking_failure_preview if item.get("code")) or "-",
                            ",".join(item["code"] for item in non_blocking_gap_preview if item.get("code")) or "-",
                        )

                    self._save_a_share_sync_summary(
                        {
                            "status": summary_status,
                            "data_source": a_share_source,
                            "target_trade_date": target_a_share_date,
                            "latest_observed_trade_date": latest_a_share_date,
                            "started_at": job_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "total_codes": len(a_share_codes),
                            "success_count": len(a_share_codes) - len(failed_a_share_runs),
                            "failed_count": len(failed_a_share_runs),
                            "target_hit_count": completion["target_hit_count"],
                            "target_hit_ratio": completion["target_hit_ratio"],
                            "market_ready": completion["market_ready"],
                            "min_target_hit_ratio": completion["min_target_hit_ratio"],
                            "max_non_blocking_gap_count": completion["max_non_blocking_gap_count"],
                            "blocking_failed_count": len(blocking_failures),
                            "non_blocking_gap_count": len(non_blocking_gap_runs),
                            "tolerated_gap_count": len(non_blocking_gap_runs),
                            "allowed_failures": allowed_failures,
                            "failed_codes_preview": [item["code"] for item in failure_preview if item.get("code")],
                            "failed_runs_preview": failure_preview,
                            "blocking_failed_codes_preview": [
                                item["code"] for item in blocking_failure_preview if item.get("code")
                            ],
                            "blocking_failed_runs_preview": blocking_failure_preview,
                            "non_blocking_gap_codes_preview": [
                                item["code"] for item in non_blocking_gap_preview if item.get("code")
                            ],
                            "non_blocking_gap_runs_preview": non_blocking_gap_preview,
                            "tolerated_gap_codes_preview": [
                                item["code"] for item in non_blocking_gap_preview if item.get("code")
                            ],
                            "tolerated_gap_runs_preview": non_blocking_gap_preview,
                        }
                    )

            # 3. HK stock collection (Futu)
            if hk_codes and futu_ok:
                logger.info(f"=== Futu HK collection: {len(hk_codes)} stocks ===")

                # Fundamental snapshot (HK)
                fund_codes = [c for c in hk_codes if not c.startswith("HK.8")]
                try:
                    self.sync_fundamentals(fund_codes)
                except Exception as e:
                    logger.error(f"HK fundamental snapshot error: {e}")

                # Short sell data
                try:
                    self.sync_short_sell()
                except Exception as e:
                    logger.error(f"Short sell data error: {e}")

                for idx, code in enumerate(hk_codes, 1):
                    try:
                        self.sync_code_data(code)
                        if idx % 10 == 0:
                            logger.info(f"HK progress: {idx}/{len(hk_codes)}")
                    except Exception as e:
                        logger.error(f"[{idx}/{len(hk_codes)}] Futu {code} failed: {e}")
                        continue

            # 4. Extra codes (indices/US stocks, via Futu)
            if extra_codes and futu_ok:
                logger.info(f"=== Extra codes collection: {extra_codes} ===")
                for code in extra_codes:
                    if code.startswith(("SH.", "SZ.")):
                        continue  # A-shares already collected via Baostock
                    try:
                        self.sync_kline(code, "K_DAY")
                    except Exception as e:
                        logger.error(f"Extra code {code} failed: {e}")

            # Flush Qlib data after all collections
            if self.qlib_writer:
                self.qlib_writer.flush()
                logger.info("Qlib bin data flushed")
            if completed_a_share_target:
                self._mark_a_share_sync_completed(
                    completed_a_share_target,
                    len(a_share_codes),
                    job_start_time,
                )
                logger.info("A-share completion metadata updated: {}", completed_a_share_target)

            # Log job completion
            duration = (datetime.now() - job_start_time).total_seconds()
            self.db_engine.log_job(
                "success",
                f"Daily job done: A-shares {len(a_share_codes)} + HK {len(hk_codes)}, duration {duration:.0f}s",
                None, "DailyJob"
            )
            logger.info(f"Daily data sync job completed, duration {duration:.0f}s")

        except Exception as e:
            error_msg = f"Daily data sync job failed: {e}"
            logger.error(error_msg)
            if self.db_engine:
                self.db_engine.log_job("error", error_msg, None, "DailyJob")
            raise

        finally:
            if self.bs_client:
                self.bs_client.close()
            if self.client:
                self.client.disconnect()

    def _sync_via_yfinance(self, code: str):
        """Sync daily K-line for a single code via YFinance (Futu fallback)."""
        yf_client = YFinanceClient()
        if self.qlib_writer:
            max_date = self.qlib_writer.get_stock_last_date(code)
        else:
            max_date = self.db_engine.get_kline_max_date(code, "K_DAY")

        if max_date is not None:
            today = datetime.now().strftime("%Y-%m-%d")
            if max_date >= today:
                logger.info(f"YFinance {code} up to date (max={max_date}), skipping")
                return
            start = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = "2006-01-01"

        data = yf_client.get_history_kline(code, start=start)
        if data:
            if self.qlib_writer:
                n = self.qlib_writer.write_stock_records(code, data)
                if n > 0:
                    logger.info(f"YFinance {code} → qlib: +{n} days")
            else:
                self.db_engine.append_kline(pd.DataFrame(data), code, "K_DAY")
                logger.info(f"YFinance {code} sync complete: {len(data)} records")
        else:
            logger.warning(f"YFinance {code} no data")

    def run_us_morning_job(self):
        """Morning job: collect previous day US stock K-line (Futu first, fallback to YFinance)."""
        job_start = datetime.now()
        logger.info(f"Starting US morning data sync ({job_start.strftime('%Y-%m-%d %H:%M:%S')})")
        us_codes = [c.strip() for c in settings.extra_codes if c.strip() and c.startswith("US.")]
        yf_only_codes = ["US.YINN", "US.CQQQ", "US.KWEB", "US.FXI"]
        if not us_codes and not yf_only_codes:
            logger.info("No US codes configured, skipping")
            return
        try:
            self.db_engine = DBEngine(settings.data_path)
            self._init_qlib_writer()
            futu_ok = False
            try:
                self.client = FutuClient(settings.futu_host, settings.futu_port)
                if self.client.connect():
                    futu_ok = True
            except Exception:
                pass

            for code in us_codes:
                try:
                    if futu_ok:
                        self.sync_kline(code, "K_DAY")
                        logger.info(f"Futu US {code} K-line sync complete")
                    else:
                        raise RuntimeError("Futu not connected")
                except Exception as e:
                    logger.warning(f"Futu US {code} failed ({e}), trying YFinance...")
                    try:
                        self._sync_via_yfinance(code)
                    except Exception as e2:
                        logger.error(f"YFinance {code} also failed: {e2}")

            # YFinance-only codes
            for code in yf_only_codes:
                try:
                    self._sync_via_yfinance(code)
                except Exception as e:
                    logger.error(f"YFinance {code} failed: {e}")

            if self.qlib_writer:
                self.qlib_writer.flush()
            logger.info(f"US morning job complete, duration {(datetime.now()-job_start).total_seconds():.1f}s")
        except Exception as e:
            logger.error(f"US morning job failed: {e}")
        finally:
            if self.client:
                self.client.disconnect()

    def run_macro_job(self):
        """Morning job: collect macro indicator data (VIX, DXY, Treasury yield, etc.)."""
        job_start = datetime.now()
        logger.info(f"Starting macro data collection ({job_start.strftime('%Y-%m-%d %H:%M:%S')})")
        try:
            self.db_engine = DBEngine(settings.data_path)
            self._init_qlib_writer()
            yf_client = YFinanceClient()
            for code in YFinanceClient.MACRO_SYMBOLS:
                try:
                    if self.qlib_writer:
                        max_date = self.qlib_writer.get_stock_last_date(code)
                    else:
                        max_date = self.db_engine.get_kline_max_date(code, "K_DAY")
                    if max_date is not None:
                        start = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                        today = datetime.now().strftime("%Y-%m-%d")
                        if max_date >= today:
                            logger.info(f"Macro {code} up to date, skipping")
                            continue
                    else:
                        start = "2006-01-01"
                    data = yf_client.get_history_kline(code, start=start)
                    if data:
                        if self.qlib_writer:
                            n = self.qlib_writer.write_stock_records(code, data)
                            if n > 0:
                                logger.info(f"Macro {code} → qlib: +{n} days")
                        else:
                            self.db_engine.append_kline(pd.DataFrame(data), code, "K_DAY")
                            logger.info(f"Macro {code} sync complete: {len(data)} records")
                except Exception as e:
                    logger.error(f"Macro {code} collection failed: {e}")
            # HSTECH via Futu subscribe+cur_kline (Yahoo ^HSTECH delisted,
            # request_history_kline has quota limits for index codes)
            try:
                from futu import OpenQuoteContext, SubType, KLType, RET_OK
                hstech_code = "MACRO.HSTECH"
                if self.qlib_writer:
                    max_date = self.qlib_writer.get_stock_last_date(hstech_code)
                else:
                    max_date = self.db_engine.get_kline_max_date(hstech_code, "K_DAY")
                today = datetime.now().strftime("%Y-%m-%d")
                if max_date is not None and max_date >= today:
                    logger.info(f"Macro {hstech_code} up to date, skipping")
                else:
                    ctx = OpenQuoteContext(host=settings.futu_host, port=settings.futu_port)
                    try:
                        ret, _ = ctx.subscribe(["HK.800700"], [SubType.K_DAY])
                        if ret != RET_OK:
                            raise RuntimeError(f"Subscribe HK.800700 failed: {_}")
                        ret, kline = ctx.get_cur_kline("HK.800700", 100, KLType.K_DAY)
                        if ret != RET_OK or kline is None or kline.empty:
                            raise RuntimeError(f"get_cur_kline failed: {kline}")
                        # Convert to Futu-compatible records, filter by max_date
                        records = []
                        for _, row in kline.iterrows():
                            day = row["time_key"][:10]
                            if max_date and day <= max_date:
                                continue
                            records.append({
                                "code": hstech_code,
                                "time_key": row["time_key"],
                                "open": float(row["open"]),
                                "close": float(row["close"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "volume": int(row.get("volume", 0)),
                                "turnover": float(row.get("turnover", 0)),
                                "pe_ratio": 0.0,
                                "turnover_rate": 0.0,
                                "change_rate": 0.0,
                            })
                        if records and self.qlib_writer:
                            n = self.qlib_writer.write_stock_records(hstech_code, records)
                            if n > 0:
                                logger.info(f"Macro {hstech_code} (Futu cur_kline HK.800700) → qlib: +{n} days")
                        elif records:
                            self.db_engine.append_kline(pd.DataFrame(records), hstech_code, "K_DAY")
                            logger.info(f"Macro {hstech_code} sync complete: {len(records)} records")
                        else:
                            logger.info(f"Macro {hstech_code} no new records")
                    finally:
                        ctx.close()
            except Exception as e:
                logger.error(f"Macro HSTECH (Futu) collection failed: {e}")
            if self.qlib_writer:
                self.qlib_writer.flush()
            logger.info(f"Macro data collection complete, duration {(datetime.now()-job_start).total_seconds():.1f}s")
        except Exception as e:
            logger.error(f"Macro data collection job failed: {e}")

    def run_weekly_job(self):
        """Weekly Monday job: refresh industry plate mapping."""
        logger.info("Starting weekly industry plate mapping refresh...")
        try:
            self.client = FutuClient(settings.futu_host, settings.futu_port)
            self.db_engine = DBEngine(settings.data_path)
            self._init_qlib_writer()
            if not self.client.connect():
                raise RuntimeError("Cannot connect to Futu OpenD")
            self.sync_industry_map()
            if self.qlib_writer:
                self.qlib_writer.flush()
            logger.info("Industry plate mapping refresh complete")
        except Exception as e:
            logger.error(f"Weekly job failed: {e}")
        finally:
            if self.client:
                self.client.disconnect()

    def start(self):
        """Start the scheduler."""
        # Weekday 18:00 main job (HK/A-share K-line + fundamentals + short sell).
        # Baostock/Futu daily bars may lag shortly after the close; moving this
        # later avoids the local evening inference pipeline missing the latest
        # A-share session and falling back to stale signals.
        self.scheduler.add_job(
            self.run_daily_job,
            trigger=CronTrigger(day_of_week='mon-fri', hour=18, minute=0, timezone='Asia/Shanghai'),
            id='daily_data_sync',
            name='Daily data sync',
            replace_existing=True,
            misfire_grace_time=3600,
        )
        # Daily 07:00 US stock K-line
        self.scheduler.add_job(
            self.run_us_morning_job,
            trigger=CronTrigger(hour=7, minute=0, timezone='Asia/Shanghai'),
            id='us_morning_sync',
            name='US morning data sync',
            replace_existing=True,
            misfire_grace_time=3600,
        )
        # Weekly Monday 08:00 industry plate refresh
        self.scheduler.add_job(
            self.run_weekly_job,
            trigger=CronTrigger(day_of_week='mon', hour=8, minute=0, timezone='Asia/Shanghai'),
            id='weekly_industry_sync',
            name='Weekly industry plate refresh',
            replace_existing=True,
            misfire_grace_time=3600,
        )
        # Daily 07:30 macro indicators
        self.scheduler.add_job(
            self.run_macro_job,
            trigger=CronTrigger(hour=7, minute=30, timezone='Asia/Shanghai'),
            id='macro_data_sync',
            name='Macro indicator sync',
            replace_existing=True,
            misfire_grace_time=3600,
        )

        logger.info("Scheduler started: weekdays 18:00 main | daily 07:00 US+ETF | 07:30 macro | weekly Mon 08:00 industry")
        if settings.index_list:
            logger.info(f"Target indexes: {', '.join(settings.index_list)}")
        if settings.extra_codes:
            logger.info(f"Extra codes: {', '.join(settings.extra_codes)}")

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")
