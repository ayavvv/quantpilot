"""Scheduler - cron job orchestration for data collection."""
import os
from datetime import datetime, timedelta
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


class DataCollectorScheduler:
    """Data collection scheduler."""

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

    def sync_a_share_kline(self, code: str):
        """
        Sync A-share daily K-line using Baostock.
        Writes directly to Qlib bin format (no parquet intermediate).
        """
        end = datetime.now().strftime("%Y-%m-%d")

        # Check last collected date from Qlib bin
        if self.qlib_writer:
            max_date = self.qlib_writer.get_stock_last_date(code)
        else:
            max_date = self.db_engine.get_kline_max_date(code, "K_DAY")

        if max_date is not None:
            if max_date >= end:
                return  # Already up to date
            start = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365 * 15)).strftime("%Y-%m-%d")
            logger.info(f"Baostock {code} K_DAY first fetch: {start} ~ {end}")

        data = self.bs_client.get_history_kline(code, start=start, end=end, ktype="K_DAY")
        if data:
            # Write directly to Qlib bin
            if self.qlib_writer:
                n = self.qlib_writer.write_stock_records(code, data)
                if n > 0:
                    self.db_engine.log_job("success", f"Baostock {code} +{n} days (qlib)", code, "K_DAY")
            else:
                self.db_engine.append_kline(pd.DataFrame(data), code, "K_DAY")
                self.db_engine.log_job("success", f"Baostock {code} +{len(data)} records", code, "K_DAY")

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

            # 1. Get target stock pool

            # A-shares: via baostock
            a_share_codes = []
            try:
                a_share_codes = self.bs_client.get_a_share_list()
                logger.info(f"Baostock A-share targets: {len(a_share_codes)}")
            except Exception as e:
                logger.error(f"Baostock A-share list failed: {e}")

            # HK stocks: via Futu index constituents
            hk_codes = []
            futu_ok = False
            hk_indexes = [idx for idx in settings.index_list if idx.startswith("HK.")]
            if hk_indexes:
                try:
                    self.client = FutuClient(settings.futu_host, settings.futu_port)
                    if self.client.connect():
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
                except Exception as e:
                    logger.error(f"Futu connection failed: {e}")

            # Extra codes
            extra_codes = [c.strip() for c in settings.extra_codes if c.strip()]

            # 2. A-share daily K-line (Baostock)
            if a_share_codes:
                logger.info(f"=== Baostock A-share collection: {len(a_share_codes)} stocks ===")
                for idx, code in enumerate(a_share_codes, 1):
                    try:
                        self.sync_a_share_kline(code)
                        if idx % 50 == 0:
                            elapsed = (datetime.now() - job_start_time).total_seconds()
                            logger.info(
                                f"A-share progress: {idx}/{len(a_share_codes)} ({idx*100//len(a_share_codes)}%) | "
                                f"elapsed: {elapsed/60:.1f} min"
                            )
                    except Exception as e:
                        logger.error(f"[{idx}/{len(a_share_codes)}] Baostock {code} failed: {e}")
                        continue

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
        # Daily 16:30 main job (HK/A-share K-line + fundamentals + short sell)
        self.scheduler.add_job(
            self.run_daily_job,
            trigger=CronTrigger(hour=16, minute=30, timezone='Asia/Shanghai'),
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

        logger.info("Scheduler started: daily 16:30 main | daily 07:00 US+ETF | 07:30 macro | weekly Mon 08:00 industry")
        if settings.index_list:
            logger.info(f"Target indexes: {', '.join(settings.index_list)}")
        if settings.extra_codes:
            logger.info(f"Extra codes: {', '.join(settings.extra_codes)}")

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")
