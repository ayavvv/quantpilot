"""Baostock client - A-share daily K-line data collection."""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
from loguru import logger


class BaostockClient:
    """A-share data client (powered by baostock, API-compatible with FutuClient)."""

    def __init__(self, rate_limit: float = 0.3, max_retries: int = 3):
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self._bs = None
        self._logged_in = False

    def _ensure_login(self):
        if not self._logged_in:
            import baostock as bs
            self._bs = bs
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"baostock login failed: {lg.error_msg}")
            self._logged_in = True
            logger.info("baostock: logged in")

    def close(self):
        if self._logged_in:
            self._bs.logout()
            self._logged_in = False
            logger.info("baostock: logged out")

    # --- Stock list -----------------------------------------------------------

    def get_a_share_list(self) -> List[str]:
        """
        Get all A-share stock codes in SH./SZ. format.
        Only returns currently listed stocks (status=1).
        """
        self._ensure_login()
        logger.info("baostock: fetching A-share stock list...")

        rs = self._bs.query_stock_basic(code_name="")
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=rs.fields)

        # Filter: type=1 (stock), status=1 (listed)
        df = df[(df["type"] == "1") & (df["status"] == "1")]

        codes = []
        for bs_code in df["code"].tolist():
            futu_code = self._to_futu_code(bs_code)
            if futu_code:
                codes.append(futu_code)

        logger.info(f"baostock: found {len(codes)} A-shares")
        return sorted(codes)

    def get_sh_stock_list(self) -> List[str]:
        """Get Shanghai-listed (SH.*) stocks only."""
        all_codes = self.get_a_share_list()
        sh_codes = [c for c in all_codes if c.startswith("SH.")]
        logger.info(f"baostock: Shanghai {len(sh_codes)} stocks")
        return sh_codes

    def get_trade_dates(self, start: str = None, end: str = None) -> List[str]:
        """Return trading dates in ``YYYY-MM-DD`` format within the range."""
        self._ensure_login()
        start_date = start or "2015-01-01"
        end_date = end or pd.Timestamp.now().strftime("%Y-%m-%d")

        rs = self._bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_code != "0":
            raise RuntimeError(f"query_trade_dates error: {rs.error_msg}")

        field_map = {name: idx for idx, name in enumerate(rs.fields)}
        cal_idx = field_map.get("calendar_date")
        trade_idx = field_map.get("is_trading_day")
        if cal_idx is None or trade_idx is None:
            raise RuntimeError(f"query_trade_dates unexpected fields: {rs.fields}")

        dates: List[str] = []
        while rs.next():
            row = rs.get_row_data()
            if row[trade_idx] == "1":
                dates.append(row[cal_idx])
        return dates

    def latest_trade_date(self, on_or_before: str = None, lookback_days: int = 31) -> Optional[str]:
        """Return the latest A-share trading day on or before the given date."""
        end_date = on_or_before or pd.Timestamp.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")
        dates = self.get_trade_dates(start=start_date, end=end_date)
        return dates[-1] if dates else None

    # --- Daily K-line ---------------------------------------------------------

    def get_history_kline(
        self,
        code: str,
        start: str = None,
        end: str = None,
        ktype: str = "K_DAY",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical daily K-line data, output format compatible with FutuClient.

        Args:
            code: Stock code (SH.600000 format)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            ktype: K-line type (only K_DAY supported)
        """
        if ktype != "K_DAY":
            logger.warning(f"baostock only supports daily K-line, skipping {code} {ktype}")
            return []

        self._ensure_login()
        bs_code = self._to_bs_code(code)
        if not bs_code:
            logger.warning(f"Cannot convert code: {code}")
            return []

        start_date = start or "1990-01-01"
        end_date = end or pd.Timestamp.now().strftime("%Y-%m-%d")

        for attempt in range(self.max_retries):
            try:
                rs = self._bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,open,high,low,close,volume,amount,turn,pctChg",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="2",  # forward-adjusted
                )
                if rs.error_code != "0":
                    raise RuntimeError(f"query error: {rs.error_msg}")

                data = []
                while rs.next():
                    data.append(rs.get_row_data())

                time.sleep(self.rate_limit)

                if not data:
                    return []

                df = pd.DataFrame(data, columns=rs.fields)
                return self._convert_kline(df, code)

            except Exception as e:
                logger.warning(f"baostock {code} attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit * 2)
                else:
                    logger.error(f"baostock {code} all retries exhausted")
                    return []

    # --- Format conversion ----------------------------------------------------

    @staticmethod
    def _to_bs_code(futu_code: str) -> Optional[str]:
        """SH.600000 -> sh.600000"""
        if futu_code.startswith(("SH.", "SZ.")):
            return futu_code.lower().replace(".", ".")
        return None

    @staticmethod
    def _to_futu_code(bs_code: str) -> Optional[str]:
        """sh.600000 -> SH.600000, sz.000001 -> SZ.000001"""
        code = str(bs_code).strip()
        if code.startswith("sh."):
            num = code[3:]
            if num.startswith("6"):
                return f"SH.{num}"
        elif code.startswith("sz."):
            num = code[3:]
            if num.startswith(("0", "3")):
                return f"SZ.{num}"
        return None

    @staticmethod
    def _convert_kline(df: pd.DataFrame, code: str) -> List[Dict[str, Any]]:
        """
        Convert baostock daily DataFrame to Futu-compatible dict list.

        baostock columns: date, code, open, high, low, close, volume, amount, turn, pctChg
        Futu columns:     code, time_key, open, close, high, low, volume, turnover, pe_ratio, turnover_rate, change_rate
        """
        records = []
        for _, row in df.iterrows():
            try:
                records.append({
                    "code": code,
                    "time_key": str(row["date"]) + " 00:00:00",
                    "open": float(row["open"]),
                    "close": float(row["close"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "volume": int(float(row["volume"])),
                    "turnover": float(row["amount"]),
                    "pe_ratio": 0.0,
                    "turnover_rate": float(row["turn"]) if row["turn"] else 0.0,
                    "change_rate": float(row["pctChg"]) if row["pctChg"] else 0.0,
                })
            except (ValueError, TypeError):
                continue
        return records
