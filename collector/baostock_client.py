"""Baostock client - A-share daily K-line data collection."""

import time
import socket
import os
from datetime import datetime, timedelta
from importlib import metadata
from typing import List, Dict, Any, Optional

import pandas as pd
from loguru import logger


class BaostockClient:
    """A-share data client (powered by baostock, API-compatible with FutuClient)."""

    LOGIN_EXPIRED_MARKERS = ("用户未登录", "未登录")
    DEFAULT_SERVER_HOST = "public-api.baostock.com"

    def __init__(self, rate_limit: float = 0.3, max_retries: int = 3, socket_timeout: float = 20.0):
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.socket_timeout = socket_timeout
        self._bs = None
        self._logged_in = False
        self._last_history_kline_status: dict[str, Any] | None = None

    def _ensure_login(self):
        if not self._logged_in:
            import baostock as bs
            self._bs = bs
            self._configure_baostock_server()
            self._patch_baostock_socket(bs)

            previous_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(self.socket_timeout)
            try:
                lg = bs.login()
            except Exception:
                self._close_default_socket()
                raise
            finally:
                socket.setdefaulttimeout(previous_timeout)
            if lg.error_code != "0":
                raise RuntimeError(f"baostock login failed: {lg.error_msg}")
            self._logged_in = True
            logger.info("baostock: logged in")

    def _configure_baostock_server(self):
        """Pin baostock to the current public socket API endpoint.

        Older baostock releases default to www.baostock.com:10030. That host can
        still accept TCP connections but closes the baostock protocol stream
        before a full response. public-api.baostock.com is the endpoint used by
        current baostock releases and works from the NAS container.
        """
        try:
            import baostock.common.contants as cons
        except Exception as exc:
            raise RuntimeError(f"failed to load baostock constants: {exc}") from exc

        target_host = os.environ.get("BAOSTOCK_SERVER_HOST", self.DEFAULT_SERVER_HOST).strip()
        if not target_host:
            target_host = self.DEFAULT_SERVER_HOST

        current_host = getattr(cons, "BAOSTOCK_SERVER_IP", "")
        current_port = getattr(cons, "BAOSTOCK_SERVER_PORT", "")
        if current_host != target_host:
            logger.warning(
                "baostock: overriding server {}:{} -> {}:{}",
                current_host or "N/A",
                current_port or "N/A",
                target_host,
                current_port or "N/A",
            )
            cons.BAOSTOCK_SERVER_IP = target_host

        try:
            version = metadata.version("baostock")
        except metadata.PackageNotFoundError:
            version = "unknown"
        logger.info(
            "baostock: package version={} server={}:{}",
            version,
            getattr(cons, "BAOSTOCK_SERVER_IP", "N/A"),
            getattr(cons, "BAOSTOCK_SERVER_PORT", "N/A"),
        )

    def _patch_baostock_socket(self, bs_module):
        """Patch baostock's blocking socket loop so closed sockets cannot spin forever."""
        try:
            import baostock.common.contants as cons
            import baostock.common.context as context
            import baostock.util.socketutil as socketutil
            import zlib
        except Exception as exc:
            raise RuntimeError(f"failed to patch baostock socket handling: {exc}") from exc

        if getattr(socketutil, "_quantpilot_safe_send_msg", False):
            socketutil._quantpilot_socket_timeout = self.socket_timeout
            return

        def safe_send_msg(msg):
            if not hasattr(context, "default_socket"):
                print("you don't login.")
                return None

            default_socket = getattr(context, "default_socket")
            if default_socket is None:
                return None

            timeout = getattr(socketutil, "_quantpilot_socket_timeout", self.socket_timeout)
            default_socket.settimeout(timeout)
            default_socket.sendall(bytes(msg + "\n", encoding="utf-8"))

            receive = b""
            while True:
                try:
                    recv = default_socket.recv(8192)
                except socket.timeout as exc:
                    raise TimeoutError(f"baostock socket recv timed out after {timeout}s") from exc

                if recv == b"":
                    raise ConnectionError("baostock socket closed before full response")

                receive += recv
                if receive[-13:] == b"<![CDATA[]]>\n":
                    break

            head_bytes = receive[0:cons.MESSAGE_HEADER_LENGTH]
            head_str = bytes.decode(head_bytes)
            head_arr = head_str.split(cons.MESSAGE_SPLIT)
            if head_arr[1] in cons.COMPRESSED_MESSAGE_TYPE_TUPLE:
                head_inner_length = int(head_arr[2])
                body_bytes = receive[
                    cons.MESSAGE_HEADER_LENGTH:cons.MESSAGE_HEADER_LENGTH + head_inner_length
                ]
                body_str = bytes.decode(zlib.decompress(body_bytes))
                return head_str + body_str
            return bytes.decode(receive)

        socketutil.send_msg = safe_send_msg
        socketutil._quantpilot_safe_send_msg = True
        socketutil._quantpilot_socket_timeout = self.socket_timeout

    def _reset_login_state(self, close_socket: bool = True):
        if close_socket:
            self._close_default_socket()
        self._logged_in = False

    def _close_default_socket(self):
        try:
            import baostock.common.context as context

            default_socket = getattr(context, "default_socket", None)
            if default_socket is not None:
                default_socket.close()
            setattr(context, "default_socket", None)
        except Exception as exc:
            logger.debug(f"baostock socket cleanup skipped: {exc}")

    def _is_login_expired_error(self, error_msg: str) -> bool:
        return any(marker in error_msg for marker in self.LOGIN_EXPIRED_MARKERS)

    def _run_query(self, query_fn, *args, allow_relogin: bool = True, **kwargs):
        self._ensure_login()
        try:
            rs = query_fn(*args, **kwargs)
        except (TimeoutError, ConnectionError, OSError) as exc:
            if not allow_relogin:
                raise
            logger.warning(f"baostock socket error, re-authenticating: {exc}")
            self._reset_login_state()
            self._ensure_login()
            rs = query_fn(*args, **kwargs)
        if rs.error_code == "0":
            return rs

        error_msg = rs.error_msg or f"error_code={rs.error_code}"
        if allow_relogin and self._is_login_expired_error(error_msg):
            logger.warning("baostock session expired, re-authenticating...")
            self._reset_login_state()
            self._ensure_login()
            rs = query_fn(*args, **kwargs)
            if rs.error_code == "0":
                return rs
            error_msg = rs.error_msg or f"error_code={rs.error_code}"

        raise RuntimeError(f"query error: {error_msg}")

    def close(self):
        if self._logged_in:
            try:
                self._bs.logout()
                logger.info("baostock: logged out")
            except Exception as exc:
                logger.warning(f"baostock logout failed: {exc}")
            finally:
                self._reset_login_state()

    # --- Stock list -----------------------------------------------------------

    def get_a_share_basic(self) -> pd.DataFrame:
        """
        Get currently listed A-share basic metadata in Futu code format.

        Baostock exposes current stock names through query_stock_basic. Downstream
        strategy code uses this metadata to exclude ST/*ST names before training
        and signal generation.
        """
        logger.info("baostock: fetching A-share stock basic metadata...")

        rs = self._run_query(lambda **kwargs: self._bs.query_stock_basic(**kwargs), code_name="")
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=rs.fields)

        if df.empty:
            return pd.DataFrame(columns=["code", "name", "ipoDate", "outDate", "type", "status"])

        # Filter: type=1 (stock), status=1 (listed), then convert sh/sz code to Futu style.
        df = df[(df["type"] == "1") & (df["status"] == "1")].copy()
        df["code"] = df["code"].map(self._to_futu_code)
        df = df.dropna(subset=["code"])
        if "code_name" in df.columns:
            df["name"] = df["code_name"].astype(str)
        elif "name" not in df.columns:
            df["name"] = ""

        keep_cols = [col for col in ["code", "name", "ipoDate", "outDate", "type", "status"] if col in df.columns]
        df = df[keep_cols].drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)
        logger.info(f"baostock: found {len(df)} listed A-share stock metadata rows")
        return df

    def get_a_share_list(self) -> List[str]:
        """
        Get all A-share stock codes in SH./SZ. format.
        Only returns currently listed stocks (status=1).
        """
        df = self.get_a_share_basic()
        codes = df["code"].astype(str).tolist() if not df.empty else []
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
        start_date = start or "2015-01-01"
        end_date = end or pd.Timestamp.now().strftime("%Y-%m-%d")

        rs = self._run_query(
            lambda **kwargs: self._bs.query_trade_dates(**kwargs),
            start_date=start_date,
            end_date=end_date,
        )

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

    def get_last_history_kline_status(self) -> dict[str, Any] | None:
        """Return the latest history K-line query status for diagnostics."""
        return self._last_history_kline_status

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
        self._last_history_kline_status = None

        if ktype != "K_DAY":
            logger.warning(f"baostock only supports daily K-line, skipping {code} {ktype}")
            self._last_history_kline_status = {
                "code": code,
                "start": start,
                "end": end,
                "ktype": ktype,
                "status": "unsupported_ktype",
            }
            return []

        bs_code = self._to_bs_code(code)
        if not bs_code:
            logger.warning(f"Cannot convert code: {code}")
            self._last_history_kline_status = {
                "code": code,
                "start": start,
                "end": end,
                "ktype": ktype,
                "status": "invalid_code",
            }
            return []

        start_date = start or "1990-01-01"
        end_date = end or pd.Timestamp.now().strftime("%Y-%m-%d")
        last_error = None

        for attempt in range(self.max_retries):
            try:
                rs = self._run_query(
                    lambda *query_args, **query_kwargs: self._bs.query_history_k_data_plus(*query_args, **query_kwargs),
                    bs_code,
                    "date,code,open,high,low,close,volume,amount,turn,pctChg,isST",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="2",  # forward-adjusted
                )

                data = []
                while rs.next():
                    data.append(rs.get_row_data())

                time.sleep(self.rate_limit)

                if not data:
                    self._last_history_kline_status = {
                        "code": code,
                        "start": start_date,
                        "end": end_date,
                        "ktype": ktype,
                        "status": "empty_data",
                        "attempt": attempt + 1,
                    }
                    return []

                df = pd.DataFrame(data, columns=rs.fields)
                converted = self._convert_kline(df, code)
                self._last_history_kline_status = {
                    "code": code,
                    "start": start_date,
                    "end": end_date,
                    "ktype": ktype,
                    "status": "ok" if converted else "converted_empty",
                    "attempt": attempt + 1,
                    "rows": len(converted),
                    "raw_rows": len(df),
                }
                return converted

            except Exception as e:
                last_error = str(e)
                logger.warning(f"baostock {code} attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit * 2)
                else:
                    logger.error(f"baostock {code} all retries exhausted")
                    self._last_history_kline_status = {
                        "code": code,
                        "start": start_date,
                        "end": end_date,
                        "ktype": ktype,
                        "status": "query_failed",
                        "attempt": attempt + 1,
                        "error": last_error,
                    }
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

        baostock columns: date, code, open, high, low, close, volume, amount, turn, pctChg, isST
        Futu columns:     code, time_key, open, close, high, low, volume, turnover, pe_ratio, turnover_rate, change_rate, is_st
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
                    "is_st": float(row["isST"]) if "isST" in row and row["isST"] else 0.0,
                })
            except (ValueError, TypeError):
                continue
        return records
