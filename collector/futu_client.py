"""Futu API client - with pagination, retry, and rate limiting."""
import time
from typing import Optional, List, Dict, Any
import pandas as pd
from futu import OpenQuoteContext, RET_OK, RET_ERROR, TrdEnv, Plate, SysConfig
from loguru import logger


class FutuClient:
    """Futu OpenD client wrapper."""

    def __init__(self, host: str, port: int):
        """
        Initialize client.

        Args:
            host: Futu OpenD host address
            port: Futu OpenD port
        """
        self.host = host
        self.port = port
        self.ctx: Optional[OpenQuoteContext] = None
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.rate_limit_delay = 0.6  # seconds
        self.connect_timeout = 30  # seconds

        # Enable protocol encryption for cross-network connections
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file("/app/keys/futu_rsa_1024.pem")

    def connect(self) -> bool:
        """
        Establish connection (with timeout protection).

        Returns:
            Whether connection was successful
        """
        import threading

        result = {"ok": False, "error": None}
        # Shared holder so background thread can store ctx for cleanup
        holder = {"ctx": None, "abandoned": False}

        def _do_connect():
            try:
                ctx = OpenQuoteContext(host=self.host, port=self.port)
                holder["ctx"] = ctx
                if holder["abandoned"]:
                    # Parent timed out, clean up immediately
                    try:
                        ctx.close()
                    except Exception:
                        pass
                    return
                self.ctx = ctx
                ret, data = ctx.get_market_state(['HK.00700'])
                if holder["abandoned"]:
                    try:
                        ctx.close()
                    except Exception:
                        pass
                    self.ctx = None
                    return
                if ret == RET_OK:
                    result["ok"] = True
                else:
                    result["error"] = f"Connection test failed: {data}"
            except Exception as e:
                result["error"] = str(e)

        t = threading.Thread(target=_do_connect, daemon=True)
        t.start()
        t.join(timeout=self.connect_timeout)

        if t.is_alive():
            logger.error(f"Futu OpenD connection timeout ({self.connect_timeout}s), target: {self.host}:{self.port}")
            holder["abandoned"] = True
            # Clean up if ctx was already created
            ctx = holder.get("ctx") or self.ctx
            if ctx:
                try:
                    ctx.close()
                except Exception:
                    pass
            self.ctx = None
            return False

        if result["ok"]:
            logger.info(f"Connected to Futu OpenD ({self.host}:{self.port})")
            return True
        else:
            logger.error(f"Futu OpenD connection error: {result['error']}")
            return False

    def disconnect(self):
        """Close connection."""
        if self.ctx:
            try:
                self.ctx.close()
                logger.info("Disconnected from Futu OpenD")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self.ctx = None

    def _retry_wrapper(self, func, *args, **kwargs):
        """
        Retry wrapper.

        Args:
            func: Function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function return value
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}"
                )

                # Network or timeout errors
                if "NetWorkError" in error_msg or "timeout" in error_msg.lower():
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue

                # Last attempt failed
                if attempt == self.max_retries - 1:
                    raise last_error

        raise last_error

    def get_index_constituents(self, index_code: str) -> List[str]:
        """
        Get all constituents of a given index.

        Args:
            index_code: Index code (e.g. 'HK.800000')

        Returns:
            List of constituent stock codes
        """
        if not self.ctx:
            raise RuntimeError("Client not connected, call connect() first")

        logger.info(f"Fetching constituents for index {index_code}")

        try:
            def _request():
                return self.ctx.get_plate_stock(index_code)

            result = self._retry_wrapper(_request)
            if isinstance(result, tuple):
                ret, data = result[0], result[1]
            else:
                logger.error(f"Unexpected Futu API return type: {type(result)}")
                raise RuntimeError(f"Unexpected Futu API return type: {type(result)}")

            if ret == RET_ERROR:
                error_msg = f"Failed to get constituents for {index_code}: {data}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if data is None or len(data) == 0:
                logger.warning(f"Index {index_code} constituent list is empty")
                return []

            # Extract stock code list
            if isinstance(data, pd.DataFrame):
                if 'code' in data.columns:
                    codes = data['code'].tolist()
                else:
                    possible_cols = ['stock_code', 'stockCode', 'code']
                    codes = []
                    for col in possible_cols:
                        if col in data.columns:
                            codes = data[col].tolist()
                            break
                    if not codes:
                        logger.error(f"No stock code column found, available columns: {data.columns.tolist()}")
                        raise RuntimeError("Cannot extract stock codes from DataFrame")
            else:
                logger.error(f"Cannot parse constituent data format: {type(data)}")
                raise RuntimeError(f"Cannot parse constituent data format: {type(data)}")

            # Filter empty values
            codes = [str(code).strip() for code in codes if code and str(code).strip()]

            logger.info(f"Got {len(codes)} constituents for index {index_code}")
            return codes

        except Exception as e:
            logger.error(f"Error fetching constituents for {index_code}: {e}")
            raise

    def get_history_kline(
        self,
        code: str,
        start: str = None,
        end: str = None,
        ktype: str = "K_DAY",
        autype: str = "qfq",
        max_count: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get historical K-line data (using Futu page_req_key pagination).

        Args:
            code: Stock code
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            ktype: K-line type (K_DAY, K_1M, etc.)
            autype: Adjustment type (qfq: forward, hfq: backward, None: none)
            max_count: Max records per request (recommended 1000)

        Returns:
            K-line data list
        """
        if not self.ctx:
            raise RuntimeError("Client not connected, call connect() first")

        all_data = []
        page_req_key = None
        max_iterations = 10000
        iteration = 0
        consecutive_empty = 0

        logger.info(f"Fetching {code} {ktype} data (start={start}, end={end})")

        while iteration < max_iterations:
            iteration += 1
            try:
                def _request():
                    return self.ctx.request_history_kline(
                        code=code,
                        start=start,
                        end=end,
                        ktype=ktype,
                        autype=autype,
                        max_count=max_count,
                        page_req_key=page_req_key,
                    )

                result = self._retry_wrapper(_request)
                if not isinstance(result, tuple) or len(result) < 3:
                    logger.error(f"Unexpected Futu API return type: {type(result)}")
                    raise RuntimeError("Unexpected Futu API return, expected (ret, data, page_req_key)")

                ret, data, page_req_key = result[0], result[1], result[2]

                if ret == RET_ERROR:
                    error_msg = f"Failed to get K-line data: {data}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                if data is not None and len(data) > 0:
                    consecutive_empty = 0
                    page_data = data.to_dict("records")
                    all_data.extend(page_data)
                    logger.info(f"Got {len(page_data)} records, total {len(all_data)} (page {iteration})")
                else:
                    consecutive_empty += 1
                    logger.warning(f"Page {iteration} returned empty data (consecutive {consecutive_empty})")

                if page_req_key is None:
                    logger.info("page_req_key is None, pagination complete")
                    break

                if consecutive_empty >= 3:
                    logger.warning(f"Consecutive {consecutive_empty} empty pages, exiting pagination")
                    break

            except Exception as e:
                logger.error(f"K-line data error: {e}")
                raise

        if iteration >= max_iterations:
            logger.error(f"Max iteration limit reached ({max_iterations}), forced exit")
            raise RuntimeError(f"Pagination loop reached max iterations ({max_iterations})")

        logger.info(f"Completed fetching {code} {ktype} data, total {len(all_data)} records")
        return all_data

    def get_rt_ticker(
        self,
        code: str,
        start: str = None,
        end: str = None,
        max_count: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get real-time tick data.

        Args:
            code: Stock code
            start: Start time (YYYY-MM-DD HH:MM:SS) - for filtering
            end: End time (YYYY-MM-DD HH:MM:SS) - for filtering
            max_count: Max records per request (Futu API max 1000)

        Returns:
            Tick data list
        """
        if not self.ctx:
            raise RuntimeError("Client not connected, call connect() first")

        logger.info(f"Fetching {code} tick data (start={start}, end={end})")

        try:
            def _request():
                return self.ctx.get_rt_ticker(code=code, num=min(max_count, 1000))

            result = self._retry_wrapper(_request)
            if isinstance(result, tuple):
                ret, data = result[0], result[1]
            else:
                logger.error(f"Unexpected Futu API return type: {type(result)}")
                raise RuntimeError(f"Unexpected Futu API return type: {type(result)}")

            if ret == RET_ERROR:
                error_msg = f"Failed to get tick data: {data}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if data is None or len(data) == 0:
                logger.warning(f"{code} tick data is empty")
                return []

            all_data = data.to_dict('records')

            # Filter by time range if specified
            if start or end:
                from datetime import datetime
                filtered_data = []
                start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S") if start else None
                end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S") if end else None

                for record in all_data:
                    record_time_str = record.get('time', '')
                    if not record_time_str:
                        continue

                    try:
                        if '.' in record_time_str:
                            record_dt = datetime.strptime(record_time_str, "%Y-%m-%d %H:%M:%S.%f")
                        else:
                            record_dt = datetime.strptime(record_time_str, "%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        continue

                    if start_dt and record_dt < start_dt:
                        continue
                    if end_dt and record_dt > end_dt:
                        continue

                    filtered_data.append(record)

                all_data = filtered_data
                logger.info(f"Filtered to {len(all_data)} records (original {len(data)})")
            else:
                logger.info(f"Got {len(all_data)} tick records")

            logger.info(f"Completed fetching {code} tick data, total {len(all_data)} records")
            return all_data

        except Exception as e:
            logger.error(f"Tick data error: {e}")
            raise

    # -------------------------------------------------------------------------
    # Fundamentals snapshot
    # -------------------------------------------------------------------------

    def get_fundamentals(self, codes: List[str]) -> List[Dict[str, Any]]:
        """
        Batch fetch stock fundamental snapshots (PB, dividend yield, EPS, ROE, etc.).
        Uses get_market_snapshot, max 200 per batch.

        Returns:
            [{"code", "date", "pb_ratio", "dividend_ttm", "net_profit_ttm",
              "return_on_equity", "net_profit_growth_rate"}, ...]
        """
        if not self.ctx:
            raise RuntimeError("Client not connected")

        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        results = []
        batch_size = 200

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            def _request(b=batch):
                return self.ctx.get_market_snapshot(b)

            try:
                ret, data = self._retry_wrapper(_request)
            except Exception as e:
                logger.warning(f"get_market_snapshot batch {i//batch_size} failed: {e}")
                continue

            if ret != RET_OK or data is None or data.empty:
                logger.warning(f"get_market_snapshot batch {i//batch_size} returned empty")
                continue

            keep = ["code", "pb_ratio", "dividend_ttm", "net_profit_ttm",
                    "return_on_equity", "net_profit_growth_rate", "lot_size"]
            row_cols = [c for c in keep if c in data.columns]
            for _, row in data[row_cols].iterrows():
                r = row.to_dict()
                r["date"] = today
                results.append(r)

        logger.info(f"Fundamentals snapshot: collected {len(results)} records")
        return results

    # -------------------------------------------------------------------------
    # Industry plates
    # -------------------------------------------------------------------------

    def get_industry_map(self, market: str = "HK") -> List[Dict[str, Any]]:
        """
        Get industry plate -> constituent mapping.
        Iterates all INDUSTRY plates, returns [{"code", "industry"}].
        """
        if not self.ctx:
            raise RuntimeError("Client not connected")

        logger.info(f"Fetching {market} industry plate list...")
        def _get_plates():
            return self.ctx.get_plate_list(market, Plate.INDUSTRY)

        try:
            ret, plates = self._retry_wrapper(_get_plates)
        except Exception as e:
            logger.error(f"get_plate_list failed: {e}")
            return []

        if ret != RET_OK or plates is None or plates.empty:
            logger.warning("Industry plate list is empty")
            return []

        results = []
        plate_codes = plates["plate_code"].tolist() if "plate_code" in plates.columns else []
        plate_names = plates.set_index("plate_code")["plate_name"].to_dict() if "plate_name" in plates.columns else {}

        logger.info(f"Found {len(plate_codes)} industry plates, fetching constituents...")
        for plate_code in plate_codes:
            industry_name = plate_names.get(plate_code, plate_code)
            def _get_stocks(pc=plate_code):
                return self.ctx.get_plate_stock(pc)
            try:
                ret2, stocks = self._retry_wrapper(_get_stocks)
                if ret2 != RET_OK or stocks is None or stocks.empty:
                    continue
                for code in stocks["code"].tolist():
                    results.append({"code": code, "plate_code": plate_code, "industry": industry_name})
            except Exception as e:
                logger.warning(f"Plate {plate_code} constituent fetch failed: {e}")

        logger.info(f"Industry mapping complete: {len(results)} records")
        return results

    # -------------------------------------------------------------------------
    # HK short selling data
    # -------------------------------------------------------------------------

    def get_short_sell_list(self, market: str = "HK") -> List[Dict[str, Any]]:
        """
        Get HK short selling data (current day).
        Returns short volume, amount, and ratio.
        """
        if not self.ctx:
            raise RuntimeError("Client not connected")

        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        logger.info(f"Fetching {market} short sell data ({today})...")

        try:
            def _request():
                return self.ctx.get_short_sell_list(market)

            ret, data = self._retry_wrapper(_request)
        except Exception as e:
            logger.warning(f"get_short_sell_list failed: {e}")
            return []

        if ret != RET_OK or data is None or data.empty:
            logger.warning("Short sell data is empty (may require subscription)")
            return []

        keep = ["code", "short_sell_qty", "short_sell_amount", "short_sell_ratio",
                "short_avg_price", "short_buy_qty", "short_buy_amount"]
        row_cols = [c for c in keep if c in data.columns]
        results = data[row_cols].copy()
        results["date"] = today
        logger.info(f"Short sell data: {len(results)} records")
        return results.to_dict("records")
