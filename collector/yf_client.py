"""Yahoo Finance data source - fallback for assets not covered by Futu."""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

import pandas as pd
from loguru import logger


class YFinanceClient:
    """Yahoo Finance client for collecting data on assets not covered by Futu."""

    # Futu code -> Yahoo symbol mapping
    CODE_MAP = {
        "US.SPY": "SPY",
        "US.QQQ": "QQQ",
        "US.YINN": "YINN",
        "US.CQQQ": "CQQQ",
        "US.FXI": "FXI",
        "US.KWEB": "KWEB",
        "US.EEM": "EEM",
        "US.TLT": "TLT",
        "US.GLD": "GLD",
        "US.VIX": "^VIX",
        "HK.800700": "^HSTECH",
    }

    # Macro indices
    MACRO_SYMBOLS = {
        "MACRO.VIX": "^VIX",
        "MACRO.DXY": "DX-Y.NYB",
        "MACRO.TNX": "^TNX",        # US 10-year Treasury yield
        "MACRO.HSI": "^HSI",        # Hang Seng Index (backup)
        # MACRO.HSTECH moved to Futu (HK.800700) — Yahoo ^HSTECH returns 404
    }

    def get_history_kline(
        self,
        code: str,
        start: str = None,
        end: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical daily data.

        Args:
            code: Futu-format code (e.g. US.SPY) or MACRO.xxx
            start: Start date YYYY-MM-DD
            end: End date YYYY-MM-DD

        Returns:
            Futu K-line compatible dict list
        """
        import yfinance as yf

        yf_symbol = self.CODE_MAP.get(code) or self.MACRO_SYMBOLS.get(code)
        if not yf_symbol:
            logger.warning(f"YFinance unmapped code: {code}")
            return []

        if start is None:
            start = "2006-01-01"
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"YFinance fetching {code} ({yf_symbol}): {start} ~ {end}")

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(start=start, end=end, auto_adjust=False)
                if df is None or df.empty:
                    logger.warning(f"YFinance {yf_symbol} no data")
                    return []

                # Convert to Futu-compatible format
                records = []
                for idx, row in df.iterrows():
                    records.append({
                        "code": code,
                        "time_key": idx.strftime("%Y-%m-%d 00:00:00"),
                        "open": float(row.get("Open", 0)),
                        "close": float(row.get("Close", 0)),
                        "high": float(row.get("High", 0)),
                        "low": float(row.get("Low", 0)),
                        "volume": int(row.get("Volume", 0)),
                        "turnover": float(row.get("Volume", 0) * row.get("Close", 0)),
                        "pe_ratio": 0.0,
                        "turnover_rate": 0.0,
                        "change_rate": 0.0,
                    })

                logger.info(f"YFinance {code}: got {len(records)} daily records")
                return records

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"YFinance fetch {code} failed (attempt {attempt}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"YFinance fetch {code} failed after {max_retries} attempts: {e}")
                    return []

    def get_macro_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch fetch macro indicator data.

        Returns:
            {code: [records...]}
        """
        results = {}
        for code in self.MACRO_SYMBOLS:
            data = self.get_history_kline(code)
            if data:
                results[code] = data
        return results

    @classmethod
    def can_handle(cls, code: str) -> bool:
        """Check if this code can be handled by YFinance."""
        return code in cls.CODE_MAP or code in cls.MACRO_SYMBOLS
