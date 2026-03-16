"""Backtest configuration"""

import os
from pathlib import Path

# --- Data paths (configurable via env vars) ---
PRED_PKL_PATH = Path(os.environ.get("PRED_PKL_PATH", str(Path.home() / "quantpilot" / "models" / "pred.pkl")))
PRED_SH_PATH = Path(os.environ.get("PRED_SH_PATH", str(Path.home() / "quantpilot" / "models" / "pred_sh.pkl")))
PRED_HK_PATH = Path(os.environ.get("PRED_HK_PATH", str(Path.home() / "quantpilot" / "models" / "pred_hk.pkl")))
PRICE_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", str(Path.home() / "quantpilot" / "qlib_data")))
OUTPUT_DIR = Path(os.environ.get("BACKTEST_OUTPUT_DIR", str(Path(__file__).parent / "output")))

# --- Strategy parameters ---
TOP_N = int(os.environ.get("TOP_N", "20"))
SLIPPAGE = float(os.environ.get("SLIPPAGE", "0.001"))  # 0.1% per side

# --- Fee rates (per side) ---
FEE_CONFIG = {
    "HK": {
        "buy": 0.001 + 0.0003,    # stamp duty 0.1% + commission 0.03%
        "sell": 0.001 + 0.0003,
    },
    "SH": {
        "buy": 0.00025,           # commission 0.025% (no stamp duty on buy)
        "sell": 0.0005 + 0.00025, # stamp duty 0.05% + commission 0.025%
    },
    "US": {
        "buy": 0.0003,            # commission 0.03%
        "sell": 0.0003,
    },
}

# Non-tradeable code prefixes
NON_TRADEABLE_PREFIXES = ("MACRO.",)
