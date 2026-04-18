"""Backtest configuration"""

import os
from pathlib import Path


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_prefixes(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values


# --- Data paths (configurable via env vars) ---
PRED_PKL_PATH = Path(os.environ.get("PRED_PKL_PATH", str(Path.home() / "quantpilot" / "models" / "pred.pkl")))
PRED_SH_PATH = Path(os.environ.get("PRED_SH_PATH", str(Path.home() / "quantpilot" / "models" / "pred_sh.pkl")))
PRED_HK_PATH = Path(os.environ.get("PRED_HK_PATH", str(Path.home() / "quantpilot" / "models" / "pred_hk.pkl")))
PRICE_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", str(Path.home() / "quantpilot" / "qlib_data")))
OUTPUT_DIR = Path(os.environ.get("BACKTEST_OUTPUT_DIR", str(Path(__file__).parent / "output")))

# --- Strategy parameters (default to live trade rules) ---
TOP_N = int(os.environ.get("TOP_N", "5"))
HOLD_BONUS = float(os.environ.get("HOLD_BONUS", "0.05"))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "-0.08"))
POSITION_RATIO = float(os.environ.get("POSITION_RATIO", "0.95"))
FILTER_LIMIT_UP = _env_flag("FILTER_LIMIT_UP", True)
TRADEABLE_PREFIXES = _env_prefixes(
    "BACKTEST_TRADEABLE_PREFIXES",
    os.environ.get("A_SHARE_TRADEABLE_PREFIXES", "SH."),
)
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
