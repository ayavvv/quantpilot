"""
Daily inference pipeline: validate data -> model prediction -> signal output.

Pipeline:
1. Validate Qlib bin data (written directly by collector)
2. Load LightGBM model, predict scores for all stocks
3. Output signal files (CSV + pkl)
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("daily_inference")

# Configurable paths via env vars
QLIB_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", "/qlib_data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", "/data/signals"))


def step1_validate():
    """Validate Qlib bin data exists and is up to date.

    Collector writes directly to Qlib bin format, so no conversion needed.
    This step just validates data freshness and logs stats.
    """
    log.info("Step 1: Validating Qlib bin data ...")

    cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    if not cal_path.exists():
        raise RuntimeError(
            f"Qlib data not found at {QLIB_DATA_DIR}. "
            "Run collector first or set QLIB_DATA_DIR correctly."
        )

    lines = cal_path.read_text().strip().splitlines()
    if not lines:
        raise RuntimeError("Calendar file is empty")

    log.info(f"  Calendar range: {lines[0]} ~ {lines[-1]}, {len(lines)} days")

    for market in ["sh", "sz"]:
        inst_path = QLIB_DATA_DIR / "instruments" / f"{market}.txt"
        if inst_path.exists():
            n = len(inst_path.read_text().strip().splitlines())
            log.info(f"  {market.upper()} stocks: {n}")

    return lines[-1]


def step2_predict(last_date: str):
    """Load model and predict latest day."""
    log.info(f"Step 2: Model inference (latest date: {last_date}) ...")

    market = os.environ.get("MARKET", "sh")
    model_name = os.environ.get("MODEL_NAME", f"lightgbm_{market}_latest.pkl")
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from strategy.engine import InferenceEngine
    engine = InferenceEngine(provider_uri=str(QLIB_DATA_DIR))
    df = engine.predict(model_path=model_path, market=market)

    log.info(f"  Prediction complete: {len(df)} stocks")
    if len(df) > 0:
        top5 = df[df["top5"]]["code"].tolist()
        log.info(f"  Top-5: {top5}")

    return df


def step3_output(df, last_date: str):
    """Output signal files."""
    log.info("Step 3: Writing signal files ...")

    import pandas as pd

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    # CSV (human-readable)
    csv_path = SIGNAL_DIR / f"signal_{today}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  CSV: {csv_path}")

    # pkl (program-readable, MultiIndex Series)
    ts = pd.Timestamp(last_date)
    idx = pd.MultiIndex.from_arrays(
        [[ts] * len(df), df["code"].tolist()],
        names=["datetime", "instrument"],
    )
    pred_series = pd.Series(df["score"].values, index=idx, name="score")

    pkl_path = SIGNAL_DIR / f"pred_daily_{today}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pred_series, f)
    log.info(f"  PKL: {pkl_path}")

    # Update latest symlinks
    for link_name, target in [("pred_latest.pkl", pkl_path), ("signal_latest.csv", csv_path)]:
        link = SIGNAL_DIR / link_name
        link.unlink(missing_ok=True)
        link.symlink_to(target.name)
    log.info("  Latest symlinks updated")


def main():
    log.info("=" * 50)
    log.info("QuantPilot Daily Inference")
    log.info("=" * 50)
    start = datetime.now()

    try:
        last_date = step1_validate()
        df = step2_predict(last_date)

        if df.empty:
            log.warning("Prediction result is empty, skipping output")
        else:
            step3_output(df, last_date)

        elapsed = (datetime.now() - start).total_seconds()
        log.info(f"Done! Elapsed {elapsed:.0f}s")

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        log.error(f"Inference failed: {e}", exc_info=True)
        log.error(f"Elapsed {elapsed:.0f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
