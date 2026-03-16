#!/usr/bin/env python3
"""
One-time migration: convert existing parquet data to Qlib bin format.

Usage:
    python scripts/migrate_parquet_to_qlib.py [--kline-dir /data/kline/K_DAY] [--qlib-dir /qlib_data]

Or via Docker:
    docker compose run --rm -e KLINE_DIR=/data/kline/K_DAY -e QLIB_DATA_DIR=/qlib_data collector \
        python scripts/migrate_parquet_to_qlib.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from converter.incremental import QlibDirectWriter
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Migrate parquet data to Qlib bin format")
    parser.add_argument(
        "--kline-dir",
        default=os.environ.get("KLINE_DIR", "/data/kline/K_DAY"),
        help="Directory containing {code}/data.parquet files",
    )
    parser.add_argument(
        "--qlib-dir",
        default=os.environ.get("QLIB_DATA_DIR", "/qlib_data"),
        help="Output Qlib binary data directory",
    )
    args = parser.parse_args()

    kline_dir = Path(args.kline_dir)
    qlib_dir = Path(args.qlib_dir)

    if not kline_dir.exists():
        logger.error(f"Kline directory not found: {kline_dir}")
        sys.exit(1)

    logger.info(f"Source:  {kline_dir}")
    logger.info(f"Target:  {qlib_dir}")

    writer = QlibDirectWriter(qlib_dir)
    result = writer.migrate_from_parquet(kline_dir)

    logger.info(f"Result: {result}")

    if result.get("status") == "ok":
        logger.info(
            f"Migration complete: {result['migrated']}/{result['total_stocks']} stocks, "
            f"{result['calendar_days']} calendar days"
        )
    else:
        logger.warning(f"Migration returned: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
