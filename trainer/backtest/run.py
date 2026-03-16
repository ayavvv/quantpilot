"""
Backtest entry point

Usage:
    python -m trainer.backtest.run
    python -m trainer.backtest.run --top-n 10 --slippage 0.002
    python -m trainer.backtest.run --pred /path/to/pred.pkl --price-dir /path/to/K_DAY/
"""

import argparse
from pathlib import Path

import pandas as pd

from .config import PRED_PKL_PATH, PRICE_DATA_DIR, OUTPUT_DIR, TOP_N, SLIPPAGE
from .data_loader import load_predictions, load_close_prices
from .backtest import run_backtest
from .report import compute_metrics, generate_charts


def main():
    parser = argparse.ArgumentParser(description="Quant strategy backtest")
    parser.add_argument("--pred", type=str, default=str(PRED_PKL_PATH))
    parser.add_argument("--price-dir", type=str, default=str(PRICE_DATA_DIR))
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--slippage", type=float, default=SLIPPAGE)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    pred_path = Path(args.pred).expanduser()
    price_dir = Path(args.price_dir).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Quant Strategy Backtest")
    print("=" * 60)
    print(f"  Prediction file: {pred_path}")
    print(f"  Price directory: {price_dir}")
    print(f"  Top-N: {args.top_n}  Slippage: {args.slippage:.2%}/side")
    print()

    # 1. Load data
    pred = load_predictions(pred_path)
    instruments = sorted(pred.index.get_level_values("instrument").unique())
    pred_dates = pred.index.get_level_values("datetime")
    start = pred_dates.min().strftime("%Y-%m-%d")
    end = pred_dates.max().strftime("%Y-%m-%d")

    # Extra days for t+2
    close_df = load_close_prices(price_dir, instruments,
                                 start_date=start, end_date="2099-12-31")

    # 2. Run backtest
    print("\nRunning backtest...")
    results = run_backtest(pred, close_df, top_n=args.top_n)

    if results.empty:
        print("Backtest results empty, please check data.")
        return

    # 3. Compute metrics
    metrics = compute_metrics(results)
    print("\n" + "=" * 60)
    print("Backtest Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 4. Generate charts
    print()
    chart_path = generate_charts(results, metrics, output_dir,
                                 top_n=args.top_n, slippage=args.slippage)

    # 5. Save daily detail
    csv_path = output_dir / "daily_results.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Daily detail: {csv_path}")

    # 6. Save metrics
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Backtest params: Top-{args.top_n}, Slippage {args.slippage:.2%}/side\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Metrics summary: {metrics_path}")


if __name__ == "__main__":
    main()
