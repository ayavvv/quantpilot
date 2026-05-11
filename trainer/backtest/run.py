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

from .config import (
    FILTER_LIMIT_UP,
    HOLD_BONUS,
    OUTPUT_DIR,
    POSITION_RATIO,
    PRED_PKL_PATH,
    PRICE_DATA_DIR,
    SLIPPAGE,
    STOP_LOSS_PCT,
    TOP_N,
    TRADEABLE_PREFIXES,
)
from .data_loader import load_change_rates, load_close_prices, load_predictions, load_st_flags
from .backtest import run_backtest
from .report import compute_metrics, generate_charts


def main():
    parser = argparse.ArgumentParser(description="Quant strategy backtest")
    parser.add_argument("--pred", type=str, default=str(PRED_PKL_PATH))
    parser.add_argument("--price-dir", type=str, default=str(PRICE_DATA_DIR))
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--hold-bonus", type=float, default=HOLD_BONUS)
    parser.add_argument("--stop-loss-pct", type=float, default=STOP_LOSS_PCT)
    parser.add_argument("--position-ratio", type=float, default=POSITION_RATIO)
    parser.add_argument("--slippage", type=float, default=SLIPPAGE)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--allowed-prefix", action="append", dest="allowed_prefixes")
    parser.add_argument("--filter-limit-up", dest="filter_limit_up", action="store_true")
    parser.add_argument("--no-filter-limit-up", dest="filter_limit_up", action="store_false")
    parser.set_defaults(filter_limit_up=FILTER_LIMIT_UP)
    args = parser.parse_args()

    pred_path = Path(args.pred).expanduser()
    price_dir = Path(args.price_dir).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_prefixes = tuple(args.allowed_prefixes or TRADEABLE_PREFIXES)

    print("=" * 60)
    print("Quant Strategy Backtest")
    print("=" * 60)
    print(f"  Prediction file: {pred_path}")
    print(f"  Price directory: {price_dir}")
    print(
        f"  Top-N: {args.top_n}  Hold bonus: {args.hold_bonus:.4f}  "
        f"Stop-loss: {args.stop_loss_pct:.2%}  Position ratio: {args.position_ratio:.0%}"
    )
    print(
        f"  Prefixes: {allowed_prefixes or 'ALL'}  "
        f"Limit-up filter: {'on' if args.filter_limit_up else 'off'}  "
        f"Slippage: {args.slippage:.2%}/side"
    )
    print()

    # 1. Load data
    pred = load_predictions(pred_path, allowed_prefixes=allowed_prefixes)
    instruments = sorted(pred.index.get_level_values("instrument").unique())
    pred_dates = pred.index.get_level_values("datetime")
    start = pred_dates.min().strftime("%Y-%m-%d")
    end = pred_dates.max().strftime("%Y-%m-%d")

    # Extra days for t+2
    close_df = load_close_prices(price_dir, instruments, start_date=start, end_date="2099-12-31")
    change_df = None
    if args.filter_limit_up:
        change_df = load_change_rates(price_dir, instruments, start_date=start, end_date="2099-12-31")
    st_df = load_st_flags(price_dir, instruments, start_date=start, end_date="2099-12-31")
    st_filter_source = "point-in-time"
    if st_df.empty:
        from strategy.stock_filter import load_a_share_st_codes

        current_st = load_a_share_st_codes(price_dir)
        matched_st = sorted(set(instruments) & current_st)
        if matched_st:
            st_df = pd.DataFrame(0.0, index=close_df.index, columns=instruments)
            st_df.loc[:, matched_st] = 1.0
            st_filter_source = "current snapshot fallback"
        else:
            st_filter_source = "unavailable"
            st_df = None
    else:
        covered_ratio = len(st_df.columns) / max(len(instruments), 1)
        if covered_ratio < 0.9:
            from strategy.stock_filter import load_a_share_st_codes

            current_st = load_a_share_st_codes(price_dir)
            matched_st = sorted(set(instruments) & current_st)
            if matched_st:
                st_df = pd.DataFrame(0.0, index=close_df.index, columns=instruments)
                st_df.loc[:, matched_st] = 1.0
                st_filter_source = "current snapshot fallback"
            else:
                st_filter_source = "unavailable"
                st_df = None

    # 2. Run backtest
    print("\nRunning backtest...")
    results = run_backtest(
        pred,
        close_df,
        top_n=args.top_n,
        hold_bonus=args.hold_bonus,
        change_df=change_df,
        st_df=st_df,
        filter_limit_up=args.filter_limit_up,
        stop_loss_pct=args.stop_loss_pct,
        position_ratio=args.position_ratio,
    )

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
        f.write(
            "Backtest params: "
            f"Top-{args.top_n}, Hold bonus {args.hold_bonus:.4f}, "
            f"Stop-loss {args.stop_loss_pct:.2%}, Position ratio {args.position_ratio:.0%}, "
            f"Prefixes {allowed_prefixes or 'ALL'}, "
            f"ST filter {st_filter_source}, "
            f"Limit-up filter {'on' if args.filter_limit_up else 'off'}, "
            f"Slippage {args.slippage:.2%}/side\n"
        )
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Metrics summary: {metrics_path}")


if __name__ == "__main__":
    main()
