"""CLI entry point for strategy training/prediction."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="QuantPilot strategy CLI")
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--market", default="all")
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()

    models_dir = os.environ.get("MODELS_DIR") or args.models_dir

    from strategy.engine import StrategyEngine
    engine = StrategyEngine(models_dir=models_dir)

    if args.action == "train":
        result = engine.train(market=args.market, save_dir=models_dir)
        print(f"IC: {result['IC']:.6f}  ICIR: {result['ICIR']:.6f}")
    elif args.action == "predict":
        df = engine.predict(market=args.market)
        print(df.to_string())


if __name__ == "__main__":
    main()
