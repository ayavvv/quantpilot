"""CLI entry point for strategy training/prediction."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="QuantPilot strategy CLI")
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--market", default="sh")
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()

    models_dir = os.environ.get("MODELS_DIR") or args.models_dir
    provider_uri = os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data/my_quant_data")

    from strategy.engine import StrategyEngine
    engine = StrategyEngine(provider_uri=provider_uri)

    if args.action == "train":
        if args.market in ("sh", "a", "all"):
            result = engine.train_a_model(save_dir=models_dir)
        elif args.market == "hk":
            result = engine.train_hk_model(save_dir=models_dir)
        else:
            print(f"Unknown market: {args.market}", file=sys.stderr)
            sys.exit(1)
        print(f"IC: {result['IC']:.6f}  ICIR: {result['ICIR']:.6f}")

    elif args.action == "predict":
        hk_mode = args.market == "hk"
        df = engine.predict_next_day(hk_mode=hk_mode)
        print(df.to_string())


if __name__ == "__main__":
    main()
