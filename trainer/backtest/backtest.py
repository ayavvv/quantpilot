"""
Backtest engine: daily Top-N equal-weight strategy based on pred.pkl.

Trading logic (matching label Ref($close,-2)/Ref($close,-1)-1):
  - Signal day t: model produces scores
  - t+1 close: buy (entry)
  - t+2 close: sell (exit)
  - Position return = close(t+2)/close(t+1) - 1
  - Continuous portfolio: daily rebalancing, fees only on turnover

Hold inertia (hold_bonus):
  - Existing positions get a score bonus during ranking
  - New stocks must exceed held stock score + hold_bonus to replace
  - Significantly reduces turnover and transaction cost drag
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEE_CONFIG, SLIPPAGE


def _get_fee_rate(code: str, side: str) -> float:
    """Get single-side fee rate + slippage for a stock."""
    if code.startswith("HK."):
        market = "HK"
    elif code.startswith("SH."):
        market = "SH"
    elif code.startswith("US."):
        market = "US"
    else:
        market = "SH"
    fee = FEE_CONFIG.get(market, FEE_CONFIG["SH"])
    return fee[side] + SLIPPAGE


def _get_limit_up_pct(code: str) -> float:
    """A-share limit-up threshold: ChiNext/STAR 20%, main board 10%."""
    if code.startswith("SH.300") or code.startswith("SH.688"):
        return 19.5
    return 9.5


def run_backtest(
    pred: pd.Series,
    close_df: pd.DataFrame,
    top_n: int = 20,
    hold_bonus: float = 0.0,
    change_df: pd.DataFrame | None = None,
    filter_limit_up: bool = False,
) -> pd.DataFrame:
    """
    Run backtest.

    Args:
        pred: MultiIndex (datetime, instrument) score Series
        close_df: date x code close price matrix
        top_n: number of positions
        hold_bonus: hold inertia bonus (held stocks score += hold_bonus)
        change_df: date x code change rate matrix (%), for limit-up filtering
        filter_limit_up: whether to filter limit-up stocks (A-shares)
    """
    price_dates = sorted(close_df.index)
    date_to_idx = {d: i for i, d in enumerate(price_dates)}
    signal_dates = sorted(pred.index.get_level_values("datetime").unique())

    records = []
    prev_portfolio = set()

    for t in signal_dates:
        if t not in date_to_idx:
            continue
        idx = date_to_idx[t]
        if idx + 2 >= len(price_dates):
            continue
        t1 = price_dates[idx + 1]
        t2 = price_dates[idx + 2]

        # Get daily scores
        day_scores = pred.xs(t, level="datetime")
        if isinstance(day_scores, pd.DataFrame):
            day_scores = day_scores.iloc[:, 0]
        day_scores = day_scores.dropna().copy()

        # Hold inertia: held stocks get score bonus
        if hold_bonus > 0 and prev_portfolio:
            for code in prev_portfolio:
                if code in day_scores.index:
                    day_scores[code] += hold_bonus

        day_scores = day_scores.sort_values(ascending=False)

        # Filter: must have close prices on t+1 and t+2, no limit-up
        candidates = []
        for code in day_scores.index:
            if code not in close_df.columns:
                continue
            c1 = close_df.at[t1, code] if t1 in close_df.index else np.nan
            c2 = close_df.at[t2, code] if t2 in close_df.index else np.nan
            if not (pd.notna(c1) and pd.notna(c2) and c1 > 0):
                continue

            # Limit-up filter: signal day t or buy day t+1 hit limit, skip
            if filter_limit_up and change_df is not None and code.startswith("SH."):
                limit_pct = _get_limit_up_pct(code)
                chg_t = change_df.at[t, code] if (t in change_df.index and code in change_df.columns) else np.nan
                chg_t1 = change_df.at[t1, code] if (t1 in change_df.index and code in change_df.columns) else np.nan
                if (pd.notna(chg_t) and chg_t >= limit_pct) or (pd.notna(chg_t1) and chg_t1 >= limit_pct):
                    continue

            candidates.append(code)
            if len(candidates) >= top_n:
                break

        new_portfolio = set(candidates)
        n = len(new_portfolio)
        if n == 0:
            continue

        # Position returns: equal weight
        returns = []
        for code in candidates:
            c1 = close_df.at[t1, code]
            c2 = close_df.at[t2, code]
            returns.append(c2 / c1 - 1)
        gross_return = np.mean(returns)

        # Turnover and fees
        sells = prev_portfolio - new_portfolio
        buys = new_portfolio - prev_portfolio
        holds = prev_portfolio & new_portfolio
        turnover = (len(sells) + len(buys)) / (2 * max(top_n, 1))

        fee_cost = 0.0
        for code in sells:
            fee_cost += (1.0 / max(len(prev_portfolio), 1)) * _get_fee_rate(code, "sell")
        for code in buys:
            fee_cost += (1.0 / n) * _get_fee_rate(code, "buy")

        net_return = gross_return - fee_cost

        records.append({
            "signal_date": t,
            "entry_date": t1,
            "exit_date": t2,
            "gross_return": gross_return,
            "fee_cost": fee_cost,
            "net_return": net_return,
            "turnover": turnover,
            "n_positions": n,
            "n_buys": len(buys),
            "n_sells": len(sells),
            "n_holds": len(holds),
            "positions": ",".join(sorted(candidates)),
        })

        prev_portfolio = new_portfolio

    df = pd.DataFrame(records)
    if not df.empty:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df
