"""
Backtest engine: daily Top-N equal-weight strategy based on pred.pkl.

Default rules are aligned with live trade:
  - Signal day t: model produces scores
  - Trade day t+1 close: rebalance using signal-day scores
  - Next trade day t+2 close: mark daily holding return
  - Universe defaults to SH. only (same as live signal extraction)
  - Hold inertia, stop-loss and dual limit-up filter match live defaults
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
    if code.startswith("SZ.300") or code.startswith("SH.688"):
        return 19.5
    return 9.5


def run_backtest(
    pred: pd.Series,
    close_df: pd.DataFrame,
    top_n: int = 5,
    hold_bonus: float = 0.05,
    change_df: pd.DataFrame | None = None,
    st_df: pd.DataFrame | None = None,
    filter_limit_up: bool = True,
    stop_loss_pct: float = -0.08,
    position_ratio: float = 0.95,
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
        stop_loss_pct: sell held names when mark-to-market PnL falls below threshold
        position_ratio: invest this share of portfolio capital, keep the rest in cash
    """
    close_df = close_df.copy()
    close_df.index = pd.to_datetime(close_df.index)
    if change_df is not None:
        change_df = change_df.copy()
        change_df.index = pd.to_datetime(change_df.index)
    if st_df is not None:
        st_df = st_df.copy()
        st_df.index = pd.to_datetime(st_df.index)
    price_dates = sorted(close_df.index)
    date_to_idx = {d: i for i, d in enumerate(price_dates)}
    signal_dates = sorted(pd.to_datetime(pred.index.get_level_values("datetime").unique()))

    records = []
    entry_prices: dict[str, float] = {}

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
        current_portfolio = set(entry_prices.keys())

        # Hold inertia: held stocks get score bonus
        if hold_bonus > 0 and current_portfolio:
            for code in current_portfolio:
                if code in day_scores.index:
                    day_scores[code] += hold_bonus

        day_scores = day_scores.sort_values(ascending=False)

        # Filter: must have close prices on t+1 and t+2, no limit-up
        eligible = []
        for code in day_scores.index:
            if code not in close_df.columns:
                continue
            c1 = close_df.at[t1, code] if t1 in close_df.index else np.nan
            c2 = close_df.at[t2, code] if t2 in close_df.index else np.nan
            if not (pd.notna(c1) and pd.notna(c2) and c1 > 0):
                continue

            if st_df is not None and not st_df.empty and code in st_df.columns:
                st_t = st_df.at[t, code] if t in st_df.index else np.nan
                st_t1 = st_df.at[t1, code] if t1 in st_df.index else np.nan
                if (pd.notna(st_t) and st_t >= 0.5) or (pd.notna(st_t1) and st_t1 >= 0.5):
                    continue

            # Limit-up filter: signal day t or buy day t+1 hit limit, skip
            if filter_limit_up and change_df is not None and code.startswith(("SH.", "SZ.")):
                limit_pct = _get_limit_up_pct(code)
                chg_t = change_df.at[t, code] if (t in change_df.index and code in change_df.columns) else np.nan
                chg_t1 = change_df.at[t1, code] if (t1 in change_df.index and code in change_df.columns) else np.nan
                if (pd.notna(chg_t) and chg_t >= limit_pct) or (pd.notna(chg_t1) and chg_t1 >= limit_pct):
                    continue

            eligible.append(code)

        target_set = set(eligible[:top_n])
        stop_loss_sells = set()
        for code in current_portfolio:
            entry_price = entry_prices.get(code)
            if entry_price is None or entry_price <= 0 or code not in close_df.columns:
                continue
            current_price = close_df.at[t1, code] if t1 in close_df.index else np.nan
            if pd.isna(current_price) or current_price <= 0:
                continue
            pl_ratio = current_price / entry_price - 1
            if pl_ratio <= stop_loss_pct:
                stop_loss_sells.add(code)
                target_set.discard(code)

        sells = (current_portfolio - target_set) | stop_loss_sells
        holds = current_portfolio - sells
        available_slots = max(top_n - len(holds), 0)
        buys = [
            code for code in eligible
            if code not in holds and code not in stop_loss_sells
        ][:available_slots]
        new_portfolio = holds | set(buys)
        ordered_positions = [code for code in eligible if code in new_portfolio]
        n = len(ordered_positions)
        if n == 0:
            entry_prices = {}
            continue

        # Position returns: equal weight
        returns = []
        for code in ordered_positions:
            c1 = close_df.at[t1, code]
            c2 = close_df.at[t2, code]
            returns.append(c2 / c1 - 1)
        gross_return = position_ratio * np.mean(returns)

        # Turnover and fees
        turnover = (len(sells) + len(buys)) / (2 * max(top_n, 1))

        fee_cost = 0.0
        for code in sells:
            fee_cost += (position_ratio / max(len(current_portfolio), 1)) * _get_fee_rate(code, "sell")
        for code in buys:
            fee_cost += (position_ratio / n) * _get_fee_rate(code, "buy")

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
            "n_stop_loss_sells": len(stop_loss_sells),
            "positions": ",".join(ordered_positions),
        })

        next_entry_prices = {}
        for code in ordered_positions:
            if code in holds and code in entry_prices:
                next_entry_prices[code] = entry_prices[code]
            else:
                next_entry_prices[code] = float(close_df.at[t1, code])
        entry_prices = next_entry_prices

    df = pd.DataFrame(records)
    if not df.empty:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df
