# QuantPilot Trading Rules

## Priority
**Data Quality > Trading Rule Authenticity > Cost Conservatism > Risk Control**

## Training Data Rules
1. Training data must use forward-adjusted prices (qfq), consistent with live trading
2. Universe: all SH A-shares (including delisted/suspended for survivorship bias)
3. Label: `Ref($close, -2) / Ref($close, -1) - 1` (next-day return)
4. Features must not use future data — all Alpha158 features use past windows only
5. Train/valid/test split must be strictly chronological (no shuffle)

## Backtesting Rules
1. Signal generated at day-t close → buy at t+1 close → sell at t+2 close
2. Transaction costs must include: stamp duty (sell 0.05%), commission (0.025% each side), slippage (0.1%)
3. Buy quantity rounded to lot size (100 shares for SH)
4. Dual limit-up filter: skip stocks hitting limit-up on signal day or buy day
5. Equal-weight allocation across Top-N positions
6. Hold inertia: existing positions get score bonus, reducing unnecessary turnover

## Live Trading Rules
1. Trading environment must be SIMULATE by default (safety lock in code)
2. Sell first, then buy — ensure cash available
3. Signal must match backtest timing: read yesterday's signal, trade today
4. Position sizing: 95% of available cash, equal-weight
5. Stop-loss: -8% per position, force sell regardless of signal
6. Buy price: market + 1% slippage to ensure execution
7. Sell price: market - 1% slippage to ensure execution

## Risk Control
1. Maximum positions: Top-N (default 5)
2. Single stock max allocation: 1/N of total assets
3. Stop-loss triggered immediately, no re-entry on same day
4. Real trading only enabled by explicit code change (not config)
