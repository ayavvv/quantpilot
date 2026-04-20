"""Mirror signal generation for Polymarket top-trader strategy."""
from __future__ import annotations

from polymarket.config import PolySettings, settings
from polymarket.models import MirrorSignal, TraderEvent


FOLLOWABLE_EVENT_TYPES = {"TRADE"}
FOLLOWABLE_SIDES = {"BUY", "SELL"}


def generate_mirror_signals(events: list[TraderEvent], cfg: PolySettings | None = None) -> list[MirrorSignal]:
    cfg = cfg or settings
    signals: list[MirrorSignal] = []
    for event in events:
        if event.event_type not in FOLLOWABLE_EVENT_TYPES:
            continue
        if event.side not in FOLLOWABLE_SIDES:
            continue
        if event.size < cfg.top_trader_min_signal_size:
            continue
        signal_size = min(event.size, cfg.top_trader_max_signal_notional)
        fingerprint = event.fingerprint()
        signals.append(
            MirrorSignal(
                wallet=event.wallet,
                market_id=event.market_id,
                asset=event.asset,
                title=event.title or "",
                outcome=event.outcome,
                side=event.side,
                source_size=event.size,
                source_price=event.price,
                signal_size=signal_size,
                lag_seconds=cfg.top_trader_mirror_lag_seconds,
                timestamp=event.timestamp,
                fingerprint=fingerprint,
            )
        )
    return signals
