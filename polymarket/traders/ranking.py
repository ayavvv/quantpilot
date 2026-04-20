"""Robustness ranking for Polymarket top-trader mirror strategy."""
from __future__ import annotations

from polymarket.config import PolySettings, settings
from polymarket.models import TraderScore


def compute_trader_scores(candidates: list[dict], history_stats: dict[str, dict], cfg: PolySettings | None = None) -> list[TraderScore]:
    cfg = cfg or settings
    scores: list[TraderScore] = []
    for row in candidates:
        wallet = str(row.get("proxyWallet") or row.get("wallet") or "").strip()
        if not wallet:
            continue
        stats = history_stats.get(wallet, {})
        trade_count = int(stats.get("trade_count", 0))
        diversity_count = int(stats.get("diversity_count", 0))
        realized_pnl = float(stats.get("realized_pnl", 0.0))
        volume = float(stats.get("volume", row.get("vol") or 0.0))
        pnl = float(row.get("pnl") or 0.0)
        if trade_count < cfg.top_trader_min_trades:
            continue
        if diversity_count < cfg.top_trader_min_diversity:
            continue
        score = realized_pnl * 0.7 + pnl * 0.2 + min(diversity_count, 20) * 100 - max(volume, 1.0) * 0.0001
        scores.append(
            TraderScore(
                wallet=wallet,
                score=score,
                pnl=pnl,
                volume=volume,
                trade_count=trade_count,
                diversity_count=diversity_count,
                realized_pnl=realized_pnl,
            )
        )
    scores.sort(key=lambda item: item.score, reverse=True)
    for index, score in enumerate(scores, start=1):
        score.rank = index
    return scores[: cfg.top_trader_tracked_count]
