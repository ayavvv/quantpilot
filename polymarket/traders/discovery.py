"""Public trader discovery for the Polymarket top-trader mirror strategy."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from polymarket.config import PolySettings, settings
from polymarket.models import TraderProfile


DATA_API_BASE = "https://data-api.polymarket.com"


class LeaderboardClient:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{DATA_API_BASE.rstrip('/')}/{path.lstrip('/')}" + query
        request = Request(url, headers={"User-Agent": self.cfg.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=self.cfg.http_timeout_seconds) as response:
            return json.load(response)

    def fetch_leaderboard(self, limit: int | None = None) -> list[dict[str, Any]]:
        params = {
            "timePeriod": "ALL",
            "orderBy": "PNL",
            "limit": limit or self.cfg.top_trader_candidate_limit,
            "offset": 0,
        }
        data = self._get_json("v1/leaderboard", params=params)
        return list(data)


def normalize_leaderboard_profiles(rows: list[dict[str, Any]]) -> list[TraderProfile]:
    profiles: list[TraderProfile] = []
    for row in rows:
        wallet = str(row.get("proxyWallet") or row.get("wallet") or "").strip()
        if not wallet:
            continue
        profiles.append(
            TraderProfile(
                wallet=wallet,
                user_name=(row.get("userName") or "") or None,
                pseudonym=(row.get("xUsername") or "") or None,
                verified_badge=bool(row.get("verifiedBadge")),
                profile_image=(row.get("profileImage") or "") or None,
            )
        )
    return profiles
