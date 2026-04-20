"""Public trader profile enrichment for Polymarket mirror strategy."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from polymarket.config import PolySettings, settings
from polymarket.models import TraderProfile


class PublicProfileClient:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{self.cfg.gamma_base_url.rstrip('/')}/{path.lstrip('/')}" + query
        request = Request(url, headers={"User-Agent": self.cfg.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=self.cfg.http_timeout_seconds) as response:
            return json.load(response)

    def fetch_public_profile(self, wallet: str) -> TraderProfile:
        data = self._get_json("public-profile", params={"address": wallet})
        return TraderProfile(
            wallet=str(data.get("proxyWallet") or wallet),
            user_name=(data.get("name") or "") or None,
            pseudonym=(data.get("pseudonym") or "") or None,
            verified_badge=bool(data.get("verifiedBadge")),
            profile_image=None,
        )
