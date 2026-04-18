"""Shared market/universe scope helpers."""

from __future__ import annotations

import os


def env_prefixes(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def a_share_model_prefixes() -> tuple[str, ...]:
    return env_prefixes("A_SHARE_MODEL_PREFIXES", "SH.,SZ.")


def a_share_tradeable_prefixes() -> tuple[str, ...]:
    return env_prefixes("A_SHARE_TRADEABLE_PREFIXES", "SH.")


def code_matches_prefixes(code: str, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return True
    return code.startswith(prefixes)
