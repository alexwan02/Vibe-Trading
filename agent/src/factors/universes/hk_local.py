"""``hk-local`` bench universe: a Hong Kong panel from the local data bridge.

Membership is whatever ``.HK`` symbols the user configured in
``~/.vibe-trading/data-bridge/config.yaml``; other symbols in the same config
are ignored. See :mod:`backtest.loaders.local_loader` for the config format.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from backtest.loaders.local_loader import DataLoader
from src.factors.base import Market, vwap

logger = logging.getLogger(__name__)

UNIVERSE = "hk-local"
_HK_SUFFIX = ".HK"
# Config keys that point at files whose contents end up in the panel.
_FILE_KEYS = ("path", "db_path")
_OHLC = ("open", "high", "low", "close")


def open_bridge() -> DataLoader:
    """Return a local loader, or raise when the bridge is not configured.

    Raises:
        ValueError: The bridge config is missing or has no sources.
    """
    loader = DataLoader()
    if not loader.is_available():
        raise ValueError(
            f"universe {UNIVERSE!r} needs the local data bridge: configure "
            "~/.vibe-trading/data-bridge/config.yaml with .HK sources"
        )
    return loader


def hk_symbols(loader: DataLoader) -> list[str]:
    """Return the configured ``.HK`` symbols, sorted.

    Raises:
        ValueError: The config declares no ``.HK`` symbol.
    """
    configured = loader.configured_symbols()
    symbols = [s for s in configured if s.upper().endswith(_HK_SUFFIX)]
    if not symbols:
        raise ValueError(
            f"universe {UNIVERSE!r} found no .HK symbol among the "
            f"{len(configured)} source(s) in the local data bridge config"
        )
    if len(symbols) < len(configured):
        logger.info(
            "%s: ignoring %d non-.HK source(s) in the bridge config",
            UNIVERSE,
            len(configured) - len(symbols),
        )
    return symbols


def bridge_ready() -> bool:
    """Return True when the bridge config declares at least one ``.HK`` source."""
    try:
        hk_symbols(open_bridge())
    except ValueError:
        return False
    return True


def cache_fingerprint(loader: DataLoader) -> str:
    """Return a short key that changes with the config or any file it uses.

    The panel is a function of the bridge config and the files it points at,
    not of the period alone. Keying the panel cache on both means an edited
    config or a refreshed data file is read again instead of served stale.

    Args:
        loader: Loader over the bridge config.

    Returns:
        16 hex characters.
    """
    files: list[list[Any]] = []
    for symbol in hk_symbols(loader):
        entry = loader.source_entry(symbol) or {}
        for key in _FILE_KEYS:
            raw = entry.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            path = Path(raw.strip()).expanduser()
            try:
                stat = path.stat()
                stamp: list[int] | None = [stat.st_mtime_ns, stat.st_size]
            except OSError:
                stamp = None
            files.append([symbol, key, str(path), stamp])
    payload = json.dumps({"config": loader.config_digest(), "files": files})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_panel(loader: DataLoader, start: str, end: str) -> dict[str, Any]:
    """Build the wide ``hk-local`` panel for ``start``..``end``.

    Args:
        loader: Loader over the bridge config.
        start: Inclusive start date, ``YYYY-MM-DD``.
        end: Inclusive end date, ``YYYY-MM-DD``.

    Returns:
        Panel keyed by field (OHLCV, ``amount`` when the files carry it,
        ``vwap``) plus a ``_meta`` dict.

    Raises:
        ValueError: No ``.HK`` source is configured, or a source set to
            ``invalid_bars: raise`` holds an invalid bar.
    """
    from src.tools.alpha_bench_tool import _wide_from_fetched

    codes = hk_symbols(loader)
    # Not retried: these are local reads, and a retry would turn an
    # ``invalid_bars: raise`` failure into a silently missing symbol.
    fetched = loader.fetch(codes, start, end)
    panel: dict[str, Any] = _wide_from_fetched(fetched, include_amount=True)
    if all(key in panel for key in _OHLC):
        # Typical price, not ``amount / volume``: back-adjusted prices and raw
        # turnover are not in the same space, so their ratio is not a price
        # comparable to ``close``.
        panel["vwap"] = vwap(panel, Market.EQUITY_HK)

    panel["_meta"] = {
        "universe": UNIVERSE,
        "requested_instruments": len(codes),
        "fetched_instruments": len(fetched),
        "missing_instruments": [code for code in codes if code not in fetched],
        # The user chose the names, so nothing here can tell whether names
        # that delisted during the period are included. No claim either way,
        # and the report shows no survivorship banner.
        "survivorship_bias": None,
        "degraded": False,
        "constituent_source": "local data bridge (user-defined)",
        "constituent_source_date": None,
        "constituent_count": len(codes),
        "sector_source": None,
        "sector_coverage": 0.0,
    }
    return panel


__all__ = [
    "UNIVERSE",
    "bridge_ready",
    "cache_fingerprint",
    "hk_symbols",
    "load_panel",
    "open_bridge",
]
