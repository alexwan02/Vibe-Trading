"""Synthetic local data-bridge files shared by the local-loader tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import backtest.loaders.local_loader as local_loader

# The middle bar has ``open`` a tick above ``high``: the shape an
# opening-auction print takes when the vendor's range covers only the
# continuous session.
AUCTION_ROWS: list[dict[str, Any]] = [
    {
        "date": "2024-01-02",
        "open": 10.0,
        "high": 10.5,
        "low": 9.8,
        "close": 10.2,
        "volume": 100,
        "amount": 1020.0,
    },
    {
        "date": "2024-01-03",
        "open": 10.6,
        "high": 10.5,
        "low": 9.9,
        "close": 10.1,
        "volume": 200,
        "amount": 2020.0,
    },
    {
        "date": "2024-01-04",
        "open": 10.1,
        "high": 10.4,
        "low": 9.7,
        "close": 10.3,
        "volume": 300,
        "amount": 3090.0,
    },
]
CLEAN_ROWS = [row for row in AUCTION_ROWS if row["date"] != "2024-01-03"]


def write_bars(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write ``rows`` as CSV or Parquet depending on ``path``'s suffix."""
    frame = pd.DataFrame(rows)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def point_bridge_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict[str, Any]
) -> Path:
    """Write ``config`` as the bridge YAML and point the loader at it."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(local_loader, "_CONFIG_PATH", config_path)
    return config_path


def fetch_symbol(
    symbol: str, start: str = "2024-01-01", end: str = "2024-01-31"
) -> pd.DataFrame | None:
    """Fetch one symbol through a fresh loader."""
    return local_loader.DataLoader().fetch([symbol], start, end).get(symbol)
