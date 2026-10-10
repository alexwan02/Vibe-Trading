"""The local loader keeps ``amount`` instead of silently dropping it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import backtest.loaders.local_loader as local_loader
from tests.local_bridge_helpers import (
    CLEAN_ROWS,
    fetch_symbol,
    point_bridge_at,
    write_bars,
)


def _bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rows: list[dict]) -> None:
    path = write_bars(tmp_path / "bars.csv", rows)
    point_bridge_at(
        monkeypatch,
        tmp_path,
        {"sources": [{"symbol": "TEST.HK", "type": "csv", "path": str(path)}]},
    )


def test_amount_is_kept_as_float(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _bridge(monkeypatch, tmp_path, CLEAN_ROWS)
    frame = fetch_symbol("TEST.HK")
    assert frame is not None
    assert frame["amount"].tolist() == [row["amount"] for row in CLEAN_ROWS]
    assert frame["amount"].dtype == "float64"


def test_file_without_amount_has_no_amount_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [{k: v for k, v in row.items() if k != "amount"} for row in CLEAN_ROWS]
    _bridge(monkeypatch, tmp_path, rows)
    frame = fetch_symbol("TEST.HK")
    assert frame is not None
    assert "amount" not in frame.columns
    assert len(frame) == len(rows)


def test_downsampling_sums_amount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Turnover over a coarser bar is the sum of its parts, like volume."""
    hours = pd.date_range("2024-01-02 01:00", periods=4, freq="h")
    rows = [
        {
            "date": ts.isoformat(),
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "volume": 10,
            "amount": 100.0,
        }
        for ts in hours
    ]
    _bridge(monkeypatch, tmp_path, rows)
    frame = local_loader.DataLoader().fetch(
        ["TEST.HK"], "2024-01-02", "2024-01-02", interval="4H"
    )["TEST.HK"]
    assert frame["amount"].sum() == pytest.approx(400.0)
    assert frame["volume"].sum() == pytest.approx(40.0)


def test_reader_schema_is_part_of_the_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache written before ``amount`` was kept must not be served now."""
    entry = {"symbol": "TEST.HK", "type": "csv", "path": "bars.csv"}
    current = local_loader._source_cache_identity(entry)
    monkeypatch.setattr(local_loader, "_READER_SCHEMA_VERSION", 1)
    assert local_loader._source_cache_identity(entry) != current
