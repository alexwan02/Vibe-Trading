"""``invalid_bars`` keeps, drops or rejects bars that fail the OHLC invariants."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import backtest.loaders.local_loader as local_loader
from backtest.loaders import base
from tests.local_bridge_helpers import (
    AUCTION_ROWS,
    CLEAN_ROWS,
    fetch_symbol,
    point_bridge_at,
    write_bars,
)


def _bridge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    top: str | None = None,
    source: str | None = None,
    rows: list[dict] | None = None,
) -> None:
    entry = {
        "symbol": "TEST.HK",
        "type": "csv",
        "path": str(write_bars(tmp_path / "bars.csv", rows or AUCTION_ROWS)),
    }
    if source is not None:
        entry["invalid_bars"] = source
    config: dict = {"sources": [entry]}
    if top is not None:
        config["invalid_bars"] = top
    point_bridge_at(monkeypatch, tmp_path, config)


def test_default_drops_the_bar_so_existing_configs_are_unaffected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _bridge(monkeypatch, tmp_path)
    frame = fetch_symbol("TEST.HK")
    assert frame is not None
    assert len(frame) == len(AUCTION_ROWS) - 1


def test_warn_keeps_the_bar_and_logs_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _bridge(monkeypatch, tmp_path, top="warn")
    with caplog.at_level(logging.WARNING, logger=base.logger.name):
        frame = fetch_symbol("TEST.HK")
    assert frame is not None
    assert len(frame) == len(AUCTION_ROWS)
    assert frame["open"].iloc[1] == pytest.approx(10.6)
    assert "1 bar(s) violate invariants (kept)" in caplog.text


def test_raise_propagates_instead_of_dropping_the_symbol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A symbol that silently goes missing is the quiet outcome ``raise`` rules out."""
    _bridge(monkeypatch, tmp_path, top="raise")
    with pytest.raises(local_loader.InvalidBarsError, match="TEST.HK") as info:
        fetch_symbol("TEST.HK")
    assert "violate OHLC invariants" in str(info.value)
    assert isinstance(info.value, ValueError)


def test_per_source_setting_overrides_the_top_level_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _bridge(monkeypatch, tmp_path, top="raise", source="warn")
    frame = fetch_symbol("TEST.HK")
    assert frame is not None
    assert len(frame) == len(AUCTION_ROWS)


def test_unknown_value_falls_back_to_drop_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _bridge(monkeypatch, tmp_path, source="warm")
    with caplog.at_level(logging.WARNING, logger=local_loader.logger.name):
        frame = fetch_symbol("TEST.HK")
    assert frame is not None
    assert len(frame) == len(AUCTION_ROWS) - 1
    assert "unknown source-level invalid_bars 'warm'" in caplog.text


@pytest.mark.parametrize("policy", ["drop", "warn", "raise"])
def test_clean_file_returns_every_row_under_each_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy: str
) -> None:
    _bridge(monkeypatch, tmp_path, top=policy, rows=CLEAN_ROWS)
    frame = fetch_symbol("TEST.HK")
    assert frame is not None
    assert len(frame) == len(CLEAN_ROWS)


def test_cache_does_not_serve_dropped_rows_after_switching_to_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``drop`` and ``warn`` read the same file into different frames."""
    monkeypatch.setattr(base, "loader_cache_enabled", lambda: True)
    monkeypatch.setattr(base, "loader_cache_root", lambda: tmp_path / "cache")

    _bridge(monkeypatch, tmp_path, top="drop")
    dropped = fetch_symbol("TEST.HK")
    assert list((tmp_path / "cache").rglob("*.parquet")), "exercise the real cache"

    _bridge(monkeypatch, tmp_path, top="warn")
    kept = fetch_symbol("TEST.HK")

    assert dropped is not None and kept is not None
    assert len(kept) == len(dropped) + 1
