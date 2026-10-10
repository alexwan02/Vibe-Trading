"""``hk-local`` builds a Hong Kong bench panel from the local data bridge."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import backtest.loaders.local_loader as local_loader
from src.api import alpha_routes
from src.factors import cli_handlers
from src.factors.universes import hk_local
from src.tools import alpha_bench_tool as abt
from tests.local_bridge_helpers import (
    AUCTION_ROWS,
    HK_SYMBOLS,
    point_bridge_at,
    write_bars,
    write_hk_bridge,
)

START, END = "2024-01-01", "2024-01-31"


def _panel() -> dict:
    return hk_local.load_panel(hk_local.open_bridge(), START, END)


def test_only_hk_symbols_enter_the_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_hk_bridge(monkeypatch, tmp_path)
    panel = _panel()
    assert list(panel["close"].columns) == list(HK_SYMBOLS)
    assert panel["_meta"]["constituent_count"] == len(HK_SYMBOLS)


def test_config_without_hk_symbols_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = write_bars(tmp_path / "us.csv", AUCTION_ROWS)
    point_bridge_at(
        monkeypatch,
        tmp_path,
        {"sources": [{"symbol": "AAPL.US", "type": "csv", "path": str(path)}]},
    )
    with pytest.raises(ValueError, match=r"no \.HK symbol"):
        _panel()
    assert hk_local.bridge_ready() is False


def test_missing_bridge_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(local_loader, "_CONFIG_PATH", tmp_path / "absent.yaml")
    with pytest.raises(ValueError, match="local data bridge"):
        hk_local.open_bridge()
    assert hk_local.bridge_ready() is False


def test_amount_is_carried_into_the_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_hk_bridge(monkeypatch, tmp_path)
    panel = _panel()
    assert panel["amount"].notna().all().all()
    assert panel["amount"].shape == panel["close"].shape


def test_files_without_amount_give_a_panel_without_amount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_hk_bridge(monkeypatch, tmp_path, with_amount=False)
    panel = _panel()
    assert "amount" not in panel
    assert "close" in panel


def test_vwap_is_the_typical_price(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_hk_bridge(monkeypatch, tmp_path)
    panel = _panel()
    expected = (panel["open"] + panel["high"] + panel["low"] + panel["close"]) / 4.0
    pd.testing.assert_frame_equal(panel["vwap"], expected)


def test_meta_reports_what_was_fetched_and_makes_no_survivorship_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_hk_bridge(
        monkeypatch,
        tmp_path,
        extra={HK_SYMBOLS[2]: {"path": str(tmp_path / "gone.parquet")}},
    )
    meta = _panel()["_meta"]
    assert meta["survivorship_bias"] is None
    assert meta["requested_instruments"] == len(HK_SYMBOLS)
    assert meta["fetched_instruments"] == len(HK_SYMBOLS) - 1
    assert meta["missing_instruments"] == [HK_SYMBOLS[2]]


def test_invalid_bars_raise_reaches_the_bench_caller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = write_bars(tmp_path / "bad.csv", AUCTION_ROWS)
    write_hk_bridge(
        monkeypatch,
        tmp_path,
        extra={
            HK_SYMBOLS[0]: {"type": "csv", "path": str(bad), "invalid_bars": "raise"}
        },
    )
    with pytest.raises(local_loader.InvalidBarsError, match=HK_SYMBOLS[0]):
        _panel()


def test_cache_key_follows_the_config_and_its_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_hk_bridge(monkeypatch, tmp_path)
    first = hk_local.cache_fingerprint(hk_local.open_bridge())
    assert hk_local.cache_fingerprint(hk_local.open_bridge()) == first

    price_file = tmp_path / f"{HK_SYMBOLS[0]}.parquet"
    stat = price_file.stat()
    os.utime(price_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    touched = hk_local.cache_fingerprint(hk_local.open_bridge())
    assert touched != first

    write_hk_bridge(
        monkeypatch, tmp_path, extra={HK_SYMBOLS[1]: {"invalid_bars": "warn"}}
    )
    assert hk_local.cache_fingerprint(hk_local.open_bridge()) not in {first, touched}


def test_panel_cache_is_not_served_after_the_config_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(abt, "get_env_config", _fake_env_config)

    write_hk_bridge(monkeypatch, tmp_path)
    first = abt._load_universe_panel("hk-local", "2024-01-01/2024-01-31")
    assert len(list((home / ".vibe-trading" / "cache").glob("hk-local_*.pkl"))) == 1

    point_bridge_at(
        monkeypatch,
        tmp_path,
        {
            "sources": [
                {
                    "symbol": symbol,
                    "type": "parquet",
                    "path": str(tmp_path / f"{symbol}.parquet"),
                }
                for symbol in HK_SYMBOLS[:2]
            ]
        },
    )
    second = abt._load_universe_panel("hk-local", "2024-01-01/2024-01-31")
    assert list(first["close"].columns) == list(HK_SYMBOLS)
    assert list(second["close"].columns) == list(HK_SYMBOLS[:2])


def test_every_bench_universe_list_offers_hk_local() -> None:
    assert abt._UNIVERSE_TAG["hk-local"] == "equity_hk"
    assert "hk-local" in cli_handlers._UNIVERSE_CHOICES
    assert cli_handlers._LIST_UNIVERSE_ALIASES["hk-local"] == "equity_hk"
    assert "hk-local" in alpha_routes._BENCH_UNIVERSES


def _fake_env_config() -> object:
    api = type("_Api", (), {"api_auth_key": "test-key"})()
    return type("_Cfg", (), {"api": api})()
