"""``invalid_bars``: keep, drop, or reject bars that fail the OHLC invariants.

Dropping is silent at the data level. A vendor whose daily ``open`` comes from
the opening auction while ``high``/``low`` cover the continuous session only
prints a handful of bars where ``open`` sits a tick outside the range — real
bars, not corrupt ones. A caller reconciling row counts against the source file
then sees an unexplained shortfall with only an INFO log to go on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

import backtest.loaders.local_loader as local_loader

# One bar has ``open`` a tick above ``high`` — the auction-print shape.
_ROWS = [
    {"date": "2024-01-02", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "volume": 100},
    {"date": "2024-01-03", "open": 10.6, "high": 10.5, "low": 9.9, "close": 10.1, "volume": 200},
    {"date": "2024-01-04", "open": 10.1, "high": 10.4, "low": 9.7, "close": 10.3, "volume": 300},
]


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict) -> Path:
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(_ROWS).to_csv(csv_path, index=False)
    for entry in config["sources"]:
        entry.setdefault("path", str(csv_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(local_loader, "_CONFIG_PATH", config_path)
    return csv_path


def _fetch(symbol: str = "TEST.HK") -> pd.DataFrame | None:
    loader = local_loader.DataLoader()
    return loader.fetch([symbol], "2024-01-01", "2024-01-31").get(symbol)


def test_defaults_to_dropping_so_existing_configs_are_unaffected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No ``invalid_bars`` key anywhere: the offending bar is still dropped."""
    _configure(monkeypatch, tmp_path, {"sources": [{"symbol": "TEST.HK", "type": "csv"}]})
    frame = _fetch()
    assert frame is not None
    assert len(frame) == len(_ROWS) - 1


def test_warn_keeps_the_bar_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _configure(
        monkeypatch,
        tmp_path,
        {"invalid_bars": "warn", "sources": [{"symbol": "TEST.HK", "type": "csv"}]},
    )
    with caplog.at_level(logging.WARNING):
        frame = _fetch()
    assert frame is not None
    assert len(frame) == len(_ROWS)
    assert pytest.approx(10.6) == frame["open"].iloc[1]


def test_raise_surfaces_the_bar_instead_of_returning_a_short_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``fetch`` swallows loader errors, so the symbol is absent rather than short.

    That is the point: a missing symbol is loud, a frame that is one row short
    is not.
    """
    _configure(
        monkeypatch,
        tmp_path,
        {"invalid_bars": "raise", "sources": [{"symbol": "TEST.HK", "type": "csv"}]},
    )
    assert _fetch() is None


def test_per_source_setting_overrides_the_file_level_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(
        monkeypatch,
        tmp_path,
        {
            "invalid_bars": "drop",
            "sources": [{"symbol": "TEST.HK", "type": "csv", "invalid_bars": "warn"}],
        },
    )
    frame = _fetch()
    assert frame is not None
    assert len(frame) == len(_ROWS)


def test_unknown_value_falls_back_to_drop_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Silently accepting a typo would leave the caller believing bars are kept."""
    _configure(
        monkeypatch,
        tmp_path,
        {"sources": [{"symbol": "TEST.HK", "type": "csv", "invalid_bars": "warm"}]},
    )
    with caplog.at_level(logging.WARNING):
        frame = _fetch()
    assert frame is not None
    assert len(frame) == len(_ROWS) - 1
    assert "invalid_bars" in caplog.text


def test_clean_data_is_untouched_by_any_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """True-negative: a file with no violations returns every row under each mode."""
    clean = [r for r in _ROWS if r["date"] != "2024-01-03"]
    csv_path = tmp_path / "clean.csv"
    pd.DataFrame(clean).to_csv(csv_path, index=False)
    for mode in local_loader._INVALID_BAR_STRATEGIES:
        config_path = tmp_path / f"config_{mode}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "invalid_bars": mode,
                    "sources": [
                        {"symbol": "TEST.HK", "type": "csv", "path": str(csv_path)}
                    ],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(local_loader, "_CONFIG_PATH", config_path)
        frame = _fetch()
        assert frame is not None, mode
        assert len(frame) == len(clean), mode
