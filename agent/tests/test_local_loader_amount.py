"""``amount`` survives the local loader instead of being silently dropped.

``amount`` is a first-class panel column: ``factors.registry._PRICE_COLS``
lists it, 18 bundled alphas declare it in ``columns_required``, and
``_wide_from_fetched`` takes an ``include_amount`` switch. Dropping it here
meant a local file that carried the column still produced a panel without it,
so those alphas were skipped with nothing to say the data had been there.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import backtest.loaders.local_loader as local_loader

_ROWS = [
    {"date": "2024-01-02", "open": 10.0, "high": 10.5, "low": 9.8,
     "close": 10.2, "volume": 100, "amount": 1020.0},
    {"date": "2024-01-03", "open": 10.2, "high": 10.6, "low": 10.0,
     "close": 10.4, "volume": 200, "amount": 2080.0},
]


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rows: list[dict]) -> None:
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"sources": [{"symbol": "TEST.HK", "type": "csv", "path": str(csv_path)}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(local_loader, "_CONFIG_PATH", config_path)


def _fetch() -> pd.DataFrame | None:
    return local_loader.DataLoader().fetch(["TEST.HK"], "2024-01-01", "2024-01-31").get("TEST.HK")


def test_amount_column_is_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path, _ROWS)
    frame = _fetch()
    assert frame is not None
    assert "amount" in frame.columns
    assert pytest.approx(1020.0) == frame["amount"].iloc[0]


def test_file_without_amount_is_unaffected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """True negative: the column is optional, and its absence is not an error."""
    rows = [{k: v for k, v in r.items() if k != "amount"} for r in _ROWS]
    _configure(monkeypatch, tmp_path, rows)
    frame = _fetch()
    assert frame is not None
    assert "amount" not in frame.columns
    assert len(frame) == len(rows)


def test_amount_is_numeric_not_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An object-dtype amount would flow into arithmetic and fail far from here."""
    _configure(monkeypatch, tmp_path, _ROWS)
    frame = _fetch()
    assert frame is not None
    assert frame["amount"].dtype.kind == "f"
