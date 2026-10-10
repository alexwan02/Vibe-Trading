"""``hk-local`` fills ``fund:*`` and ``sector`` from optional bridge files."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.factors.universes import hk_local
from src.tools import alpha_bench_tool as abt
from tests.local_bridge_helpers import HK_SYMBOLS, write_hk_bridge

START, END = "2024-01-01", "2024-01-31"
A, B, C = HK_SYMBOLS


def _fund_file(path: Path, rows: list[dict]) -> str:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)


def _panel() -> dict:
    return hk_local.load_panel(hk_local.open_bridge(), START, END)


def _on(frame: pd.DataFrame, day: str, code: str) -> float:
    return frame.loc[pd.Timestamp(day), code]


def test_values_appear_on_known_at_and_the_first_disclosure_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [
        {"period": "2023-12-31", "known_at": "2024-01-04", "net_income": 100.0},
        # Restatement of the same period, published later: never used.
        {"period": "2023-12-31", "known_at": "2024-01-08", "net_income": 999.0},
        {"period": "2024-03-31", "known_at": "2024-01-09", "net_income": 200.0},
        {"period": "2024-06-30", "known_at": None, "net_income": 300.0},
    ]
    fund = _fund_file(tmp_path / "a_fund.parquet", rows)
    write_hk_bridge(monkeypatch, tmp_path, extra={A: {"fundamentals": fund}})
    net_income = _panel()["fund:net_income"]

    # The period ended before the panel starts; anchoring on it would leak.
    assert np.isnan(_on(net_income, "2024-01-03", A))
    assert _on(net_income, "2024-01-04", A) == 100.0
    assert _on(net_income, "2024-01-08", A) == 100.0
    assert _on(net_income, "2024-01-09", A) == 200.0
    assert net_income[B].isna().all()


def test_timezone_aware_known_at_keeps_its_local_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An early-morning Hong Kong release is still that day's news, not the day before."""
    known = pd.Series(pd.to_datetime(["2024-01-04 06:00"])).dt.tz_localize(
        "Asia/Hong_Kong"
    )
    path = tmp_path / "a_fund.parquet"
    pd.DataFrame(
        {"period": ["2023-12-31"], "known_at": known, "total_assets": [5.0]}
    ).to_parquet(path, index=False)
    write_hk_bridge(monkeypatch, tmp_path, extra={A: {"fundamentals": str(path)}})
    total_assets = _panel()["fund:total_assets"]
    assert np.isnan(_on(total_assets, "2024-01-03", A))
    assert _on(total_assets, "2024-01-04", A) == 5.0


def test_file_without_period_column_still_loads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fund = _fund_file(tmp_path / "a.parquet", [{"known_at": "2024-01-02", "cash": 7.0}])
    write_hk_bridge(monkeypatch, tmp_path, extra={A: {"fundamentals": fund}})
    with caplog.at_level(logging.WARNING, logger=hk_local.logger.name):
        cash = _panel()["fund:cash"]
    assert _on(cash, "2024-01-02", A) == 7.0
    assert "no period column" in caplog.text


def test_one_corrupt_file_is_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    good = _fund_file(
        tmp_path / "a.parquet",
        [{"period": "2023-12-31", "known_at": "2024-01-02", "cash": 1.0}],
    )
    corrupt = tmp_path / "b.parquet"
    corrupt.write_bytes(b"not a parquet file")
    write_hk_bridge(
        monkeypatch,
        tmp_path,
        extra={A: {"fundamentals": good}, B: {"fundamentals": str(corrupt)}},
    )
    with caplog.at_level(logging.WARNING, logger=hk_local.logger.name):
        panel = _panel()
    assert _on(panel["fund:cash"], "2024-01-02", A) == 1.0
    assert panel["fund:cash"][B].isna().all()
    assert f"skipping fundamentals for {B}" in caplog.text
    assert panel["_meta"]["fundamental_fields"] == ["cash"]


def test_derived_and_fallback_fields_use_the_schema_formulas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    row = {
        "period": "2023-12-31",
        "known_at": "2024-01-02",
        "net_income": 10.0,
        "total_equity": 40.0,
        "revenue": 100.0,
        "cogs": 60.0,
        "total_assets": 200.0,
        "currency": "HKD",
    }
    fund = _fund_file(tmp_path / "a.parquet", [row])
    write_hk_bridge(monkeypatch, tmp_path, extra={A: {"fundamentals": fund}})
    panel = _panel()
    assert _on(panel["fund:roe"], "2024-01-03", A) == pytest.approx(0.25)
    assert _on(panel["fund:gross_profit"], "2024-01-03", A) == pytest.approx(40.0)
    assert _on(panel["fund:gross_profitability"], "2024-01-03", A) == pytest.approx(0.2)
    assert "fund:currency" not in panel
    assert panel["_meta"]["fundamental_pit"] is True


def test_fundamentals_path_expands_the_home_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _fund_file(
        tmp_path / "a.parquet",
        [{"period": "2023-12-31", "known_at": "2024-01-02", "cash": 3.0}],
    )
    write_hk_bridge(monkeypatch, tmp_path, extra={A: {"fundamentals": "~/a.parquet"}})
    assert _on(_panel()["fund:cash"], "2024-01-02", A) == 3.0


def _sector_file(path: Path, labels: dict[str, str | None]) -> str:
    pd.DataFrame({"code": list(labels), "sector": list(labels.values())}).to_parquet(
        path, index=False
    )
    return str(path)


def test_sector_tag_is_dropped_below_the_coverage_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    table = _sector_file(tmp_path / "sectors.parquet", {A: "Tech", B: "Banks", C: None})
    write_hk_bridge(
        monkeypatch, tmp_path, extra={s: {"sector_table": table} for s in HK_SYMBOLS}
    )
    panel = _panel()
    assert "sector" not in panel
    assert panel["_meta"]["sector_source"] is None
    assert panel["_meta"]["sector_coverage"] == pytest.approx(0.6667)


def test_sector_tag_leaves_unlabelled_names_blank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    table = _sector_file(tmp_path / "sectors.parquet", {A: "Tech", B: "Banks", C: " "})
    write_hk_bridge(
        monkeypatch, tmp_path, extra={s: {"sector_table": table} for s in HK_SYMBOLS}
    )
    monkeypatch.setattr(abt, "_MIN_SECTOR_COVERAGE", 0.5)
    panel = _panel()
    sector = panel["sector"]
    assert sector.shape == panel["close"].shape
    assert sector[A].eq("Tech").all() and sector[B].eq("Banks").all()
    assert sector[C].isna().all()
    assert panel["_meta"]["sector_source"] == "local data bridge (sectors.parquet)"


def test_several_sector_tables_pick_the_first_sorted_and_warn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = _sector_file(tmp_path / "a.parquet", {s: "First" for s in HK_SYMBOLS})
    second = _sector_file(tmp_path / "b.parquet", {s: "Second" for s in HK_SYMBOLS})
    write_hk_bridge(
        monkeypatch,
        tmp_path,
        extra={A: {"sector_table": second}, B: {"sector_table": first}},
    )
    with caplog.at_level(logging.WARNING, logger=hk_local.logger.name):
        panel = _panel()
    assert panel["sector"].eq("First").all().all()
    assert "2 sector tables configured" in caplog.text


def test_cache_key_changes_when_a_fundamentals_file_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a.parquet"
    rows = [{"period": "2023-12-31", "known_at": "2024-01-02", "cash": 1.0}]
    write_hk_bridge(
        monkeypatch, tmp_path, extra={A: {"fundamentals": _fund_file(path, rows)}}
    )
    before = hk_local.cache_fingerprint(hk_local.open_bridge())
    _fund_file(
        path, rows + [{"period": "2024-03-31", "known_at": "2024-04-02", "cash": 2.0}]
    )
    assert hk_local.cache_fingerprint(hk_local.open_bridge()) != before


def test_bench_uses_bridge_fundamentals_instead_of_sec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_sec(**_kwargs: object) -> None:
        raise AssertionError("hk-local must not pull SEC filings")

    monkeypatch.setattr(
        "backtest.loaders.fundamentals_loader.load_fundamental_panel", no_sec
    )
    registry = SimpleNamespace(
        get=lambda _aid: SimpleNamespace(meta={"columns_required": ["fund:roe"]})
    )
    panel = {"close": pd.DataFrame({A: [1.0]})}
    assert abt._validate_bench_data_requirements(registry, ["x"], "hk-local") == [
        "fund:roe"
    ]
    assert (
        abt._prepare_bench_panel(registry, ["x"], panel, "hk-local", "2024-2024")
        is panel
    )
    with pytest.raises(ValueError, match="SEC US filings"):
        abt._validate_bench_data_requirements(registry, ["x"], "csi300")
