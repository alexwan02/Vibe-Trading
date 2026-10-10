"""``hk-local`` bench universe: a Hong Kong panel from the local data bridge.

Membership is whatever ``.HK`` symbols the user configured in
``~/.vibe-trading/data-bridge/config.yaml``; other symbols in the same config
are ignored. Optional ``fundamentals`` and ``sector_table`` keys on those
entries fill the ``fund:*`` columns and the ``sector`` tag. See
:mod:`backtest.loaders.local_loader` for the config and file formats.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.loaders import fundamentals_loader as fundamentals
from backtest.loaders.local_loader import DataLoader
from src.factors.base import Market, vwap

logger = logging.getLogger(__name__)

UNIVERSE = "hk-local"
_HK_SUFFIX = ".HK"
# Config keys that point at files whose contents end up in the panel.
_FILE_KEYS = ("path", "db_path", "fundamentals", "sector_table")
_OHLC = ("open", "high", "low", "close")
# Columns of a fundamentals file that describe a row rather than hold a value.
_FUNDAMENTAL_META_COLUMNS = (
    "period",
    "known_at",
    "valid_until",
    "currency",
    "decay_horizon",
)


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
        ``vwap``, ``fund:*`` and ``sector`` when configured) plus a ``_meta``
        dict.

    Raises:
        ValueError: No ``.HK`` source is configured, or a source set to
            ``invalid_bars: raise`` holds an invalid bar.
    """
    from src.tools.alpha_bench_tool import _MIN_SECTOR_COVERAGE, _wide_from_fetched

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

    entries = {code: loader.source_entry(code) or {} for code in fetched}
    fund_fields = _attach_fundamentals(panel, entries)
    sector_source, sector_coverage = _attach_sector(
        panel, entries, _MIN_SECTOR_COVERAGE
    )

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
        "sector_source": sector_source,
        "sector_coverage": round(sector_coverage, 4),
        "fundamental_source": "local data bridge" if fund_fields else None,
        "fundamental_pit": bool(fund_fields),
        "fundamental_fields": fund_fields,
    }
    return panel


def _expanded(entry: dict[str, Any], key: str) -> Path | None:
    """Return the user-expanded path stored under ``key``, or None."""
    raw = entry.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return Path(raw.strip()).expanduser()


def _read_disclosures(path: Path) -> pd.DataFrame:
    """Read one fundamentals file down to the first disclosure of each period.

    ``known_at`` is when the numbers became public. Its wall-clock date is
    kept and any timezone dropped: converting to UTC first could move an
    early-morning Hong Kong release onto the previous day, a day before the
    market saw it.
    """
    table = pd.read_parquet(path)
    if "known_at" not in table.columns:
        raise ValueError("no known_at column")
    known = pd.to_datetime(table["known_at"], errors="coerce")
    if isinstance(known.dtype, pd.DatetimeTZDtype):
        known = known.dt.tz_localize(None)
    table = table.assign(known_at=known.dt.normalize()).dropna(subset=["known_at"])
    if "period" not in table.columns:
        logger.warning(
            "%s: %s has no period column; every row is its own disclosure",
            UNIVERSE,
            path,
        )
        return table.sort_values("known_at", kind="stable")
    # A restatement arrives later than the original; the backtest must see
    # the value the market saw first, as ``fundamentals_loader`` does.
    table = table.assign(_period_key=table["period"].astype(str))
    table = table.sort_values(["known_at", "_period_key"], kind="stable")
    first = ~table["_period_key"].duplicated() | table["period"].isna()
    return table[first].drop(columns="_period_key")


def _dense_series(table: pd.DataFrame, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Expand every value column of one disclosure table onto ``index``."""
    dense: dict[str, pd.Series] = {}
    for column in table.columns:
        if column in _FUNDAMENTAL_META_COLUMNS or str(column).endswith("_json"):
            continue
        sparse = pd.DataFrame(
            {
                "filed": table["known_at"],
                "value": pd.to_numeric(table[column], errors="coerce"),
            }
        ).dropna(subset=["value"])
        if sparse.empty:
            continue
        # Two periods disclosed the same day: the later period (last after
        # the sort) is the newer number.
        sparse = sparse.drop_duplicates("filed", keep="last")
        dense[str(column)] = fundamentals._ffill_pit(sparse, index)
    return dense


def _attach_fundamentals(
    panel: dict[str, Any], entries: dict[str, dict[str, Any]]
) -> list[str]:
    """Fill ``fund:*`` columns from each entry's optional fundamentals file.

    Values appear on their ``known_at`` date and are held forward, never on
    the period end, which would hand the backtest a report before it was
    public. One unreadable file is skipped with a warning rather than
    failing the whole panel.

    Returns:
        Sorted names of the fields attached, without the ``fund:`` prefix.
    """
    if "close" not in panel:
        return []
    close = panel["close"]
    raw: dict[str, dict[str, pd.Series]] = {}
    for code, entry in entries.items():
        path = _expanded(entry, "fundamentals")
        if path is None:
            continue
        try:
            dense = _dense_series(_read_disclosures(path), close.index)
        except Exception as exc:  # noqa: BLE001 — skip one bad file
            logger.warning("%s: skipping fundamentals for %s: %s", UNIVERSE, code, exc)
            continue
        for field, series in dense.items():
            raw.setdefault(field, {})[code] = series
    if not raw:
        return []

    frames = {
        field: pd.DataFrame(by_code, index=close.index).reindex(columns=close.columns)
        for field, by_code in raw.items()
    }
    _add_schema_fields(frames)
    for field, frame in frames.items():
        panel[f"fund:{field}"] = frame
    return sorted(frames)


def _add_schema_fields(frames: dict[str, pd.DataFrame]) -> None:
    """Fill schema fallbacks and derived fields with the schema's own formulas.

    Reusing the schema keeps one definition of each ratio: a second copy of
    ROE written here would drift from the SEC path while both looked right.
    Fallbacks run first because derived fields can depend on them.
    """
    schema = fundamentals._load_schema()
    for wanted in ("raw", "derived"):
        for field in schema.list_supported_fields():
            kind, spec = schema.resolve_field(field)
            deps = fundamentals._derived_dependencies(spec)
            if kind != wanted or not deps or any(d not in frames for d in deps):
                continue
            if kind == "derived" and field in frames:
                continue
            try:
                computed = fundamentals._compute_derived(
                    spec, {dep: frames[dep] for dep in deps}
                )
            except Exception as exc:  # noqa: BLE001 — skip one bad formula
                logger.warning("%s: cannot derive fund:%s: %s", UNIVERSE, field, exc)
                continue
            if field in frames:
                # Reported values win; the formula only fills the gaps.
                computed = frames[field].where(frames[field].notna(), computed)
            frames[field] = computed


def _attach_sector(
    panel: dict[str, Any],
    entries: dict[str, dict[str, Any]],
    min_coverage: float,
) -> tuple[str | None, float]:
    """Attach the ``sector`` tag from the bridge's sector table.

    An industry label is a cross-sectional attribute, so the whole market
    shares one table. Names without a label (an ETF has no industry) stay
    NaN, which the registry masks out of industry-neutralized outputs
    instead of demeaning them as one "unknown" industry. Below
    ``min_coverage`` the tag is left off entirely, so those alphas skip
    rather than run a global demean that claims to be sector-neutral.

    Returns:
        ``(source label, coverage)``; the label is None when no tag was
        attached.
    """
    paths = sorted(
        {str(p) for e in entries.values() if (p := _expanded(e, "sector_table"))}
    )
    if not paths or "close" not in panel:
        return None, 0.0
    if len(paths) > 1:
        logger.warning(
            "%s: %d sector tables configured; using %s", UNIVERSE, len(paths), paths[0]
        )
    try:
        table = pd.read_parquet(paths[0])
        if not {"code", "sector"}.issubset(table.columns):
            raise ValueError("needs code and sector columns")
    except Exception as exc:  # noqa: BLE001 — no tag is better than no panel
        logger.warning("%s: sector table %s unusable: %s", UNIVERSE, paths[0], exc)
        return None, 0.0

    labels = table.dropna(subset=["code"]).astype({"code": str})
    labels = labels.drop_duplicates("code", keep="first").set_index("code")["sector"]
    labels = labels.where(labels.notna() & labels.astype(str).str.strip().ne(""))
    close = panel["close"]
    tagged = close.columns.to_series().astype(str).map(labels)
    coverage = float(tagged.notna().mean()) if len(tagged) else 0.0
    if coverage < min_coverage:
        logger.warning(
            "%s: sector coverage %.1f%% below %.0f%%; leaving the tag off",
            UNIVERSE,
            coverage * 100,
            min_coverage * 100,
        )
        return None, coverage
    row = tagged.to_numpy(dtype=object)[None, :]
    panel["sector"] = pd.DataFrame(
        np.repeat(row, len(close.index), axis=0),
        index=close.index,
        columns=close.columns,
    )
    return f"local data bridge ({Path(paths[0]).name})", coverage


__all__ = [
    "UNIVERSE",
    "bridge_ready",
    "cache_fingerprint",
    "hk_symbols",
    "load_panel",
    "open_bridge",
]
