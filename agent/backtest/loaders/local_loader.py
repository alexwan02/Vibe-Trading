"""Local data loader: reads CSV, Parquet, and DuckDB files from user config.

Configuration lives at ``~/.vibe-trading/data-bridge/config.yaml``.
Each entry maps a symbol to a data file with optional column-name overrides,
date format, and (for DuckDB) an SQL query.

``invalid_bars`` decides what happens to bars that fail the OHLC structural
invariants: ``drop`` (default), ``warn`` (keep the bar and log), or ``raise``
(fail the fetch with :class:`InvalidBarsError`). Set it at the top level, per
source, or both; the per-source value wins. Dropping is the safe default, but
it also hides bars that are real yet sit a tick outside the session range,
such as an opening-auction ``open`` that the vendor's ``high``/``low`` omit.

Note:
    The policy applies to frames this loader returns. ``backtest.runner``
    re-validates every fetched frame with the default ``drop`` policy before
    the engine sees it, so ``warn`` keeps flagged bars only for callers that
    consume the loader directly rather than through the runner. ``raise``
    still fails a backtest, because it fires while the loader reads the file.

Example config::

    invalid_bars: warn      # optional; drop (default) | warn | raise

    sources:
      - symbol: "AAPL.US"
        type: csv
        path: "~/data/aapl_2024.csv"
        columns:
          date: "Date"
          open: "Open"
          high: "High"
          low: "Low"
          close: "Close"
          volume: "Volume"
        date_format: "%Y-%m-%d"

      - symbol: "BTC-USDT"
        type: parquet
        path: "~/data/btc.parquet"
        invalid_bars: raise   # overrides the top-level setting for this symbol

      - symbol: "MYINDEX"
        type: duckdb
        db_path: "~/data/market.duckdb"
        query: "SELECT * FROM prices WHERE ticker = 'MYINDEX'"
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast, get_args

import pandas as pd
import yaml

from backtest.loaders.base import cached_loader_fetch, validate_date_range, validate_ohlc
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".vibe-trading" / "data-bridge"
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

InvalidBarsPolicy = Literal["drop", "warn", "raise"]
_INVALID_BAR_STRATEGIES: tuple[str, ...] = get_args(InvalidBarsPolicy)
# ``drop`` is what the loader always did, so a config without the key is
# unaffected.
_DEFAULT_INVALID_BARS: InvalidBarsPolicy = "drop"


class InvalidBarsError(ValueError):
    """A source configured with ``invalid_bars: raise`` holds invalid bars.

    Raised out of :meth:`DataLoader.fetch` rather than logged and skipped: the
    user asked for a hard failure, and a symbol that silently goes missing is
    the quiet outcome ``raise`` exists to rule out.
    """


_DEFAULT_COLUMNS = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

# Map the project's bar intervals (see ``backtest.runner._VALID_INTERVALS``) to
# pandas offset aliases used for resampling. All entries are fixed-duration so
# ``pd.Timedelta`` of the alias is well defined.
_RESAMPLE_RULES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "1h": "1h",
    "4H": "4h",
    "4h": "4h",
    "1D": "1D",
    "1d": "1D",
}

_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def _resample_to_interval(df: pd.DataFrame, interval: str, symbol: str) -> pd.DataFrame:
    """Resample an OHLCV frame to the requested bar ``interval``.

    Local files can hold any native granularity, so a requested interval the
    file does not already match must be honored explicitly. Previously
    ``interval`` was passed only to the cache key, so requesting e.g. ``4H``
    against hourly data silently returned the file's native bars.

    Coarser-than-source requests are downsampled with the standard OHLCV
    aggregation. A finer-than-source request cannot be fabricated from a file,
    so the source bars are returned unchanged with a warning. Requests that
    already match the source granularity are returned unchanged.
    """
    rule = _RESAMPLE_RULES.get(interval)
    if df.empty:
        return df
    if rule is None:
        logger.warning(
            "local loader: unsupported interval %r for %s; returning source bars",
            interval,
            symbol,
        )
        return df

    target = pd.Timedelta(rule)
    if len(df.index) >= 2:
        source = df.index.to_series().diff().dropna().median()
        if pd.notna(source):
            if target < source:
                logger.warning(
                    "local loader: cannot upsample %s source bars to %s for %s; "
                    "returning source bars",
                    source,
                    interval,
                    symbol,
                )
                return df
            if target == source:
                return df

    resampled = df.resample(rule).agg(_OHLCV_AGG)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled.index.name = df.index.name
    return resampled


def _source_cache_identity(
    entry: dict[str, Any], invalid_bars: InvalidBarsPolicy = _DEFAULT_INVALID_BARS
) -> str:
    """Hash reader settings while excluding unrelated YAML annotations.

    Args:
        entry: One ``sources`` entry from the bridge config.
        invalid_bars: Resolved invalid-bar policy. Part of the identity because
            ``drop`` and ``warn`` return different rows from the same file.

    Returns:
        Hex digest identifying what the reader would return for ``entry``.
    """
    identity: dict[str, Any] = {
        key: entry.get(key)
        for key in ("type", "path", "db_path", "query", "date_format")
    }
    identity["invalid_bars"] = invalid_bars
    columns = entry.get("columns")
    identity["columns"] = (
        sorted(
            (str(key), value)
            for key, value in columns.items()
            if isinstance(value, str)
        )
        if isinstance(columns, dict)
        else None
    )
    payload = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_file_revision(entry: dict[str, Any]) -> str:
    # A local source can be corrected or replaced without changing its YAML.
    # Identify filesystem revisions as well as reader declarations so settled
    # cached bars cannot hide the updated file. DuckDB writes may reside in WAL.
    source_type = str(entry.get("type") or "csv").strip().lower()
    source_path = (
        entry.get("db_path") if source_type == "duckdb" else entry.get("path")
    )
    revisions = []
    if isinstance(source_path, str) and source_path.strip():
        reader_path = source_path if source_type == "duckdb" else source_path.strip()
        path = Path(reader_path).expanduser()
        paths = [path, Path(str(path) + ".wal")] if source_type == "duckdb" else [path]
        for file in paths:
            try:
                stat = file.stat()
                revisions.append(
                    (str(file), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                )
            except OSError:
                revisions.append((str(file), None))
    payload = json.dumps(revisions).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _load_config() -> dict[str, Any] | None:
    if not _CONFIG_PATH.exists():
        return None
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_columns(
    df: pd.DataFrame,
    col_map: dict[str, str],
    date_fmt: str | None,
    invalid_bars: InvalidBarsPolicy = _DEFAULT_INVALID_BARS,
) -> pd.DataFrame | None:
    """Map a raw file frame onto the loader's UTC-naive OHLCV contract.

    Args:
        df: Frame as read from the source file or query.
        col_map: Standard column name -> source column name.
        date_fmt: Optional ``strptime`` format for the date column.
        invalid_bars: What to do with bars that fail the OHLC invariants.

    Returns:
        The normalized frame, or None when required columns are missing.

    Raises:
        InvalidBarsError: ``invalid_bars="raise"`` and a bar is invalid.
    """
    # A DatetimeIndex carries the bar dates (parquet files, and duckdb
    # relations whose query set one). An explicitly mapped date column wins
    # over the index; otherwise the index becomes the date column, whatever
    # date_fmt is configured. Dropping the index on the first path also
    # sidesteps the reset_index crash when index and column share a name.
    if isinstance(df.index, pd.DatetimeIndex):
        if col_map.get("date", "date") in df.columns:
            df = df.reset_index(drop=True)
        else:
            df = df.reset_index()
            col_map = {**col_map, "date": df.columns[0]}
    rename: dict[str, str] = {}
    for std_name, src_name in col_map.items():
        if std_name == "date":
            continue
        if src_name in df.columns:
            rename[src_name] = std_name
    df = df.rename(columns=rename)

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    date_col = col_map.get("date", "date")
    if date_col not in df.columns:
        return None

    if date_fmt:
        parsed_dates = pd.to_datetime(
            df[date_col], format=date_fmt, errors="coerce", utc=True
        )
    else:
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    # Normalize every input to the loader's UTC-naive index contract. Parsing
    # as UTC first also handles files that span daylight-saving offset changes.
    df["trade_date"] = parsed_dates.dt.tz_convert(None)
    df = df.dropna(subset=["trade_date"])
    df = df.set_index("trade_date").sort_index()

    ohlcv_cols: list[str] = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        ohlcv_cols.append("volume")

    df = df[ohlcv_cols]
    for col in ohlcv_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    try:
        df = validate_ohlc(df, strategy=invalid_bars)
    except ValueError as exc:
        raise InvalidBarsError(str(exc)) from exc
    for col in ohlcv_cols:
        df[col] = df[col].astype("float64")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df


def _read_csv(
    path: str,
    col_map: dict[str, str],
    date_fmt: str | None,
    invalid_bars: InvalidBarsPolicy = _DEFAULT_INVALID_BARS,
) -> pd.DataFrame | None:
    """Read and normalize one CSV source; see :func:`_normalize_columns`."""
    df = pd.read_csv(path)
    return _normalize_columns(df, col_map, date_fmt, invalid_bars)


def _read_parquet(
    path: str,
    col_map: dict[str, str],
    date_fmt: str | None,
    invalid_bars: InvalidBarsPolicy = _DEFAULT_INVALID_BARS,
) -> pd.DataFrame | None:
    """Read and normalize one Parquet source; see :func:`_normalize_columns`."""
    df = pd.read_parquet(path)
    return _normalize_columns(df, col_map, date_fmt, invalid_bars)


def _read_duckdb(
    db_path: str,
    query: str,
    col_map: dict[str, str],
    date_fmt: str | None,
    invalid_bars: InvalidBarsPolicy = _DEFAULT_INVALID_BARS,
) -> pd.DataFrame | None:
    """Run one DuckDB query and normalize it; see :func:`_normalize_columns`."""
    import duckdb

    with duckdb.connect(db_path, read_only=True) as conn:
        df = conn.execute(query).df()
    return _normalize_columns(df, col_map, date_fmt, invalid_bars)


def _resolve_invalid_bars(
    entry: dict[str, Any], config: dict[str, Any]
) -> InvalidBarsPolicy:
    """Return the invalid-bar policy for one source.

    The per-source value wins over the top-level one. An unrecognised value
    falls back to ``drop`` and logs a warning: accepting a typo silently would
    leave the user believing flagged bars are kept while they are dropped.

    Args:
        entry: One ``sources`` entry from the bridge config.
        config: The whole bridge config.

    Returns:
        One of ``drop``, ``warn`` or ``raise``.
    """
    for scope, raw in (
        ("source", entry.get("invalid_bars")),
        ("top", config.get("invalid_bars")),
    ):
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in _INVALID_BAR_STRATEGIES:
            return cast(InvalidBarsPolicy, value)
        logger.warning(
            "local loader: unknown %s-level invalid_bars %r for symbol %s; "
            "expected one of %s, using %r",
            scope,
            raw,
            entry.get("symbol", "?"),
            ", ".join(_INVALID_BAR_STRATEGIES),
            _DEFAULT_INVALID_BARS,
        )
        return _DEFAULT_INVALID_BARS
    return _DEFAULT_INVALID_BARS


_READERS = {
    "csv": _read_csv,
    "parquet": _read_parquet,
    "duckdb": _read_duckdb,
}


@register
class DataLoader:
    """Config-driven local data loader for CSV, Parquet, and DuckDB."""

    name = "local"
    markets = {"us_equity", "a_share", "hk_equity", "crypto", "futures", "forex", "macro", "fund"}
    requires_auth = False

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None
        self._source_by_symbol: dict[str, dict[str, Any]] = {}

    def is_available(self) -> bool:
        """Return True when the YAML config file exists and has sources."""
        config = _load_config()
        if config is None:
            return False
        sources = config.get("sources")
        return isinstance(sources, list) and len(sources) > 0

    def _ensure_config(self) -> None:
        if self._config is not None:
            return
        self._config = _load_config() or {}
        self._source_by_symbol = {}
        for entry in self._config.get("sources", []):
            symbol = entry.get("symbol", "").strip()
            if symbol:
                self._source_by_symbol[symbol] = entry

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for each code from configured local sources.

        Args:
            codes: Symbol list, optionally prefixed with ``local:``.
            start_date: YYYY-MM-DD.
            end_date: YYYY-MM-DD.
            interval: Bar size (all intervals supported if data contains them).
            fields: Ignored.

        Returns:
            Mapping clean_symbol -> OHLCV DataFrame.

        Raises:
            InvalidBarsError: A source set to ``invalid_bars: raise`` holds a
                bar that fails the OHLC invariants. Other per-symbol failures
                are logged and the symbol is left out of the result.
        """
        validate_date_range(start_date, end_date)
        self._ensure_config()

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            clean = code.split(":", 1)[1] if code.startswith("local:") else code
            entry = self._source_by_symbol.get(clean)
            if entry is None:
                logger.warning("local loader: no config entry for symbol %s", clean)
                continue
            invalid_bars = _resolve_invalid_bars(entry, self._config or {})
            try:
                df = cached_loader_fetch(
                    source=self.name,
                    symbol=clean,
                    timeframe=interval,
                    start_date=start_date,
                    end_date=end_date,
                    # Local symbols are user-defined: the same name can point
                    # to another file/query/schema. Keep settled-data caching,
                    # but key it by reader settings as well as the range.
                    fields=[_source_cache_identity(entry, invalid_bars)],
                    source_revision=_source_file_revision(entry),
                    fetch=lambda c=clean, policy=invalid_bars: self._fetch_one(
                        c, start_date, end_date, interval, policy
                    ),
                )
                if df is not None and not df.empty:
                    result[clean] = df
            except InvalidBarsError as exc:
                raise InvalidBarsError(
                    f"local loader: {clean}: {exc} (invalid_bars: raise)"
                ) from exc
            except Exception as exc:
                logger.warning("local loader failed for %s: %s", clean, exc)

        return result

    def _fetch_one(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1D",
        invalid_bars: InvalidBarsPolicy = _DEFAULT_INVALID_BARS,
    ) -> pd.DataFrame | None:
        """Read one configured source and trim it to the requested window.

        Args:
            symbol: Configured symbol.
            start_date: Inclusive start date.
            end_date: Inclusive end date.
            interval: Requested bar size.
            invalid_bars: Resolved invalid-bar policy for this source.

        Returns:
            OHLCV frame, or None when the source is unusable or empty.
        """
        entry = self._source_by_symbol.get(symbol)
        if entry is None:
            return None

        src_type: str = entry.get("type", "csv").strip().lower()
        reader = _READERS.get(src_type)
        if reader is None:
            logger.warning("local loader: unsupported type %r for symbol %s", src_type, symbol)
            return None

        col_map: dict[str, str] = dict(_DEFAULT_COLUMNS)
        user_cols = entry.get("columns")
        if isinstance(user_cols, dict):
            for k, v in user_cols.items():
                if isinstance(v, str):
                    col_map[k] = v

        date_fmt: str | None = None
        user_fmt = entry.get("date_format")
        if isinstance(user_fmt, str) and user_fmt.strip():
            date_fmt = user_fmt.strip()

        if src_type == "duckdb":
            db_path = str(Path(entry.get("db_path", "")).expanduser())
            query = entry.get("query", "").strip()
            if not db_path or not query:
                logger.warning(
                    "local loader: missing db_path or query for duckdb symbol %s", symbol
                )
                return None
            df = _read_duckdb(db_path, query, col_map, date_fmt, invalid_bars)
        else:
            path = entry.get("path", "").strip()
            if not path:
                logger.warning("local loader: missing path for symbol %s", symbol)
                return None
            expanded = str(Path(path).expanduser())
            df = reader(expanded, col_map, date_fmt, invalid_bars)

        if df is None:
            return None

        start = pd.Timestamp(start_date)
        # Treat ``end_date`` as inclusive of the whole day so intraday bars on
        # the end day survive the filter (a bare midnight bound dropped them,
        # which would defeat any sub-daily ``interval``).
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        # A half-open boundary includes fractional seconds on the end day
        # without admitting the following midnight.
        df = df[(df.index >= start) & (df.index < end)]
        if df.empty:
            return None
        df = _resample_to_interval(df, interval, symbol)
        if df.empty:
            return None
        return df
