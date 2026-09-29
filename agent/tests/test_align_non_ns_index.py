"""_align must honour a data index that is not nanosecond-resolution.

pandas 2 keeps the unit a source gives it. Parquet written by pyarrow reads
back as ``datetime64[us]``, and ``DatetimeIndex.asi8`` then counts
microseconds. Feeding those integers to ``pd.DatetimeIndex()`` reads them as
nanoseconds, so 2015-01-02 lands on 1970-01-17 -- a thousand-fold error that
raises nothing.

The damage is silent and total: the unified calendar no longer matches any
per-symbol index, so every label lookup in the rebalance path misses, no
order is ever planned, and the run still reports exit code 0 with a full
metrics block and zero trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engines.base import _align


def _frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(dates)
    close = np.linspace(10.0, 20.0, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=dates,
    )


def _data_and_signals(unit: str):
    dates = pd.date_range("2015-01-02", periods=8, freq="D").as_unit(unit)
    codes = ["AAA.HK", "BBB.HK"]
    data_map = {c: _frame(dates) for c in codes}
    signal_map = {c: pd.Series(0.5, index=dates) for c in codes}
    return data_map, signal_map, codes, dates


def test_align_preserves_calendar_for_microsecond_index():
    """A [us] index must produce the same calendar as a [ns] one."""
    data_map, signal_map, codes, dates = _data_and_signals("us")

    aligned_dates = _align(data_map, signal_map, codes)[0]

    assert aligned_dates[0] == pd.Timestamp("2015-01-02"), (
        f"calendar start drifted to {aligned_dates[0]}"
    )
    assert list(aligned_dates) == list(dates.as_unit("ns"))


def test_align_calendar_stays_addressable_in_the_source_frame():
    """Every aligned date must still index the frame it came from.

    This is the property the rebalance path depends on: it looks a bar up by
    label (``ts in frame.index``) and silently skips the symbol when it misses.
    """
    data_map, signal_map, codes, _ = _data_and_signals("us")

    aligned_dates = _align(data_map, signal_map, codes)[0]

    for ts in aligned_dates:
        for code in codes:
            assert ts in data_map[code].index, f"{ts} missing from {code}"


def test_align_close_matrix_matches_nanosecond_baseline():
    """Unit normalisation must not disturb the close matrix.

    ``searchsorted`` compares raw int64 on both sides, so normalising only the
    calendar would silently misplace every price.
    """
    us_close = _align(*_data_and_signals("us")[:3])[1]
    ns_close = _align(*_data_and_signals("ns")[:3])[1]

    np.testing.assert_allclose(us_close.values, ns_close.values)
