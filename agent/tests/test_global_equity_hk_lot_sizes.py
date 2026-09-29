"""HK board lots are per-symbol, so ``round_size`` must not assume 100.

Measured across 103 Hong Kong names: only 24 (23%) trade in lots of 100. The
rest span 20 (00100 MINIMAX-W) to 5000 (06051), through 50/200/250/400/500/
1000/2000/3000/4000. Rounding every HK order to a flat 100 leaves 77% of the
sizes wrong, and the failure is quiet -- a target smaller than one assumed lot
rounds to zero and the symbol simply never trades.

A symbol missing from the table still gets 100, but the assumption is recorded
in ``lot_assumptions`` so a run can say which lot sizes were measured and which
were guessed.
"""

from __future__ import annotations

import pytest

from backtest.engines.global_equity import GlobalEquityEngine


LOTS = {"00100.HK": 20, "00700.HK": 100, "06051.HK": 5000}


def _engine(**extra) -> GlobalEquityEngine:
    config = {"initial_cash": 1_000_000, "hk_lot_sizes": LOTS, **extra}
    return GlobalEquityEngine(config, market="hk")


@pytest.mark.parametrize(
    ("symbol", "raw", "expected"),
    [
        ("00100.HK", 99, 80),      # lot 20 -> 4 lots, not 0 under a 100 grid
        ("00100.HK", 19, 0),       # below one real lot
        ("00700.HK", 250, 200),    # lot 100 unchanged
        ("06051.HK", 12_345, 10_000),  # lot 5000
    ],
)
def test_round_size_uses_the_symbols_own_lot(symbol, raw, expected):
    engine = _engine()
    engine._active_symbol = symbol

    assert engine.round_size(raw, price=10.0) == expected


def test_a_symbol_with_a_small_lot_is_tradable_where_a_flat_100_would_not_be():
    """The regression this guards: 99 shares of a lot-20 name is 4 lots."""
    engine = _engine()
    engine._active_symbol = "00100.HK"

    assert engine.round_size(99, price=10.0) > 0
    assert engine.round_size(99, price=10.0) % LOTS["00100.HK"] == 0


def test_unknown_symbol_falls_back_to_100_and_says_so():
    engine = _engine()
    engine._active_symbol = "09999.HK"

    assert engine.round_size(250, price=10.0) == 200
    assert engine.lot_assumptions == {"09999.HK": 100}


def test_missing_active_symbol_is_recorded_rather_than_silently_sized():
    engine = _engine()
    engine._active_symbol = ""

    assert engine.round_size(250, price=10.0) == 200
    assert engine.lot_assumptions == {"<no active symbol>": 100}


def test_no_lot_table_reproduces_the_previous_behaviour():
    """Without ``hk_lot_sizes`` the engine must behave exactly as before."""
    engine = GlobalEquityEngine({"initial_cash": 1_000_000}, market="hk")
    engine._active_symbol = "00100.HK"

    assert engine.round_size(99, price=10.0) == 0
    assert engine.lot_assumptions == {"00100.HK": 100}


def test_lot_table_keys_tolerate_leading_zero_differences():
    engine = GlobalEquityEngine(
        {"initial_cash": 1_000_000, "hk_lot_sizes": {"100.HK": 20}}, market="hk"
    )
    engine._active_symbol = "00100.HK"

    assert engine.round_size(99, price=10.0) == 80
    assert engine.lot_assumptions == {}
