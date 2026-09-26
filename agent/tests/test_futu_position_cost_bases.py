"""Futu reports two cost bases; the mapping must carry both.

The rows below are the shapes Futu actually returned on a live HK book on
2026-09-26, trimmed to the fields under test.
"""

from __future__ import annotations

from src.trading.connectors.futu import sdk as futu_sdk

# 01810 after round-trip trading: realised gains have pushed the *diluted* cost
# negative, so `pl_ratio` -- which is computed on that basis -- is filled with
# 0.0 while `pl_ratio_valid` stays True. The average-cost pair is the readable
# one.
DILUTED_COST_WENT_NEGATIVE = {
    "code": "HK.01810",
    "qty": 200.0,
    "can_sell_qty": 200.0,
    "cost_price": -2.78,
    "cost_price_valid": True,
    "average_cost": 23.738,
    "market_val": 5180.0,
    "pl_ratio": 0.0,
    "pl_ratio_valid": True,
    "pl_ratio_avg_cost": 9.11,
    "pl_val": 5736.0,
    "pl_val_valid": True,
    "unrealized_pl": 432.3125,
    "realized_pl": 5303.6875,
    "position_side": "LONG",
    "position_market": "HK",
    "currency": "HKD",
}

# 01428: `pl_val` is the sum of a large unrealised loss and a realised gain.
PL_VAL_MIXES_REALISED = {
    "code": "HK.01428",
    "qty": 2000.0,
    "cost_price": 9.12,
    "average_cost": 12.813,
    "market_val": 13970.0,
    "pl_ratio": -23.41,
    "pl_ratio_avg_cost": -45.49,
    "pl_val": -4270.0,
    "unrealized_pl": -11655.0,
    "realized_pl": 7385.0,
    "position_market": "HK",
    "currency": "HKD",
}


def test_both_cost_bases_are_carried() -> None:
    """Without `average_cost`, a negative `cost_price` reads as a bad value."""
    row = futu_sdk._position_to_dict(DILUTED_COST_WENT_NEGATIVE)

    assert row["cost_price"] == -2.78
    assert row["average_cost"] == 23.738


def test_both_pl_ratios_are_carried() -> None:
    """`pl_ratio` 0.0 here means "cannot be computed", not "flat".

    Futu leaves `pl_ratio_valid` True in this state, so nothing in the
    payload distinguishes the two. The average-cost ratio is the only
    readable return on this position and must survive the mapping.
    """
    row = futu_sdk._position_to_dict(DILUTED_COST_WENT_NEGATIVE)

    assert row["pl_ratio"] == 0.0
    assert row["pl_ratio_avg_cost"] == 9.11


def test_pl_val_splits_into_realised_and_unrealised() -> None:
    """The identity is what makes `pl_val / market_val` visibly wrong."""
    row = futu_sdk._position_to_dict(PL_VAL_MIXES_REALISED)

    assert row["unrealized_pl"] == -11655.0
    assert row["realized_pl"] == 7385.0
    assert row["unrealized_pl"] + row["realized_pl"] == row["pl_val"]


def test_original_keys_are_untouched() -> None:
    """Keys are added, never renamed or dropped -- existing callers read these."""
    row = futu_sdk._position_to_dict(DILUTED_COST_WENT_NEGATIVE)

    for key in (
        "code",
        "qty",
        "can_sell_qty",
        "cost_price",
        "market_val",
        "pl_ratio",
        "pl_val",
        "position_side",
        "market",
        "currency",
    ):
        assert key in row
    assert row["market"] == "HK"
    assert row["currency"] == "HKD"


def test_missing_new_fields_become_none_not_an_error() -> None:
    """A broker row without the extra fields must still map.

    Futu's own paper accounts and the other connectors' rows do not carry
    them; the mapping degrades to None rather than raising, which is what
    `_first` already does for every other optional field.
    """
    row = futu_sdk._position_to_dict(
        {"code": "HK.SYNTH", "qty": 100.0, "market_val": 800.0, "currency": "HKD"}
    )

    assert row["average_cost"] is None
    assert row["pl_ratio_avg_cost"] is None
    assert row["unrealized_pl"] is None
    assert row["realized_pl"] is None
