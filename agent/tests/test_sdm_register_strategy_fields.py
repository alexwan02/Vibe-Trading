"""A strategy registration must carry how the strategy trades, not just its signal.

`Artifact` has had `entry_rules`, `exit_rules`, `position_sizing` and
`decay_horizon` since the store was written, and `sdm_register` exposed none
of them — so an agent registering a strategy could describe the signal and
nothing else. A field nothing can write to is the same as no field
(the same argument `test_sdm_governance_wiring` makes for the governance
block).

`decay_horizon` matters beyond documentation: it is the label span purged
validation uses, so a wrong or absent value changes what counts as leakage.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest


@pytest.fixture
def register(monkeypatch):
    """Fresh store per test, so registrations cannot leak between them."""
    db = pathlib.Path(tempfile.mkdtemp()) / "store.db"
    monkeypatch.setenv("VIBE_TRADING_STRATEGY_STORE_DB_PATH", str(db))
    import src.strategy_store._shared as shared

    monkeypatch.setattr(shared, "_store", None)
    from src.tools.sdm_register_tool import SdmRegisterTool

    return SdmRegisterTool()


def _register(tool, **kwargs) -> dict:
    payload = {"artifact_type": "strategy", "name": "inverse vol weighting",
               "universe": "equity_hk", **kwargs}
    envelope = json.loads(tool.execute(**payload))
    assert envelope["status"] == "ok", envelope
    return envelope["artifact"]


def test_the_three_strategy_rules_round_trip(register):
    artifact = _register(
        register,
        signal_definition="1/sigma over a 60-day window",
        entry_rules="rebalance on the last trading day of each month",
        exit_rules="no independent exit; only on failing the inclusion test",
        position_sizing="weight is the position, rounded down to a board lot",
    )

    assert artifact["entry_rules"] == "rebalance on the last trading day of each month"
    assert artifact["exit_rules"] == (
        "no independent exit; only on failing the inclusion test")
    assert artifact["position_sizing"] == (
        "weight is the position, rounded down to a board lot")


def test_decay_horizon_round_trips(register):
    artifact = _register(register, decay_horizon=60)

    assert artifact["decay_horizon"] == 60


def test_an_omitted_decay_horizon_keeps_the_dataclass_default(register):
    """Forwarding an absent value as None would overwrite the default with a
    null the rest of the stack does not expect — purged validation reads this
    as a label span and cannot use None."""
    artifact = _register(register)

    assert artifact["decay_horizon"] == 20


def test_omitted_rules_stay_none_rather_than_blank(register):
    """None means 'not stated'; an empty string would read as 'stated, and
    there are none' — which for exit_rules is a materially different claim."""
    artifact = _register(register, signal_definition="1/sigma")

    assert artifact["entry_rules"] is None
    assert artifact["exit_rules"] is None
    assert artifact["position_sizing"] is None


def test_every_strategy_field_is_reachable_from_the_tool_schema(register):
    """Mirrors test_sdm_governance_wiring's schema check: the gap this file
    closes was a schema gap, so the schema is what has to be pinned against
    it reopening."""
    props = register.parameters["properties"]

    for field in ("signal_definition", "entry_rules", "exit_rules",
                  "position_sizing", "decay_horizon"):
        assert field in props, f"{field} is on Artifact but not in the tool schema"
    assert props["decay_horizon"]["type"] == "integer"
    assert props["decay_horizon"]["minimum"] == 1, "a zero-day horizon is not a horizon"
