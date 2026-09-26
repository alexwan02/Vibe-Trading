"""A backtest that declares an artifact must land in that artifact's bench_history.

``store.record_bench()`` shipped with a schema, a SQLite backend and fifteen
tests, and **no production caller** — so ``bench_history`` was always empty, so
both decay entry points (``sdm_status(decay_check)`` and ``sdm_decay_scan``)
sat behind a hardcoded ``len(bench_history) < 3`` floor that nothing could ever
fill. A bench is what a backtest *is*, so the runner is where the record
belongs.

Same class of gap as the ``state.json`` one in ``test_backtest_tool_state.py``
(#1412): a tool-driven run stayed invisible to a downstream system for want of
one written record.

The link is **declared, never inferred** — a backtest knows its ``run_dir``,
not which registered artifact it serves. These tests pin the four ways that
declaration can be absent, stale or broken, because a bookkeeping write that
fails loudly is fine and one that fails silently is not.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.backtest_tool import run_backtest

_METRICS_CSV = (
    "sharpe,annual_return,max_drawdown,calmar,trade_count\n"
    "0.7845,0.1067,-0.2123,0.5026,1427\n"
)


@dataclass
class _FakeRunResult:
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    artifacts: dict = field(default_factory=dict)


@pytest.fixture()
def store(monkeypatch):
    """A fresh SQLite store per test, so records cannot leak between them."""
    db = pathlib.Path(tempfile.mkdtemp()) / "store.db"
    monkeypatch.setenv("VIBE_TRADING_STRATEGY_STORE_DB_PATH", str(db))
    import src.strategy_store._shared as shared

    # The module-level singleton is named ``_store``; resetting it is what
    # actually isolates the test, the env var alone would not.
    monkeypatch.setattr(shared, "_store", None)
    return shared.get_store()


@pytest.fixture()
def artifact_id(store) -> str:
    """A really registered artifact.

    ``bench_results`` carries a FOREIGN KEY onto ``artifacts``, so a bench row
    cannot exist without one — a mistyped id in config.json is refused by the
    database rather than stored as an orphan.
    """
    from src.strategy_store.models import Artifact, ArtifactType

    return store.register_artifact(Artifact(
        id="", type=ArtifactType.STRATEGY,
        name="inverse vol weighting", universe="equity_hk"))


@pytest.fixture()
def run_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("VIBE_TRADING_ALLOWED_RUN_ROOTS", str(tmp_path))
    d = tmp_path / "run"
    (d / "code").mkdir(parents=True)
    (d / "artifacts").mkdir(parents=True)
    (d / "code" / "signal_engine.py").write_text("", encoding="utf-8")
    (d / "artifacts" / "metrics.csv").write_text(_METRICS_CSV, encoding="utf-8")
    return d


def _write_config(run_dir: Path, **extra) -> None:
    cfg = {"source": "yfinance", "start_date": "2021-01-04",
           "end_date": "2026-09-25", **extra}
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _run(run_dir: Path, *, success: bool = True, exit_code: int = 0) -> dict:
    with patch("src.tools.backtest_tool.emit_progress"), \
         patch("src.tools.backtest_tool.Runner") as runner_cls:
        runner_cls.return_value.execute.return_value = _FakeRunResult(
            success=success, exit_code=exit_code)
        return json.loads(run_backtest(str(run_dir)))


# -- the happy path it exists for ------------------------------------------

def test_a_declared_artifact_gets_its_bench_recorded(run_dir, store, artifact_id):
    _write_config(run_dir, artifact_id=artifact_id)

    envelope = _run(run_dir)

    assert envelope["status"] == "ok", envelope
    assert envelope["bench_record"] == {
        "status": "recorded", "artifact_id": artifact_id, "bench_type": "initial"}
    (row,) = store.get_bench_history(artifact_id)
    assert row.sharpe == pytest.approx(0.7845)
    assert row.annual_return == pytest.approx(0.1067)
    assert row.max_drawdown == pytest.approx(-0.2123)
    assert row.calmar == pytest.approx(0.5026)
    assert row.test_start == "2021-01-04" and row.test_end == "2026-09-25"
    assert row.run_dir == str(run_dir)


def test_the_runner_does_not_grade_its_own_run(run_dir, store, artifact_id):
    """``category`` stays None: alive/dead is the EVALUATE phase's verdict
    against its own thresholds. A runner that graded its own output would make
    the measurement and the verdict inseparable."""
    _write_config(run_dir, artifact_id=artifact_id)

    _run(run_dir)

    (row,) = store.get_bench_history(artifact_id)
    assert row.category is None
    # A strategy backtest produces no cross-sectional IC — absent, not zero.
    assert (row.ic_mean, row.ic_std, row.ir, row.ic_positive_ratio) == (
        None, None, None, None)


def test_a_second_distinct_run_is_periodic_not_initial(run_dir, store, tmp_path, artifact_id):
    """The decay baseline is built from ``initial`` entries; if every run
    claimed to be initial, the baseline would drift with the runs it is
    supposed to be measured against."""
    _write_config(run_dir, artifact_id=artifact_id)
    _run(run_dir)

    second = tmp_path / "run2"
    (second / "code").mkdir(parents=True)
    (second / "artifacts").mkdir(parents=True)
    (second / "code" / "signal_engine.py").write_text("", encoding="utf-8")
    (second / "artifacts" / "metrics.csv").write_text(_METRICS_CSV, encoding="utf-8")
    _write_config(second, artifact_id=artifact_id)

    envelope = _run(second)

    assert envelope["bench_record"]["bench_type"] == "periodic"
    assert len(store.get_bench_history(artifact_id)) == 2


# -- the four ways it must NOT write ---------------------------------------

def test_no_artifact_id_writes_nothing(run_dir, store):
    """The normal path. Most runs are exploratory and belong to no artifact —
    skipping them is correct behaviour, not a degraded one."""
    _write_config(run_dir)

    envelope = _run(run_dir)

    assert envelope["status"] == "ok", envelope
    assert envelope["bench_record"] == {
        "status": "skipped", "reason": "no artifact_id in config"}
    assert store.get_bench_history("any-artifact") == []


def test_a_failed_backtest_records_no_bench(run_dir, store, artifact_id):
    """A failed run is not a bench result. Recording one would put a number
    into the decay baseline that no successful run ever produced."""
    _write_config(run_dir, artifact_id=artifact_id)

    envelope = _run(run_dir, success=False, exit_code=3)

    assert envelope["status"] == "error"
    assert envelope["bench_record"] == {
        "status": "skipped", "reason": "backtest did not succeed"}
    assert store.get_bench_history(artifact_id) == []


def test_rerunning_the_same_run_dir_does_not_stack_duplicates(run_dir, store, artifact_id):
    """Duplicates would silently inflate the baseline: the same run counted
    twice looks like two independent confirmations."""
    _write_config(run_dir, artifact_id=artifact_id)
    _run(run_dir)

    envelope = _run(run_dir)

    assert envelope["bench_record"] == {
        "status": "skipped", "reason": "run_dir already recorded",
        "artifact_id": artifact_id}
    assert len(store.get_bench_history(artifact_id)) == 1


def test_a_store_failure_is_reported_but_never_fails_the_backtest(run_dir, store, artifact_id):
    """The backtest already succeeded. Losing a bookkeeping write must not
    retroactively turn a good run into a failed one — but it must be visible,
    because a silently skipped record is indistinguishable from a run that
    declared no artifact."""
    _write_config(run_dir, artifact_id=artifact_id)

    with patch("src.strategy_store.sqlite_store.SqliteStrategyStore.record_bench",
               side_effect=RuntimeError("disk on fire")):
        envelope = _run(run_dir)

    assert envelope["status"] == "ok", "the run itself succeeded"
    assert envelope["bench_record"]["status"] == "failed"
    assert "disk on fire" in envelope["bench_record"]["reason"]


def test_missing_metrics_csv_is_skipped_not_recorded_as_blank(run_dir, store, artifact_id):
    """A row of all-None metrics would pass the ``>= 3 entries`` floor while
    carrying nothing to measure decay with."""
    _write_config(run_dir, artifact_id=artifact_id)
    (run_dir / "artifacts" / "metrics.csv").unlink()

    envelope = _run(run_dir)

    assert envelope["bench_record"]["status"] == "skipped"
    assert "metrics.csv" in envelope["bench_record"]["reason"]
    assert store.get_bench_history(artifact_id) == []


def test_an_unregistered_artifact_id_is_refused_not_orphaned(run_dir, store):
    """``bench_results`` has a FOREIGN KEY onto ``artifacts``. A config naming
    an id that was never registered — a typo, or a copy-paste from another
    machine's store — must surface as a reported failure, not as a row nobody
    can reach from the artifact side."""
    _write_config(run_dir, artifact_id="never-registered")

    envelope = _run(run_dir)

    assert envelope["status"] == "ok", "the backtest itself is unaffected"
    assert envelope["bench_record"]["status"] == "failed"
    assert "FOREIGN KEY" in envelope["bench_record"]["reason"]
    assert store.get_bench_history("never-registered") == []
