"""Backtest execution tool: validates config.json + signal_engine.py and runs the built-in engine."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from backtest.loaders.registry import VALID_SOURCES
from src.agent.progress import emit_progress
from src.agent.tools import BaseTool
from src.config.accessor import get_env_config
from src.core.runner import Runner
from src.core.state import RunStateStore
from src.tools.path_utils import safe_run_dir


def _backtest_timeout_seconds() -> float | None:
    """Return the configured backtest subprocess timeout.

    The backtest is a write-style tool, so the agent-loop timeout does not
    cancel it.  The subprocess still needs its own bound, which follows the
    same ``VIBE_TRADING_TOOL_TIMEOUT_SECONDS`` setting used by the loop.  A
    non-positive value keeps the historical "disabled" semantics.

    Returns:
        Positive timeout in seconds, or ``None`` to disable the bound.
    """
    configured = float(get_env_config().agent_tuning.vibe_trading_tool_timeout_seconds)
    return configured if configured > 0 else None


def _timeout_output(value: Any) -> str:
    """Normalize ``TimeoutExpired`` output for persistence and JSON."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _persist_timeout_output(run_path: Path, exc: subprocess.TimeoutExpired) -> dict[str, str]:
    """Persist partial subprocess output after a timeout.

    ``subprocess.run`` exposes captured output on ``TimeoutExpired`` when pipes
    are used.  Preserve it before returning so a timed-out run remains
    diagnosable instead of appearing to have produced nothing.
    """
    output = {
        "stdout": _timeout_output(getattr(exc, "stdout", None)),
        "stderr": _timeout_output(getattr(exc, "stderr", None)),
    }
    log_dir = run_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "runner_stdout.txt").write_text(output["stdout"], encoding="utf-8")
    (log_dir / "runner_stderr.txt").write_text(output["stderr"], encoding="utf-8")
    return output


#: Columns copied verbatim from ``artifacts/metrics.csv`` into ``BenchResult``.
#: Strategy metrics only — a strategy backtest produces no cross-sectional IC,
#: so ``ic_mean``/``ic_std``/``ir``/``ic_positive_ratio``/``t_stat`` stay None
#: instead of being filled with a stand-in.
_BENCH_METRIC_COLUMNS = ("sharpe", "annual_return", "max_drawdown", "calmar")


def _read_metrics_row(run_path: Path) -> dict[str, float] | None:
    """Return the single data row of ``artifacts/metrics.csv``, floats only."""
    import csv

    metrics_path = run_path / "artifacts" / "metrics.csv"
    if not metrics_path.is_file():
        return None
    with metrics_path.open(encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh), None)
    if not row:
        return None
    out: dict[str, float] = {}
    for col in _BENCH_METRIC_COLUMNS:
        raw = (row.get(col) or "").strip()
        if not raw:
            continue
        try:
            out[col] = float(raw)
        except ValueError:
            continue        # a non-numeric cell is a missing metric, not a crash
    return out


def _record_bench_if_declared(
    config: dict, run_path: Path, *, success: bool
) -> dict[str, str]:
    """Append this run to the artifact's ``bench_history`` when it declares one.

    A backtest knows its ``run_dir``; it does not know which registered
    artifact it belongs to. The link is therefore **declared, never inferred**:
    ``config.json`` may carry an ``artifact_id``, and only then is the run
    recorded. Most exploratory runs belong to no artifact and are skipped —
    that is the normal path, not a failure.

    Why this exists: ``store.record_bench()`` had no production caller at all,
    so ``bench_history`` stayed empty, so both decay entry points
    (``sdm_status(decay_check)`` and ``sdm_decay_scan``) hit their hardcoded
    ``len(bench_history) < 3`` floor forever. A bench is what a backtest *is*,
    so this is where the record belongs. Same class of gap as the ``state.json``
    one fixed just below (#1412): a tool-driven run stayed invisible to a
    downstream system for want of one written record.

    ``category`` is deliberately left ``None``. Grading a run ``alive``/``dead``
    belongs to the EVALUATE phase and its own thresholds; a runner that graded
    its own output would make the measurement and the verdict inseparable.

    Never raises: the backtest already succeeded, and a bookkeeping failure must
    not turn that into a failed run. Failures are reported in the return value
    rather than swallowed.
    """
    if not success:
        return {"status": "skipped", "reason": "backtest did not succeed"}
    artifact_id = str(config.get("artifact_id") or "").strip()
    if not artifact_id:
        return {"status": "skipped", "reason": "no artifact_id in config"}

    try:
        from src.strategy_store._shared import get_store
        from src.strategy_store.models import BenchResult

        store = get_store()
        history = list(store.get_bench_history(artifact_id, limit=1000))
        # Idempotent by run_dir: re-running the same directory must not stack
        # duplicate rows, which would silently inflate the decay baseline.
        if any(r.run_dir == str(run_path) for r in history):
            return {"status": "skipped", "reason": "run_dir already recorded",
                    "artifact_id": artifact_id}

        metrics = _read_metrics_row(run_path)
        if metrics is None:
            return {"status": "skipped",
                    "reason": "artifacts/metrics.csv missing or empty",
                    "artifact_id": artifact_id}

        bench_type = "initial" if not history else "periodic"
        store.record_bench(BenchResult(
            artifact_id=artifact_id,
            bench_type=bench_type,
            test_start=str(config.get("start_date") or "") or None,
            test_end=str(config.get("end_date") or "") or None,
            run_dir=str(run_path),
            **metrics,
        ))
        return {"status": "recorded", "artifact_id": artifact_id,
                "bench_type": bench_type}
    except Exception as exc:                    # noqa: BLE001 — see docstring
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def run_backtest(run_dir: str) -> str:
    """Run backtest: validate config.json + signal_engine.py, invoke built-in engine.

    Args:
        run_dir: Path to the run directory.

    Returns:
        JSON-formatted execution result.
    """
    emit_progress("validate", message="validating run_dir and config")
    try:
        run_path = safe_run_dir(run_dir)
    except ValueError as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)

    config_path = run_path / "config.json"
    if not config_path.exists():
        return json.dumps(
            {
                "status": "error",
                "error": f"config.json not found in {run_path}",
                "hint": "config.json belongs at the root of run_dir.",
            },
            ensure_ascii=False,
        )

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"config.json parse error: {e}"}, ensure_ascii=False)

    if "source" not in config:
        return json.dumps({"status": "error", "error": "config.json missing 'source' field (tushare/okx/yfinance)"}, ensure_ascii=False)

    if config["source"] not in VALID_SOURCES:
        return json.dumps({"status": "error", "error": f"source must be one of {VALID_SOURCES}, got: {config['source']}"}, ensure_ascii=False)

    signal_path = run_path / "code" / "signal_engine.py"
    if not signal_path.exists():
        return json.dumps(
            {
                "status": "error",
                "error": f"code/signal_engine.py not found in {run_path}",
                "hint": "signal_engine.py belongs in code/ inside run_dir.",
            },
            ensure_ascii=False,
        )

    agent_root = Path(__file__).resolve().parents[2]
    entry_script = agent_root / "backtest" / "runner.py"

    source = config.get("source", "?")
    emit_progress(
        "simulate",
        message=f"running backtest engine (source={source})",
    )
    runner = Runner(timeout=_backtest_timeout_seconds())
    try:
        result = runner.execute(
            entry_script,
            run_path,
            cwd=agent_root,
            cli_args=[str(run_path)],
        )
    except subprocess.TimeoutExpired as exc:
        # The lifecycle block below is unreachable on a timeout, so record the
        # failure here — otherwise the run is indistinguishable from never-run
        # (the evidence gate fail-closes either way, but the reason is lost).
        timeout_output = _persist_timeout_output(run_path, exc)
        timeout_label = f"{runner.timeout}s" if runner.timeout is not None else "the configured limit"
        reason = f"backtest engine timed out after {timeout_label}"
        RunStateStore().mark_failure(run_path, reason)
        response = {
            "status": "error",
            "error": reason,
            "run_dir": run_dir,
        }
        if timeout_output["stdout"]:
            response["stdout"] = timeout_output["stdout"][-2000:]
        if timeout_output["stderr"]:
            response["stderr"] = timeout_output["stderr"][-2000:]
        return json.dumps(response, ensure_ascii=False)

    # Record lifecycle status so tool-driven runs are ingestible by the
    # evidence pipeline: refresh_strategy_evidence fail-closes without
    # state.json, which previously only the runtime loop wrote (#1412).
    # Same contract as the runtime — success, or failure with a reason.
    state_store = RunStateStore()
    if result.success:
        state_store.mark_success(run_path)
    else:
        state_store.mark_failure(run_path, f"backtest engine exited with code {result.exit_code}")

    bench_record = _record_bench_if_declared(config, run_path, success=result.success)

    emit_progress("finalize", message="collecting artifacts")
    artifacts_found = {name: str(path) for name, path in result.artifacts.items()}
    return json.dumps({
        "status": "ok" if result.success else "error",
        "exit_code": result.exit_code,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
        "artifacts": artifacts_found,
        "run_dir": run_dir,
        "bench_record": bench_record,
    }, ensure_ascii=False)


class BacktestTool(BaseTool):
    """Backtest execution tool."""

    name = "backtest"
    description = "Run backtest: validate config.json + signal_engine.py, invoke built-in engine."
    parameters = {
        "type": "object",
        "properties": {
            "run_dir": {"type": "string", "description": "Path to the run directory"},
        },
        "required": ["run_dir"],
    }
    repeatable = True
    is_readonly = False

    def execute(self, **kwargs) -> str:
        """Execute backtest."""
        return run_backtest(kwargs["run_dir"])
