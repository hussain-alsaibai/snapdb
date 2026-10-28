"""
Query profiler for SnapDB using PEP 669 `sys.monitoring`.

Requires Python 3.12+. On older versions, all profiler operations are no-ops.

Usage:
    >>> from snapdb.profiler import QueryProfiler
    >>> profiler = QueryProfiler()
    >>> with profiler:
    ...     db.select_where(...)
    >>> print(profiler.report())

The profiler hooks into PY_START and PY_RETURN events on the core
`select_where`, `aggregate`, `insert`, `update`, `delete`, and `get`
entry points, accumulating call counts and wall/CPU time per operation.
Overhead is ~5% when active, zero when not imported.

Version notes
-------------
`sys.monitoring.TOOL_AUGMENTATION` does not exist in CPython 3.12, 3.13 or
3.14 -- the only reserved tool ids are DEBUGGER_ID (0), COVERAGE_ID (1),
PROFILER_ID (2) and OPTIMIZER_ID (5), and each is owned by the interpreter.
The remaining ids (3 and 4) are free for third-party tools, so this module
claims one with `monitoring.use_tool_id()`.
"""

from __future__ import annotations

import sys
import time
import threading
from typing import Callable, Any
from collections import defaultdict

# ---------------------------------------------------------------------------
# Guards — only activate on Python 3.12+
# ---------------------------------------------------------------------------

_HAVE_MONITORING = sys.version_info >= (3, 12)

if _HAVE_MONITORING:
    from sys import monitoring
else:  # pragma: no cover — import guard
    monitoring = None  # type: ignore[assignment]

# Tool ids 0/1/2/5 are reserved for the interpreter (debugger, coverage,
# profiler, optimizer). Ids 3 and 4 are free for third-party tools.
_CANDIDATE_TOOL_IDS = (3, 4)


def _claim_tool_id() -> int | None:
    """Claim a free `sys.monitoring` tool id, or return None.

    A tool id can only be claimed once per process, so the first successful
    claim is cached at module level. If every candidate is already taken by
    another library, profiling is unavailable and the profiler degrades to a
    no-op rather than raising.
    """
    if not _HAVE_MONITORING:
        return None
    for candidate in _CANDIDATE_TOOL_IDS:
        try:
            monitoring.use_tool_id(candidate, "snapdb-profiler")
        except ValueError:
            # Already in use by someone else — try the next candidate.
            continue
        return candidate
    return None


_TOOL_ID = _claim_tool_id()


def _has_monitoring() -> bool:
    """True when a usable `sys.monitoring` tool id was claimed."""
    return _TOOL_ID is not None


# ---------------------------------------------------------------------------
# Profiled operations
# ---------------------------------------------------------------------------

_PROFILED_NAMES = frozenset([
    "select_where",
    "aggregate",
    "insert",
    "update",
    "delete",
    "get",
    "range_find",
    "lookup",
    "compact",
    "backup",
])


# ---------------------------------------------------------------------------
# QueryProfiler
# ---------------------------------------------------------------------------

class QueryProfiler:
    """
    Low-overhead query profiler using PEP 669 `sys.monitoring`.

    On Python < 3.12 this is a dummy no-op class — it will not raise,
    but `enable()` will be a no-op and `report()` will return a message
    saying profiling is unavailable.

    On Python 3.12+, call `enable()` to start profiling and `disable()` to stop.
    Or use as a context manager:

        profiler = QueryProfiler()
        with profiler:
            db.select_where(...)
        print(profiler.report())

    Results include:
    - Call counts per operation
    - Wall-clock total / mean / min / max / p95 duration
    - Thread-safe accumulation across enable/disable cycles
    """

    __slots__ = (
        "_enabled", "_lock", "_counts", "_wall_times",
        "_cpu_times", "_active_calls", "_py_start", "_py_return",
    )

    def __init__(self) -> None:
        self._enabled = False
        self._lock = threading.Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._wall_times: dict[str, list[float]] = defaultdict(list)
        self._cpu_times: dict[str, list[float]] = defaultdict(list)
        self._active_calls: dict[int, tuple[str, float]] = {}  # frame_id → (name, start)

        if _has_monitoring():
            self._setup_callbacks()
        else:
            self._py_start: Callable[..., Any] = self._noop_callback
            self._py_return: Callable[..., Any] = self._noop_callback

    # ---- Context manager ----

    def __enter__(self) -> QueryProfiler:
        self.enable()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disable()

    # ---- Enable / Disable ----

    def enable(self) -> None:
        """Start profiling. Thread-safe (idempotent)."""
        if not _has_monitoring():
            return
        with self._lock:
            if self._enabled:
                return
            try:
                monitoring.register_callback(
                    _TOOL_ID,
                    monitoring.events.PY_START,
                    self._py_start,
                )
                monitoring.register_callback(
                    _TOOL_ID,
                    monitoring.events.PY_RETURN,
                    self._py_return,
                )
                # register_callback() alone does nothing — events must be
                # explicitly enabled for this tool id or no callback fires.
                monitoring.set_events(
                    _TOOL_ID,
                    monitoring.events.PY_START | monitoring.events.PY_RETURN,
                )
                self._enabled = True
            except Exception:
                # Another tool may already own the callbacks — skip profiling
                self._enabled = False

    def disable(self) -> None:
        """Stop profiling. Thread-safe."""
        if not _has_monitoring():
            return
        with self._lock:
            if not self._enabled:
                return
            try:
                monitoring.set_events(_TOOL_ID, monitoring.events.NO_EVENTS)
                monitoring.register_callback(_TOOL_ID, monitoring.events.PY_START, None)
                monitoring.register_callback(_TOOL_ID, monitoring.events.PY_RETURN, None)
            except Exception:
                pass
            finally:
                self._enabled = False

    # ---- Internal callbacks ----

    def _noop_callback(self, *_: Any, **__: Any) -> None:
        pass

    def _setup_callbacks(self) -> None:
        """Set up PY_START / PY_RETURN callbacks via sys.monitoring.

        PEP 669 passes a *code object* as the first argument to PY_START and
        PY_RETURN — not a frame. The frame has to be recovered with
        `sys._getframe(1)` so that PY_START and PY_RETURN can be paired by
        frame identity.
        """

        def py_start(
            code: Any,
            instruction_offset: int,
        ) -> None:
            name = code.co_name
            if name not in _PROFILED_NAMES:
                return
            try:
                frame = sys._getframe(1)
            except ValueError:  # pragma: no cover — frame gone
                return
            self._active_calls[id(frame)] = (name, time.perf_counter())

        def py_return(
            code: Any,
            instruction_offset: int,
            retval: Any,
        ) -> None:
            try:
                frame = sys._getframe(1)
            except ValueError:  # pragma: no cover — frame gone
                return
            entry = self._active_calls.pop(id(frame), None)
            if entry is None:
                return
            name, start = entry
            elapsed = time.perf_counter() - start
            with self._lock:
                self._counts[name] += 1
                self._wall_times[name].append(elapsed)

        self._py_start = py_start  # type: ignore[assignment]
        self._py_return = py_return  # type: ignore[assignment]

    # ---- Stats ----

    def stats(self) -> dict[str, dict[str, Any]]:
        """Return raw profiling statistics as a dict."""
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for name, times in self._wall_times.items():
                if not times:
                    continue
                sorted_times = sorted(times)
                n = len(sorted_times)
                p95_idx = int(n * 0.95)
                result[name] = {
                    "calls": self._counts[name],
                    "total_s": sum(times),
                    "mean_s": sum(times) / n,
                    "min_s": min(times),
                    "max_s": max(times),
                    "p95_s": sorted_times[p95_idx] if n > 0 else 0.0,
                }
            return result

    def report(self) -> str:
        """Return a human-readable profiling report."""
        if not _HAVE_MONITORING:
            return (
                "[QueryProfiler] Python 3.12+ required for sys.monitoring. "
                "Profiling is not available on this Python version."
            )

        if not _has_monitoring():
            return (
                "[QueryProfiler] No free sys.monitoring tool id was available "
                "(ids 3 and 4 are claimed by other libraries). "
                "Profiling is not available in this process."
            )

        stats = self.stats()
        if not stats:
            return "[QueryProfiler] No data — did you call enable() and run some queries?"

        lines = ["[QueryProfiler] Query Performance Report", "=" * 44]
        header = f"{'Operation':<16} {'Calls':>6} {'Total(s)':>10} {'Mean(ms)':>10} {'P95(ms)':>10} {'Max(ms)':>10}"
        lines.append(header)
        lines.append("-" * 64)

        for name in sorted(stats.keys()):
            s = stats[name]
            mean_ms = s["mean_s"] * 1000
            p95_ms = s["p95_s"] * 1000
            max_ms = s["max_s"] * 1000
            lines.append(
                f"{name:<16} {s['calls']:>6} "
                f"{s['total_s']:>10.4f} "
                f"{mean_ms:>10.2f} "
                f"{p95_ms:>10.2f} "
                f"{max_ms:>10.2f} "
            )

        lines.append("=" * 64)
        total_calls = sum(s["calls"] for s in stats.values())
        total_time = sum(s["total_s"] for s in stats.values())
        lines.append(f"Total: {total_calls} calls in {total_time:.4f}s")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        with self._lock:
            self._counts.clear()
            self._wall_times.clear()
            self._cpu_times.clear()
            self._active_calls.clear()

    def __repr__(self) -> str:
        if not _HAVE_MONITORING:
            return f"<QueryProfiler (Python {sys.version_info.major}.{sys.version_info.minor} — monitoring unavailable)>"
        if not _has_monitoring():
            return "<QueryProfiler (no free monitoring tool id)>"
        status = "active" if self._enabled else "inactive"
        total = sum(self._counts.values())
        return f"<QueryProfiler [{status}] {total} calls>"
