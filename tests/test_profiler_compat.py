"""
Regression tests for the PEP 669 query profiler.

Background
----------
The profiler originally did `monitoring.TOOL_AUGMENTATION`. That constant does
not exist in CPython 3.12, 3.13 or 3.14 — the only reserved tool ids are
DEBUGGER_ID (0), COVERAGE_ID (1), PROFILER_ID (2) and OPTIMIZER_ID (5), and
each is owned by the interpreter. The old code guarded on
`sys.version_info >= (3, 12)` but used a 3.14+ name, so on 3.12/3.13 the
package failed to import at all:

    AttributeError: module 'sys.monitoring' has no attribute 'TOOL_AUGMENTATION'

Because `snapdb/__init__.py` imports the profiler unconditionally, that
turned into 11 pytest collection errors — the entire suite was unrunnable
on the two newest Python versions in CI.

These tests lock in the three separate fixes:
  1. the tool id is claimed with `use_tool_id()`, and no 3.14-only name is
     referenced at module scope;
  2. the profiler degrades to a no-op instead of raising when no tool id is
     available;
  3. the PEP 669 callbacks receive a *code object*, not a frame, and
     `set_events()` is called so the callbacks actually fire.
"""

from __future__ import annotations

import sys
import unittest


class TestProfilerImports(unittest.TestCase):
    """Importing snapdb must never raise, on any supported Python."""

    def test_import_snapdb_succeeds(self):
        import snapdb  # noqa: F401
        self.assertTrue(hasattr(snapdb, "QueryProfiler"))

    def test_profiler_module_imports(self):
        from snapdb.profiler import QueryProfiler
        self.assertIsNotNone(QueryProfiler)

    def test_no_removed_tool_constant_referenced(self):
        """`TOOL_AUGMENTATION` must not be referenced anywhere.

        It is not a real attribute of `sys.monitoring` in any released
        CPython, so even a guarded reference is a landmine.
        """
        import ast
        from pathlib import Path

        import snapdb.profiler as profiler_mod

        tree = ast.parse(Path(profiler_mod.__file__).read_text(encoding="utf-8"))
        offenders = [
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "TOOL_AUGMENTATION"
        ]
        self.assertEqual(
            offenders,
            [],
            "profiler.py still accesses monitoring.TOOL_AUGMENTATION",
        )

    def test_tool_id_is_a_real_int_or_none(self):
        from snapdb.profiler import _TOOL_ID
        self.assertTrue(_TOOL_ID is None or isinstance(_TOOL_ID, int))

    def test_claimed_tool_id_is_not_reserved(self):
        """We must not squat on an interpreter-owned tool id."""
        from snapdb.profiler import _TOOL_ID

        if _TOOL_ID is None:
            self.skipTest("no monitoring tool id claimed")
        if not hasattr(sys, "monitoring"):
            self.skipTest("sys.monitoring unavailable")
        reserved = {
            sys.monitoring.DEBUGGER_ID,
            sys.monitoring.COVERAGE_ID,
            sys.monitoring.PROFILER_ID,
            sys.monitoring.OPTIMIZER_ID,
        }
        self.assertNotIn(_TOOL_ID, reserved)


class TestProfilerDegradesGracefully(unittest.TestCase):
    """The profiler must never raise, whatever the environment."""

    def test_construction_never_raises(self):
        from snapdb.profiler import QueryProfiler
        self.assertIsNotNone(QueryProfiler())

    def test_enable_disable_never_raise(self):
        from snapdb.profiler import QueryProfiler
        p = QueryProfiler()
        p.enable()
        p.enable()  # idempotent
        p.disable()
        p.disable()  # idempotent

    def test_report_never_raises(self):
        from snapdb.profiler import QueryProfiler
        p = QueryProfiler()
        self.assertIsInstance(p.report(), str)

    def test_stats_and_reset_never_raise(self):
        from snapdb.profiler import QueryProfiler
        p = QueryProfiler()
        self.assertIsInstance(p.stats(), dict)
        p.reset()
        self.assertIsInstance(p.stats(), dict)


@unittest.skipUnless(
    sys.version_info >= (3, 12) and hasattr(sys, "monitoring"),
    "sys.monitoring requires Python 3.12+",
)
class TestProfilerRecordsCalls(unittest.TestCase):
    """The profiler must actually record calls when monitoring is available."""

    def setUp(self):
        from snapdb.profiler import _has_monitoring
        if not _has_monitoring():
            self.skipTest("no free monitoring tool id in this process")
        import tempfile
        from pathlib import Path

        from snapdb.core import ColumnDef, Schema, SnapDB

        self._tmp = tempfile.mkdtemp(prefix="snapdb-prof-")
        path = Path(self._tmp) / "data.snap"
        self._db = SnapDB(
            path,
            Schema([
                ColumnDef("id", "i32"),
                ColumnDef("age", "i32"),
                ColumnDef("name", "bytes:16"),
            ]),
        )
        for i in range(5):
            self._db.insert({"id": i, "age": i, "name": f"user{i}".encode()})

    def tearDown(self):
        import shutil
        try:
            self._db.close()
        except Exception:
            pass
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_records_call_counts(self):
        from snapdb.profiler import QueryProfiler

        p = QueryProfiler()
        with p:
            self._db.get(0)
            self._db.insert({"id": 99, "age": 9, "name": b"z"})

        stats = p.stats()
        self.assertTrue(stats, "profiler recorded no statistics")
        total = sum(v["calls"] for v in stats.values())
        self.assertGreater(total, 0, "profiler recorded zero calls")
        self.assertIn("get", stats)
        self.assertIn("insert", stats)

    def test_report_contains_totals(self):
        from snapdb.profiler import QueryProfiler

        p = QueryProfiler()
        with p:
            self._db.get(1)
        report = p.report()
        self.assertIn("Query Performance Report", report)
        self.assertIn("get", report)

    def test_timings_are_non_negative(self):
        from snapdb.profiler import QueryProfiler

        p = QueryProfiler()
        with p:
            self._db.get(2)
        for name, s in p.stats().items():
            self.assertGreaterEqual(s["total_s"], 0.0, name)
            self.assertGreaterEqual(s["max_s"], s["min_s"], name)

    def test_disabled_profiler_records_nothing(self):
        from snapdb.profiler import QueryProfiler

        p = QueryProfiler()
        self._db.get(3)
        self.assertEqual(sum(v["calls"] for v in p.stats().values()), 0)


if __name__ == "__main__":
    unittest.main()
