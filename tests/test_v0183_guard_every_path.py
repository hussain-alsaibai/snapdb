"""v0.18.3 - every path that consumes an index plan honours the guard.

v0.18.2 added a selectivity guard so a wide range query falls back to a
sequential scan. It was only wired into ``execute()``. ``count()`` and the
aggregations kept calling the raw planner, so:

  * a **selective** range (1% of rows) still decoded the whole table -
    ``count()`` was ~34x slower than ``execute()`` on the same predicate
    (37 ms vs 1.09 ms on 50K rows), and it *scanned* despite holding an
    index that named the 20 matching rows.
  * a **wide** range (62% of rows) got the plan that ``explain()`` had just
    reported as rejected, and paid ``_execute_plan``'s set materialisation
    for a plan the engine had already decided was the worse one.

The tests below assert on a **row-decode counter** rather than on wall
clock. Timing would be flaky on a shared runner; the number of rows
decoded is exactly what the guard is about, and it is deterministic:

    selective (1%)   decode 0 rows (index names them)
    wide (62%)       decode n rows (one sequential pass, no index)

Measured on the unfixed tree: every one of these decoded all 2000 rows,
for both widths.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import snapdb.core as core
from snapdb.core import SnapDB, Schema, ColumnDef
import importlib

qmod = importlib.import_module("snapdb.query")
from snapdb.query import query


class _Base(unittest.TestCase):
    n = 2000

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="snapdb-v0183-")
        self.path = Path(self._tmp) / "data.snap"
        # val is i32 and dense, so "wide" and "selective" are exact.
        schema = Schema([ColumnDef("id", "i32"), ColumnDef("val", "i32")])
        self.db = SnapDB(self.path, schema)
        self.db.batch_insert([{"id": i, "val": i} for i in range(self.n)])
        self.db.create_range_index("val")
        self._decodes = 0
        self._orig_iter = core.SnapDB.__iter__

    def tearDown(self):
        core.SnapDB.__iter__ = self._orig_iter
        try:
            self.db.close()
        except Exception:
            pass
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _measure(self, fn):
        """Run ``fn``, returning (result, rows decoded through __iter__)."""
        base = self.n * 0  # counter reset
        counted = {"n": 0}
        orig = self._orig_iter

        def iter_counting(self_):
            for item in orig(self_):
                counted["n"] += 1
                yield item

        core.SnapDB.__iter__ = iter_counting
        try:
            result = fn()
        finally:
            core.SnapDB.__iter__ = orig
        return result, counted["n"]


class TestSelectiveUsesIndex(_Base):
    """1% of the table: the index names 20 rows. Decode 20, not 2000."""

    cond = {"val": {"gte": 1980}}
    want = list(range(1980, 2000))

    def _q(self):
        return query(self.db).filter(**self.cond)

    def test_plan_is_the_index(self):
        self.assertTrue(self._q().explain().startswith("RangeIndex["),
                        self._q().explain())

    def test_count_does_not_scan(self):
        got, decoded = self._measure(lambda: self._q().count())
        self.assertEqual(got, 20)
        self.assertLess(decoded, self.n // 4,
                        f"count() decoded {decoded} rows for a 20-row result")

    def test_aggregations_do_not_scan(self):
        q = self._q()
        cases = (
            ("sum", lambda: q.sum("val"), sum(self.want)),
            ("avg", lambda: q.avg("val"), sum(self.want) / len(self.want)),
            ("min", lambda: q.min("val"), min(self.want)),
            ("max", lambda: q.max("val"), max(self.want)),
            ("count_distinct", lambda: q.count_distinct("val"), 20),
        )
        for name, call, expect in cases:
            got, decoded = self._measure(call)
            self.assertEqual(got, expect, f"{name} returned the wrong value")
            self.assertLess(decoded, self.n // 4,
                            f"{name}() decoded {decoded} rows for a 20-row result")

    def test_count_matches_len_execute(self):
        self.assertEqual(self._q().count(), len(self._q().execute()))


class TestWideRangeFallsBack(_Base):
    """62% of the table: one sequential pass, zero index materialisation."""

    cond = {"val": {"gte": 380}}
    want = list(range(380, 2000))

    def _q(self):
        return query(self.db).filter(**self.cond)

    def test_plan_is_reported_as_rejected(self):
        self.assertIn("index rejected", self._q().explain())

    def test_wide_aggregations_scan_exactly_once(self):
        q = self._q()
        cases = (
            ("sum", lambda: q.sum("val"), sum(self.want)),
            ("avg", lambda: q.avg("val"), sum(self.want) / len(self.want)),
            ("min", lambda: q.min("val"), min(self.want)),
            ("max", lambda: q.max("val"), max(self.want)),
        )
        for name, call, expect in cases:
            got, decoded = self._measure(call)
            self.assertEqual(got, expect, f"{name} returned the wrong value")
            # A rejected plan means no candidate set is built: exactly one
            # sequential pass over the table.
            self.assertEqual(decoded, self.n,
                             f"{name}() decoded {decoded} rows, expected one "
                             f"sequential pass of {self.n}")

    def test_wide_aggregations_build_no_candidate_set(self):
        """The rejected plan must not be executed anyway.

        The unfixed code scanned *and* materialised the candidate set: on a
        200K-row table ``_execute_plan`` built a 162K-element set in 706 ms,
        and each of four aggregations paid that again - ~2.8 s of pure waste
        on a plan ``explain()`` had already called rejected. Asserting on the
        decode count cannot catch it (the scan happens either way), so this
        asserts that the plan executor is never called for a rejected plan.
        """
        calls = {"n": 0}
        real = qmod._execute_plan

        def counting(db, plan):
            calls["n"] += 1
            return real(db, plan)

        qmod._execute_plan = counting
        try:
            q = self._q()
            q.sum("val")
            q.avg("val")
            q.min("val")
            q.max("val")
            q.count_distinct("val")
        finally:
            qmod._execute_plan = real
        self.assertEqual(calls["n"], 0,
                         f"a rejected plan was executed {calls['n']} times")

    def test_wide_count_matches_execute(self):
        self.assertEqual(self._q().count(), len(self._q().execute()))


class TestValuesNeverChangeWithPlan(_Base):
    """The guard may change how rows are found, never which."""

    def test_all_reductions_agree_across_selectivities(self):
        for low in (0, 340, 900, 1240, 1600, 1980, 1999):
            q = query(self.db).filter(val={"gte": low})
            want = [v for v in range(low, self.n)]
            msg = f"val >= {low}"
            self.assertEqual(q.count(), len(want), msg)
            self.assertEqual(q.sum("val"), sum(want), msg)
            self.assertEqual(q.min("val"), min(want) if want else None, msg)
            self.assertEqual(q.max("val"), max(want) if want else None, msg)
            self.assertEqual(q.count_distinct("val"), len(set(want)), msg)
            self.assertAlmostEqual(q.avg("val"), sum(want) / len(want),
                                   places=9, msg=msg)

    def test_empty_result_is_zero_not_error(self):
        q = query(self.db).filter(val={"gte": 10_000})
        self.assertEqual(q.count(), 0)
        self.assertEqual(q.sum("val"), 0)
        self.assertIsNone(q.min("val"))
        self.assertEqual(q.count_distinct("val"), 0)


class TestSingleSourceOfTruth(unittest.TestCase):
    def test_guard_helpers_exist(self):
        src = Path(qmod.__file__).read_text()
        self.assertIn("def _rejected_by_guard", src)
        self.assertIn("def _plan_or_scan", src)

    def test_no_consumer_bypasses_the_guard(self):
        """A source-level check: the bug was a bypass, not a wrong formula.

        ``_plan_query`` is the raw planner. Exactly two call sites may reach
        it: the one inside ``_plan_or_scan`` (which applies the guard) and
        the one inside ``explain()`` (which inspects the raw plan in order
        to report the rejection). A third is a new bypass.
        """
        src = Path(qmod.__file__).read_text()
        offenders = []
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if "_plan_query(" not in stripped:
                continue
            if stripped.startswith("def _plan_query"):
                continue
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue
            offenders.append((i, stripped))
        self.assertEqual(
            len(offenders), 2,
            "an index consumer reaches _plan_query directly and can bypass "
            "the selectivity guard - use _plan_or_scan instead: " + repr(offenders),
        )

    def test_threshold_is_still_the_documented_value(self):
        self.assertEqual(getattr(qmod, "_INDEX_SCAN_RATIO", None), 0.30)


if __name__ == "__main__":
    unittest.main()
