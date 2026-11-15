"""v0.18.2 - the index is only used when it is the cheaper plan.

Covers:
  * an index plan above the selectivity threshold is dropped, not used
  * the rows returned are identical with and without an index, at every
    selectivity (the guard must change *how* rows are found, never which)
  * explain() reports what execute() will actually do
  * targeted single-column hash lookups remain on the index
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from snapdb.core import SnapDB, Schema, ColumnDef
import snapdb.query as qmod
from snapdb.query import query, _plan_query


def _estimate_rows(idx, mode, **kw):
    """Resolve the helper lazily so that on a tree without it the *tests*
    fail on behaviour instead of the whole file failing to import - a
    collection error is not proof that a behaviour regressed."""
    fn = getattr(qmod, "_estimate_rows", None)
    if fn is None:
        raise AssertionError("snapdb.query._estimate_rows is missing")
    return fn(idx, mode, **kw)


class _Base(unittest.TestCase):
    n = 400

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="snapdb-v0182-")
        self.path = Path(self._tmp) / "data.snap"
        schema = Schema([ColumnDef("id", "i32"), ColumnDef("val", "f32")])
        self.db = SnapDB(self.path, schema)
        for i in range(self.n):
            self.db.insert({"id": i, "val": float(i)})

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _ids(self, lo, hi):
        return {i for i, _ in query(self.db).between("val", lo, hi).execute()}

    def _expected(self, lo, hi):
        return {i for i, row in self.db
                if row["val"] is not None and lo <= row["val"] <= hi}


class TestEstimate(_Base):
    def test_estimate_matches_scan(self):
        self.db.create_range_index("val")
        idx = self.db._range_indexes["val"]
        for lo, hi in ((0.0, 10.0), (0.0, 200.0), (100.0, 300.0)):
            est = _estimate_rows(idx, "between", low=lo, high=hi)
            self.assertEqual(est, len(self._expected(lo, hi)))

    def test_estimate_honours_exclusive_bounds(self):
        self.db.create_range_index("val")
        idx = self.db._range_indexes["val"]
        self.assertEqual(
            _estimate_rows(idx, "between", low=10.0, high=20.0, include_low=False),
            10,  # 11..20
        )
        self.assertEqual(
            _estimate_rows(idx, "between", low=10.0, high=20.0, include_high=False),
            10,  # 10..19
        )

    def test_estimate_is_none_without_items(self):
        class _Fake:
            pass
        self.assertIsNone(_estimate_rows(_Fake(), "between", low=0, high=1))


class TestGuard(_Base):
    def test_rows_identical_with_and_without_index(self):
        """The guard changes the plan, never the result set."""
        bounds = ((0.0, 5.0), (0.0, 200.0), (0.0, 399.0), (50.0, 300.0))
        self.db.create_range_index("val")
        for lo, hi in bounds:
            self.assertEqual(self._ids(lo, hi), self._expected(lo, hi),
                             f"index result differed for {lo}..{hi}")

    def test_wide_range_is_not_using_the_index(self):
        self.db.create_range_index("val")
        q = query(self.db).between("val", 0.0, float(self.n))  # 100% of rows
        self.assertIn("index rejected", q.explain())
        self.assertEqual(len(self._ids(0.0, float(self.n))), self.n)

    def test_selective_range_still_uses_the_index(self):
        self.db.create_range_index("val")
        q = query(self.db).between("val", 0.0, 4.0)  # 5 rows of 400
        self.assertTrue(q.explain().startswith("RangeIndex["),
                        q.explain())

    def test_threshold_is_the_documented_value(self):
        # A guard with an undocumented cutoff is a magic number. If this
        # fails, the constant changed - update the README/changelog too.
        self.assertEqual(getattr(qmod, "_INDEX_SCAN_RATIO", None), 0.30)


class TestHashUnaffected(_Base):
    def test_hash_lookup_still_planned(self):
        self.db.create_index("id")
        plan, _ = _plan_query(self.db, {"id": 7})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "hash")
        self.assertEqual(query(self.db).filter(id=7).explain(), "HashIndex[id = 7]")


if __name__ == "__main__":
    unittest.main()
