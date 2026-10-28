"""
SnapDB v0.18.0 - Index-aware query planning tests

Covers:
  * HashIndex equality planning
  * RangeIndex single-bound (gt/gte/lt/lte) planning
  * RangeIndex between planning (both bounds)
  * explain() output formats
  * count() with index candidates
  * Aggregations (sum/min/max/avg/count_distinct) honour index plan
  * Fallback to scan when no index available
  * Fallback to scan when only ne operator is used
  * Empty candidate set short-circuits
  * between() shortcut wires through correctly
  * where_fn() disqualifies index planning
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from snapdb.core import SnapDB, Schema, ColumnDef
from snapdb.query import query, _plan_query


def _build_db(n: int = 100):
    """Build a simple ``users`` table with n rows.

    Columns:
      id (int)   - primary key, no two rows share a value
      age (int)  - 0..(n-1) for easy reasoning
      name (str) - synthetic

    Returns (db, tmpdir) so the test can clean up via ``addCleanup``.
    """
    tmp = tempfile.mkdtemp(prefix="snapdb-v018-")
    path = Path(tmp) / "data.snap"
    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("age", "i32"),
        ColumnDef("name", "bytes:16"),
    ])
    db = SnapDB(path, schema)
    for i in range(n):
        db.insert({"id": i, "age": i % 50, "name": f"user{i}".encode()})
    return db, tmp


class _SnapTestCase(unittest.TestCase):
    """Base class that auto-closes the db and removes the tmp dir."""

    def setUp(self):
        self._db, self._tmpdir = _build_db()

    def tearDown(self):
        try:
            self._db.close()
        except Exception:
            pass
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TestHashIndexPlanning(_SnapTestCase):
    """HashIndex equality lookups are served from the index."""

    def test_eq_plan_uses_hash_index(self):
        self._db.create_index("id")
        plan, _ = _plan_query(self._db, {"id": 42})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "hash")
        self.assertEqual(plan.column, "id")
        self.assertEqual(plan.value, 42)

    def test_filter_eq_returns_correct_row(self):
        self._db.create_index("id")
        rows = [r for _, r in query(self._db).filter(id=42).execute()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], 42)

    def test_filter_eq_no_index_falls_back_to_scan(self):
        # No create_index call.
        rows = [r for _, r in query(self._db).filter(id=42).execute()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], 42)

    def test_eq_count_uses_index(self):
        # age appears twice (i and i+50) -> 2 matches.
        self._db.create_index("age")
        c = query(self._db).filter(age=10).count()
        self.assertEqual(c, 2)


class TestRangeIndexPlanning(_SnapTestCase):
    """RangeIndex serves single-bound and between-style specs."""

    def test_gt_plan_uses_range_index(self):
        self._db.create_range_index("age")
        plan, _ = _plan_query(self._db, {"age": {"gt": 25}})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "range")
        self.assertEqual(plan.side, "low")
        self.assertEqual(plan.bound, 25)
        self.assertFalse(plan.include)  # gt excludes 25

    def test_gte_plan_uses_range_index(self):
        self._db.create_range_index("age")
        plan, _ = _plan_query(self._db, {"age": {"gte": 25}})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "range")
        self.assertTrue(plan.include)

    def test_lt_lte_plan(self):
        self._db.create_range_index("age")
        plan, _ = _plan_query(self._db, {"age": {"lt": 30}})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.side, "high")
        self.assertEqual(plan.bound, 30)
        self.assertFalse(plan.include)

    def test_between_plan_two_bounds(self):
        self._db.create_range_index("age")
        plan, _ = _plan_query(self._db, {"age": {"gte": 25, "lt": 65}})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "between")
        self.assertEqual(plan.low, 25)
        self.assertEqual(plan.high, 65)
        self.assertTrue(plan.inc_low)
        self.assertFalse(plan.inc_high)

    def test_between_shortcut(self):
        self._db.create_range_index("age")
        rows = [r for _, r in query(self._db).between("age", 25, 35).execute()]
        ages = sorted(r["age"] for r in rows)
        # age 25..35 inclusive (lte=35). each appears twice -> 22 rows.
        self.assertEqual(len(ages), 22)
        self.assertEqual(min(ages), 25)
        self.assertEqual(max(ages), 35)

    def test_between_exclusive_low(self):
        self._db.create_range_index("age")
        # Default include_high=True: between(25, 35, False, True) = (25, 35]
        rows = [r for _, r in query(self._db).between("age", 25, 35, include_low=False).execute()]
        ages = sorted({r["age"] for r in rows})
        self.assertEqual(ages, list(range(26, 36)))

    def test_range_filter_returns_correct_set(self):
        self._db.create_range_index("age")
        rows = [r for _, r in query(self._db).filter(age={"gte": 40, "lt": 50}).execute()]
        ages = sorted({r["age"] for r in rows})
        self.assertEqual(ages, list(range(40, 50)))

    def test_no_range_index_falls_back_to_scan(self):
        rows = [r for _, r in query(self._db).filter(age={"gte": 40, "lt": 50}).execute()]
        ages = sorted({r["age"] for r in rows})
        self.assertEqual(ages, list(range(40, 50)))

    def test_ne_disqualifies_index_planning(self):
        self._db.create_range_index("age")
        plan, _ = _plan_query(self._db, {"age": {"ne": 25}})
        # ne is not in _RANGE_OPS, so the planner should refuse.
        self.assertIsNone(plan)

    def test_empty_candidate_short_circuits(self):
        self._db.create_range_index("age")
        rows = query(self._db).filter(age={"gt": 1000}).execute()
        self.assertEqual(rows, [])


class TestExplain(_SnapTestCase):
    """Query.explain() returns a one-line plan description."""

    def test_explain_no_filter_is_scan_full(self):
        self.assertEqual(query(self._db).explain(), "Scan(full)")

    def test_explain_hash_index(self):
        self._db.create_index("id")
        self.assertEqual(query(self._db).filter(id=1).explain(), "HashIndex[id = 1]")

    def test_explain_range_index_lt(self):
        self._db.create_range_index("age")
        self.assertEqual(
            query(self._db).filter(age={"lte": 30}).explain(),
            "RangeIndex[age lte= 30]",
        )

    def test_explain_range_index_gt_exclusive(self):
        self._db.create_range_index("age")
        self.assertEqual(
            query(self._db).filter(age={"gt": 25}).explain(),
            "RangeIndex[age gt 25]",
        )

    def test_explain_between(self):
        self._db.create_range_index("age")
        self.assertEqual(
            query(self._db).filter(age={"gte": 25, "lt": 65}).explain(),
            "RangeIndex[age [25, 65)]",
        )

    def test_explain_scan_when_no_index(self):
        self.assertEqual(
            query(self._db).filter(age={"gte": 25, "lt": 65}).explain(),
            "Scan(filter=['age'])",
        )


class TestAggregationsIndexAware(_SnapTestCase):
    """sum/min/max/avg/count_distinct honour index candidates."""

    def test_sum_with_index(self):
        self._db.create_range_index("age")
        s = query(self._db).filter(age={"gte": 10, "lt": 20}).sum("age")
        # age 10..19 (each appears twice: i=10..19 and i=60..69).
        expected = 2 * sum(range(10, 20))
        self.assertEqual(s, expected)

    def test_count_distinct_with_index(self):
        self._db.create_range_index("age")
        c = query(self._db).filter(age={"gte": 10, "lt": 20}).count_distinct("age")
        self.assertEqual(c, 10)  # distinct ages 10..19

    def test_min_with_index(self):
        self._db.create_range_index("age")
        m = query(self._db).filter(age={"gte": 10, "lt": 20}).min("age")
        self.assertEqual(m, 10)

    def test_max_with_index(self):
        self._db.create_range_index("age")
        m = query(self._db).filter(age={"gte": 10, "lt": 20}).max("age")
        self.assertEqual(m, 19)

    def test_avg_with_index(self):
        self._db.create_range_index("age")
        a = query(self._db).filter(age={"gte": 10, "lt": 20}).avg("age")
        total = 2 * sum(range(10, 20))
        self.assertEqual(a, total / 20)

    def test_min_empty_returns_default(self):
        self._db.create_range_index("age")
        self.assertIsNone(query(self._db).filter(age={"gt": 1000}).min("age"))


class TestQueryCompatibility(_SnapTestCase):
    """Historical query behaviour (no index) still works."""

    def test_filter_no_index(self):
        rows = [r for _, r in query(self._db).filter(id={"gt": 90}).execute()]
        ids = sorted(r["id"] for r in rows)
        self.assertEqual(ids, list(range(91, 100)))

    def test_order_no_index(self):
        rows = query(self._db).order("id", desc=True).slice(5).execute()
        ids = [row["id"] for _, row in rows]
        self.assertEqual(ids, [99, 98, 97, 96, 95])

    def test_where_fn_disqualifies_index(self):
        self._db.create_range_index("age")
        # where_fn() should clear where_conditions, so explain shows full scan.
        q = query(self._db).filter(age={"lt": 40}).where_fn(lambda r: r["age"] > 30)
        self.assertEqual(q.explain(), "Scan(full)")
        rows = [r for _, r in q.execute()]
        ages = sorted({r["age"] for r in rows})
        self.assertEqual(ages, [31, 32, 33, 34, 35, 36, 37, 38, 39])

    def test_slice_with_index(self):
        self._db.create_range_index("age")
        rows = query(self._db).filter(age={"gte": 0, "lt": 50}).slice(3).execute()
        self.assertEqual(len(rows), 3)

    def test_first_with_index(self):
        self._db.create_range_index("age")
        first = query(self._db).filter(age={"gte": 30, "lt": 40}).first()
        self.assertIsNotNone(first)
        self.assertGreaterEqual(first[1]["age"], 30)
        self.assertLess(first[1]["age"], 40)


class TestImmutableQuery(_SnapTestCase):
    """Query.filter() returns a new Query (never mutates self)."""

    def test_filter_returns_new_instance(self):
        a = query(self._db)
        b = a.filter(id=1)
        self.assertIsNot(a, b)
        self.assertIsNone(a.where_conditions)
        self.assertEqual(b.where_conditions, {"id": 1})

    def test_chained_filter_merges(self):
        self._db.create_range_index("age")
        q = query(self._db).filter(id={"gt": 0}).filter(age={"lt": 10})
        self.assertEqual(q.where_conditions, {"id": {"gt": 0}, "age": {"lt": 10}})


if __name__ == "__main__":
    unittest.main()
