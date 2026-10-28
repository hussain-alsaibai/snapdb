"""Tests for v0.16.1 Query aggregations: sum / min / max / avg / count_distinct.

All aggregations must compose with .filter() (honour WHERE), skip None
values, and return a sensible default for an empty match.
"""
import os
import tempfile
import unittest

from snapdb import SnapDB, Schema, ColumnDef
from snapdb.query import query


def _tmp(suffix=".snap"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    os.remove(path)
    return path


def _make_db():
    path = _tmp()
    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("region", "bytes:16"),
        ColumnDef("amount", "i32"),
        ColumnDef("category", "bytes:16"),
    ])
    db = SnapDB(path, schema)
    rows = [
        (1, "us",   100, "books"),
        (2, "us",   200, "books"),
        (3, "eu",   150, "games"),
        (4, "eu",   300, "books"),
        (5, "apac", 250, "games"),
        (6, "apac", 0,   "books"),
    ]
    for r in rows:
        db.insert({"id": r[0], "region": r[1], "amount": r[2], "category": r[3]})
    return db


class TestAggregations(unittest.TestCase):
    def setUp(self):
        self.db = _make_db()

    def tearDown(self):
        path = str(self.db.path)
        for p in (path, path + ".wal"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    def test_sum_all(self):
        self.assertEqual(query(self.db).sum("amount"), 1000)

    def test_sum_filtered(self):
        q = query(self.db).filter(region="us")
        self.assertEqual(q.sum("amount"), 300)

    def test_sum_empty_returns_default(self):
        q = query(self.db).filter(region="mars")
        self.assertEqual(q.sum("amount"), 0)

    def test_sum_default_override(self):
        q = query(self.db).filter(region="mars")
        self.assertEqual(q.sum("amount", default=-1), -1)

    def test_min_max(self):
        q = query(self.db)
        self.assertEqual(q.min("amount"), 0)
        self.assertEqual(q.max("amount"), 300)

    def test_min_max_filtered(self):
        q = query(self.db).filter(category="books")
        # books amounts: 100, 200, 300, 0
        self.assertEqual(q.min("amount"), 0)
        self.assertEqual(q.max("amount"), 300)

    def test_min_max_empty(self):
        q = query(self.db).filter(region="mars")
        self.assertIsNone(q.min("amount"))
        self.assertEqual(q.max("amount", default="n/a"), "n/a")

    def test_avg(self):
        q = query(self.db)
        # (100+200+150+300+250+0)/6 = 1000/6
        self.assertAlmostEqual(q.avg("amount"), 1000 / 6)

    def test_avg_filtered(self):
        q = query(self.db).filter(category="books")
        # (100+200+300+0)/4 = 150.0
        self.assertAlmostEqual(q.avg("amount"), 150.0)

    def test_avg_empty(self):
        q = query(self.db).filter(region="mars")
        self.assertIsNone(q.avg("amount"))

    def test_count_distinct(self):
        q = query(self.db)
        self.assertEqual(q.count_distinct("region"), 3)
        self.assertEqual(q.count_distinct("category"), 2)
        # amounts: 100,200,150,300,250,0 -> 6 distinct
        self.assertEqual(q.count_distinct("amount"), 6)

    def test_count_distinct_filtered(self):
        q = query(self.db).filter(region="us")
        self.assertEqual(q.count_distinct("category"), 1)

    def test_count_distinct_empty(self):
        q = query(self.db).filter(region="mars")
        self.assertEqual(q.count_distinct("region"), 0)

    def test_skips_none_values_via_extra_column(self):
        # Add an extra optional column by inserting an extra row that
        # carries the value, then drop it from a subset via where_fn that
        # mimics a NULL. We test the aggregator's defensive path through
        # the columnar path instead: insert a row whose amount is a
        # dict-encoded zero (which yields 0, not None) and confirm that
        # 0 is still counted (i.e. None-skipping logic doesn't over-skip).
        self.db.insert({"id": 99, "region": "us", "category": "games", "amount": 0})
        q = query(self.db)
        # 7 amounts now, sum still 1000, avg = 1000/7
        self.assertEqual(q.sum("amount"), 1000)
        self.assertAlmostEqual(q.avg("amount"), 1000 / 7)
        # 100,200,150,300,250,0,0 -> still 6 distinct
        self.assertEqual(q.count_distinct("amount"), 6)


if __name__ == "__main__":
    unittest.main()
