"""
Tests for the v0.14.0 feature set:

  * per-slab zone maps behind ``range_find`` (must stay correct across a
    reopen that mixes disk-loaded and freshly-inserted rows in one slab, and
    across deletes/updates that leave stale-but-conservative bounds),
  * named snapshots (``snapshot`` / ``list_snapshots`` / ``open_snapshot`` /
    ``drop_snapshot``), including point-in-time isolation from later writes,
  * the codegen-compiled ``query.filter`` predicate,
  * the columnar liveness fix (a legitimately-null first column must not hide
    a live row from scans/aggregates), in both the pure-Python and, when
    NumPy is present, the vectorized paths.
"""
import os
import shutil
import tempfile
import unittest

from snapdb import SnapDB, Schema, ColumnDef, ColumnarTable
from snapdb.query import query


def _tmpdir():
    return tempfile.mkdtemp(prefix="snapdb_v014_")


class TestZoneMapRangeFind(unittest.TestCase):
    def setUp(self):
        self.dir = _tmpdir()
        self.path = os.path.join(self.dir, "z.snap")
        self.schema = Schema([ColumnDef("id", "i32"), ColumnDef("v", "i32")])

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _brute(self, db, lo, hi):
        return sorted(r["id"] for _, r in db if lo <= r["v"] <= hi)

    def test_range_find_matches_brute_force_with_deletes(self):
        # Small page_size forces many slabs so pruning actually kicks in.
        db = SnapDB(self.path, self.schema, page_size=64)
        vals = [(i * 37) % 1000 for i in range(400)]
        ids = [db.insert({"id": i, "v": vals[i]}) for i in range(400)]
        for i in range(0, 400, 5):
            db.delete(ids[i])
        for lo, hi in [(100, 300), (0, 50), (950, 1000), (500, 500)]:
            got = sorted(r["id"] for r in db.range_find("v", lo, hi))
            self.assertEqual(got, self._brute(db, lo, hi), (lo, hi))
        db.close()

    def test_range_find_correct_after_reopen_mixed_slab(self):
        # Rows loaded from disk + new rows land in the same (last) slab; the
        # zone map must be rebuilt to include the disk rows, not just new ones.
        db = SnapDB(self.path, self.schema, page_size=64)
        db.batch_insert([{"id": i, "v": i} for i in range(60)])
        db.close()

        db = SnapDB(self.path, self.schema, page_size=64)
        for j in range(10):
            db.insert({"id": 1000 + j, "v": 5000 + j})
        # A range that only the disk-loaded rows fall into must still be found
        # after new (higher) rows were appended to the last slab.
        got = sorted(r["id"] for r in db.range_find("v", 10, 20))
        self.assertEqual(got, list(range(10, 21)))
        got_new = sorted(r["id"] for r in db.range_find("v", 5000, 5005))
        self.assertEqual(got_new, [1000, 1001, 1002, 1003, 1004, 1005])
        db.close()

    def test_range_find_correct_after_update(self):
        db = SnapDB(self.path, self.schema, page_size=64)
        ids = [db.insert({"id": i, "v": i}) for i in range(40)]
        db.range_find("v", 0, 5)  # build zone maps
        db.update(ids[0], {"v": 999})  # widen upward
        self.assertEqual(sorted(r["id"] for r in db.range_find("v", 900, 1000)), [0])
        db.close()


class TestSnapshots(unittest.TestCase):
    def setUp(self):
        self.dir = _tmpdir()
        self.path = os.path.join(self.dir, "s.snap")
        self.schema = Schema([ColumnDef("id", "i32"), ColumnDef("v", "f64")])

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_named_snapshot_roundtrip_and_isolation(self):
        db = SnapDB(self.path, self.schema)
        db.batch_insert([{"id": i, "v": float(i)} for i in range(100)])
        db.snapshot("v1")
        db.insert({"id": 100, "v": 100.0})
        db.delete(0)

        names = {s["name"] for s in db.list_snapshots()}
        self.assertIn("v1", names)

        snap = db.open_snapshot("v1")
        try:
            self.assertEqual(len(snap), 100)          # state as of snapshot
            self.assertIsNotNone(snap.get(0))         # row 0 not yet deleted
            self.assertIsNone(snap.get(100))          # row 100 not yet inserted
            snap.insert({"id": 999, "v": 9.0})        # write to snapshot copy
        finally:
            snap.close()

        # Live db and manifest are untouched by reads/writes to the snapshot.
        self.assertEqual(len(db), 100)                # 101 inserted - 1 deleted
        self.assertIsNone(db.get(0))
        self.assertIsNone(db.get(999))
        self.assertEqual(next(s for s in db.list_snapshots()
                              if s["name"] == "v1")["rows"], 100)
        db.close()

    def test_auto_named_and_drop(self):
        db = SnapDB(self.path, self.schema)
        db.insert({"id": 1, "v": 1.0})
        name = db.snapshot()
        self.assertTrue(name.startswith("snap_"))
        self.assertEqual(len(db.list_snapshots()), 1)
        db.drop_snapshot(name)
        self.assertEqual(db.list_snapshots(), [])
        with self.assertRaises(KeyError):
            db.open_snapshot(name)
        db.close()

    def test_columnar_snapshot(self):
        p = os.path.join(self.dir, "c.snap")
        db = SnapDB(p, self.schema, storage_type="columnar")
        db.batch_insert([{"id": i, "v": float(i)} for i in range(50)])
        db.snapshot("c1")
        db.insert({"id": 50, "v": 50.0})
        snap = db.open_snapshot("c1")
        try:
            self.assertEqual(len(snap), 50)
            self.assertIsNone(snap.get(50))
        finally:
            snap.close()
        db.close()


class TestQueryCodegenPredicate(unittest.TestCase):
    def setUp(self):
        self.dir = _tmpdir()
        self.path = os.path.join(self.dir, "q.snap")
        self.schema = Schema([ColumnDef("id", "i32"), ColumnDef("age", "i32")])

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_compiled_filter_matches_semantics(self):
        db = SnapDB(self.path, self.schema)
        db.batch_insert([{"id": i, "age": 20 + (i % 50)} for i in range(500)])

        got = sorted(r["id"] for _, r in query(db).filter(age=25).execute())
        self.assertEqual(got, sorted(i for i in range(500) if 20 + (i % 50) == 25))

        got = sorted(r["id"] for _, r in
                     query(db).filter(age={"gte": 60, "lt": 65}).execute())
        self.assertEqual(got, sorted(i for i in range(500)
                                     if 60 <= 20 + (i % 50) < 65))

        got = sorted(r["id"] for _, r in
                     query(db).filter(age={"gte": 60}, id={"lt": 100}).execute())
        self.assertEqual(got, sorted(i for i in range(500)
                                     if 20 + (i % 50) >= 60 and i < 100))
        db.close()

    def test_empty_filter_is_identity(self):
        db = SnapDB(self.path, self.schema)
        db.batch_insert([{"id": i, "age": i} for i in range(10)])
        self.assertEqual(len(query(db).filter().execute()), 10)
        db.close()


class TestColumnarNullableFirstColumn(unittest.TestCase):
    """A live row whose first column is legitimately None must appear in every
    scan/aggregate, not just get() — the pre-v0.14 first-column-null 'deleted'
    heuristic hid it."""

    def _table(self):
        t = ColumnarTable("t", [("a", "i32"), ("b", "i32")])
        t.insert({"a": None, "b": 5})
        t.insert({"a": 2, "b": 10})
        return t

    def test_pure_python_paths(self):
        t = self._table()
        self.assertEqual(len(t.select()), 2)
        self.assertEqual(t.count_where([("b", "eq", 5)], use_numpy=False), 1)
        self.assertEqual(t.aggregate("b", "count",
                                     where=lambda r: r["b"] > 5), 1)

    def test_numpy_paths_when_available(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        t = self._table()
        self.assertEqual(len(t.select_where([("b", "gte", 0)], use_numpy=True)), 2)
        self.assertEqual(t.count_where([("b", "eq", 5)], use_numpy=True), 1)

    def test_delete_hides_row(self):
        t = self._table()
        t.delete(0)
        self.assertEqual(len(t.select()), 1)


if __name__ == "__main__":
    unittest.main()
