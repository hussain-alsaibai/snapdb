"""
Regression tests for the v0.6.0 performance/correctness work:

  * Hash indexes stay in sync across insert / batch_insert / update / delete
    (previously they went stale after the first build).
  * Delta-encoded columns reconstruct correctly and reads are O(1)/O(n)
    instead of O(n)/O(n^2).
  * Deleting or nulling a delta-encoded row no longer corrupts other rows.
  * Vectorized aggregates match the naive computation.
  * Transaction rollback actually undoes writes (and keeps indexes correct).

These tests close every database handle before deleting its file so they pass
cleanly on Windows (where an open mmap locks the file).
"""
import os
import tempfile
import unittest

from snapdb import SnapDB, Schema, ColumnDef
from snapdb.columnar import ColumnarTable


def _tmp(suffix=".snap"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    os.remove(path)  # we only want the unique name
    return path


def _cleanup(path):
    for p in (path, path.replace(".snap", ".wal")):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


class TestHashIndexMaintenance(unittest.TestCase):
    def setUp(self):
        self.path = _tmp()
        self.schema = Schema([
            ColumnDef("id", "i32"),
            ColumnDef("name", "bytes:16"),
            ColumnDef("score", "f64"),
        ])

    def tearDown(self):
        _cleanup(self.path)

    def test_index_tracks_writes(self):
        db = SnapDB(self.path, self.schema)
        try:
            for i in range(100):
                db.insert({"id": i, "name": f"user{i % 10}".encode(), "score": float(i)})
            db.create_index("name")
            self.assertEqual(len(db.find(name=b"user5")), 10)

            # delete keeps the index in sync
            db.delete(5)
            self.assertEqual(len(db.find(name=b"user5")), 9)

            # insert keeps the index in sync
            db.insert({"id": 200, "name": b"user5", "score": 1.0})
            self.assertEqual(len(db.find(name=b"user5")), 10)

            # update moves the row between index buckets
            db.update(15, {"id": 15, "name": b"userX", "score": 1.0})
            self.assertEqual(len(db.find(name=b"user5")), 9)
            self.assertEqual(len(db.find(name=b"userX")), 1)

            # lookup() uses the index and returns a matching row
            self.assertIsNotNone(db.lookup("name", b"user7"))
            self.assertIsNone(db.lookup("name", b"nobody"))
        finally:
            db.close()

    def test_batch_insert_maintains_index(self):
        db = SnapDB(self.path, self.schema)
        try:
            db.create_index("name")
            db.batch_insert([
                {"id": i, "name": f"u{i % 3}".encode(), "score": float(i)}
                for i in range(30)
            ])
            self.assertEqual(len(db.find(name=b"u0")), 10)
            self.assertEqual(len(db.find(name=b"u1")), 10)
        finally:
            db.close()


class TestColumnarIndex(unittest.TestCase):
    def setUp(self):
        self.path = _tmp()
        self.schema = Schema([
            ColumnDef("id", "i32"),
            ColumnDef("name", "bytes:16"),
        ])

    def tearDown(self):
        _cleanup(self.path)

    def test_columnar_create_index_and_lookup(self):
        db = SnapDB(self.path, self.schema, storage_type="columnar")
        try:
            for i in range(50):
                db.insert({"id": i, "name": f"n{i}".encode()})
            db.create_index("id")          # used to raise for columnar
            row = db.lookup("id", 42)
            self.assertIsNotNone(row)
            self.assertEqual(row["id"], 42)
            db.insert({"id": 999, "name": b"new"})
            self.assertEqual(db.lookup("id", 999)["name"], "new")
        finally:
            db.close()


class TestDeltaEncoding(unittest.TestCase):
    def _table(self):
        t = ColumnarTable("t", [("ts", "i64"), ("v", "i32")], delta_columns=["ts"])
        base = 1_700_000_000
        t.batch_insert([{"ts": base + i * 5, "v": i} for i in range(200)])
        return t, base

    def test_reconstruction_and_scan(self):
        t, base = self._table()
        col = t.columns["ts"]
        self.assertTrue(col._delta_mode and not col._delta_fallback)
        self.assertEqual(col[0], base)
        self.assertEqual(col[199], base + 199 * 5)
        self.assertEqual(t.select_column("ts"), [base + i * 5 for i in range(200)])

    def test_aggregates_match(self):
        t, base = self._table()
        expected = [base + i * 5 for i in range(200)]
        self.assertEqual(t.aggregate("ts", "sum"), sum(expected))
        self.assertEqual(t.aggregate("ts", "min"), min(expected))
        self.assertEqual(t.aggregate("ts", "max"), max(expected))
        self.assertEqual(t.aggregate("ts", "count"), len(expected))

    def test_delete_does_not_corrupt_neighbors(self):
        t, base = self._table()
        t.delete(100)
        self.assertIsNone(t.get(100))
        for i in (0, 50, 99, 101, 150, 199):
            self.assertEqual(t.get(i)["ts"], base + i * 5)

    def test_update_converts_and_stays_correct(self):
        t, base = self._table()
        t.update(150, {"ts": 999})
        self.assertEqual(t.get(150)["ts"], 999)
        self.assertEqual(t.get(151)["ts"], base + 151 * 5)


class TestVectorizedAggregate(unittest.TestCase):
    def test_matches_naive(self):
        t = ColumnarTable("t", [("x", "i64")])
        values = [i * 3 - 7 for i in range(1000)]
        t.batch_insert([{"x": v} for v in values])
        self.assertEqual(t.aggregate("x", "sum"), sum(values))
        self.assertEqual(t.aggregate("x", "min"), min(values))
        self.assertEqual(t.aggregate("x", "max"), max(values))
        self.assertAlmostEqual(t.aggregate("x", "avg"), sum(values) / len(values))

    def test_with_nulls_falls_back_correctly(self):
        t = ColumnarTable("t", [("x", "i64")])
        for i in range(10):
            t.insert({"x": None if i % 2 == 0 else i})
        present = [i for i in range(10) if i % 2 == 1]
        self.assertEqual(t.aggregate("x", "sum"), sum(present))
        self.assertEqual(t.aggregate("x", "count"), len(present))


class TestTransactionRollback(unittest.TestCase):
    def setUp(self):
        self.path = _tmp()
        self.schema = Schema([
            ColumnDef("id", "i32"),
            ColumnDef("name", "bytes:16"),
            ColumnDef("score", "f32"),
        ])

    def tearDown(self):
        _cleanup(self.path)

    def test_rollback_undoes_inserts_and_index(self):
        db = SnapDB(self.path, self.schema)
        try:
            db.create_index("name")
            with db.transaction():
                db.insert({"id": 1, "name": b"alice", "score": 100.0})
                db.insert({"id": 2, "name": b"bob", "score": 90.0})
            self.assertEqual(len(db), 2)

            try:
                with db.transaction():
                    db.insert({"id": 3, "name": b"charlie", "score": 80.0})
                    raise RuntimeError("force rollback")
            except RuntimeError:
                pass

            self.assertEqual(len(db), 2)
            # index must not retain the rolled-back row
            self.assertEqual(db.find(name=b"charlie"), [])
            self.assertEqual(len(db.find(name=b"alice")), 1)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
