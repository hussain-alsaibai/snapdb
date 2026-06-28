"""
Tests for the v0.6.0 feature additions:

  * #4 Vectorized bitmask predicates (ColumnarTable.select_where / SnapDB.select_where)
  * #6 Auto-indexing (SnapDB(auto_index=True))
  * #7 Zero-copy NumPy / buffer export (Column.to_numpy / buffer)

NumPy tests are skipped when numpy is unavailable.
"""
import os
import tempfile
import unittest

from snapdb import SnapDB, Schema, ColumnDef
from snapdb.columnar import ColumnarTable

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _tmp(suffix=".snap"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    os.remove(path)
    return path


def _cleanup(path):
    for p in (path, path.replace(".snap", ".wal")):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


class TestVectorizedPredicates(unittest.TestCase):
    def setUp(self):
        self.t = ColumnarTable(
            "t", [("id", "i32"), ("age", "i32"), ("status", "bytes:8"), ("score", "f64")]
        )
        self.rows = [
            {"id": i, "age": 20 + (i % 50), "status": b"active" if i % 2 == 0 else b"idle",
             "score": float(i)}
            for i in range(1000)
        ]
        self.t.batch_insert(self.rows)

    def _ref(self, pred):
        return [r for r in (dict(id=x["id"], age=x["age"],
                                 status=x["status"].decode(), score=x["score"])
                            for x in self.rows) if pred(r)]

    def test_and_matches_reference(self):
        got = self.t.select_where([("age", ">", 40), ("status", "==", b"active")])
        ref = self._ref(lambda r: r["age"] > 40 and r["status"] == "active")
        self.assertEqual(len(got), len(ref))
        self.assertTrue(all(g["age"] > 40 and g["status"] == "active" for g in got))

    def test_or_combine(self):
        got = self.t.select_where([("age", "<", 22), ("age", ">", 68)], combine="or",
                                  columns=["age"])
        ref = self._ref(lambda r: r["age"] < 22 or r["age"] > 68)
        self.assertEqual(len(got), len(ref))

    def test_between_and_in(self):
        btw = self.t.select_where([("age", "between", (30, 32))], columns=["age"])
        self.assertTrue(all(30 <= r["age"] <= 32 for r in btw))
        self.assertTrue(len(btw) > 0)
        inx = self.t.select_where([("status", "in", [b"active"])], columns=["status"])
        self.assertTrue(all(r["status"] == "active" for r in inx))

    def test_projection_limit_offset(self):
        rows = self.t.select_where([("age", ">=", 20)], columns=["id"], limit=5, offset=3)
        self.assertEqual(len(rows), 5)
        self.assertEqual(list(rows[0].keys()), ["id"])

    def test_dict_shorthand_and_empty(self):
        got = self.t.select_where({"status": b"active", "age": {"gte": 60}})
        self.assertTrue(all(g["status"] == "active" and g["age"] >= 60 for g in got))
        # no conditions -> all live rows
        self.assertEqual(len(self.t.select_where([])), 1000)

    def test_errors(self):
        with self.assertRaises(ValueError):
            self.t.select_where([("nope", "==", 1)])
        with self.assertRaises(ValueError):
            self.t.select_where([("age", "??", 1)])
        with self.assertRaises(ValueError):
            self.t.select_where([("age", ">", 1)], combine="xor")

    def test_excludes_deleted_rows(self):
        self.t.delete(0)  # id 0, age 20, active
        got = self.t.select_where([("id", "==", 0)])
        self.assertEqual(got, [])


class TestAutoIndex(unittest.TestCase):
    def setUp(self):
        self.path = _tmp()

    def tearDown(self):
        _cleanup(self.path)

    def test_row_auto_index_builds_after_threshold(self):
        db = SnapDB(self.path, Schema([ColumnDef("id", "i32"), ColumnDef("name", "bytes:16")]),
                    auto_index=True, auto_index_threshold=3)
        try:
            for i in range(50):
                db.insert({"id": i, "name": f"u{i % 5}".encode()})
            self.assertNotIn("name", db._indexes)
            for _ in range(3):
                db.find(name=b"u2")
            self.assertIn("name", db._indexes)
            self.assertEqual(len(db.find(name=b"u2")), 10)
        finally:
            db.close()

    def test_find_scan_fallback_without_index(self):
        db = SnapDB(self.path, Schema([ColumnDef("id", "i32"), ColumnDef("name", "bytes:16")]))
        try:
            for i in range(20):
                db.insert({"id": i, "name": f"u{i % 4}".encode()})
            # no index created at all -> scan fallback still returns correct rows
            self.assertEqual(len(db.find(name=b"u1")), 5)
            self.assertEqual(len(db.find(id=7)), 1)
        finally:
            db.close()

    def test_columnar_auto_index_on_eq_filter(self):
        db = SnapDB(self.path, Schema([ColumnDef("id", "i32"), ColumnDef("grp", "i32")]),
                    storage_type="columnar", auto_index=True, auto_index_threshold=2)
        try:
            db.batch_insert([{"id": i, "grp": i % 4} for i in range(40)])
            for _ in range(2):
                db.select_where([("grp", "==", 1)])
            self.assertTrue(getattr(db._table, "_indexes", None) and "grp" in db._table._indexes)
        finally:
            db.close()


@unittest.skipIf(np is None, "numpy not installed")
class TestZeroCopyExport(unittest.TestCase):
    def setUp(self):
        self.path = _tmp()

    def tearDown(self):
        _cleanup(self.path)

    def test_to_numpy_copy_and_view(self):
        db = SnapDB(self.path, Schema([ColumnDef("id", "i32"), ColumnDef("v", "f64")]),
                    storage_type="columnar")
        try:
            db.batch_insert([{"id": i, "v": float(i)} for i in range(100)])
            arr = db.to_numpy("v")
            self.assertEqual(arr.shape, (100,))
            self.assertEqual(float(arr.sum()), float(sum(range(100))))
            view = db.to_numpy("v", zero_copy=True)
            self.assertIsNotNone(view.base)  # shares memory
            buf = db.column_buffer("id")
            self.assertEqual(buf.nbytes, 100 * 4)
            del view, buf  # release buffer locks before close
        finally:
            db.close()

    def test_encoded_column_falls_back_to_copy(self):
        # delta-encoded column has no contiguous buffer -> to_numpy copies
        t = ColumnarTable("t", [("ts", "i64")], delta_columns=["ts"])
        base = 1_700_000_000
        t.batch_insert([{"ts": base + i} for i in range(200)])
        self.assertTrue(t.columns["ts"]._delta_mode)
        arr = t.to_numpy("ts")
        self.assertEqual(int(arr[199]), base + 199)
        with self.assertRaises(TypeError):
            t.column_buffer("ts")  # zero-copy unavailable for encoded column


if __name__ == "__main__":
    unittest.main(verbosity=2)
