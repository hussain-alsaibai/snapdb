"""
Durability tests for the row store: data must survive close() / reopen,
including across slab expansion, deletes (incl. trailing deletes), and updates.

Before v0.6.0 the on-disk slab/bitmap geometry read by _load did not match what
_create/_expand wrote, and the liveness bitmap was never persisted — so
reopening a multi-slab database lost or corrupted all rows. These tests lock in
the fix.
"""
import os
import tempfile
import unittest

from snapdb import SnapDB, Schema, ColumnDef


def _tmp():
    fd, path = tempfile.mkstemp(suffix=".snap")
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


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.path = _tmp()
        self.schema = Schema([
            ColumnDef("id", "i32"),
            ColumnDef("name", "bytes:16"),
            ColumnDef("score", "f64"),
        ])

    def tearDown(self):
        _cleanup(self.path)

    def _open(self):
        return SnapDB(self.path, self.schema)

    def test_multi_slab_reopen(self):
        n = 500  # spans several 4 KiB slabs (~146 rows each)
        db = self._open()
        for i in range(n):
            db.insert({"id": i, "name": f"u{i}".encode(), "score": float(i)})
        self.assertGreater(len(db._slabs), 1)
        db.close()

        db = self._open()
        try:
            self.assertEqual(len(db), n)
            for i in (0, 145, 146, 300, 499):
                self.assertEqual(db.get(i)["id"], i)
                self.assertEqual(db.get(i)["name"], f"u{i}")
        finally:
            db.close()

    def test_reopen_preserves_deletes_and_updates(self):
        n = 500
        db = self._open()
        for i in range(n):
            db.insert({"id": i, "name": f"u{i}".encode(), "score": float(i)})
        db.delete(3)
        db.delete(499)  # trailing delete in the last slab (high-water edge case)
        db.update(7, {"id": 7, "name": b"seven", "score": -1.0})
        db.close()

        db = self._open()
        try:
            self.assertEqual(len(db), n - 2)
            self.assertIsNone(db.get(3))
            self.assertIsNone(db.get(499))
            self.assertEqual(db.get(7)["name"], "seven")
            self.assertEqual(db.get(7)["score"], -1.0)
            self.assertEqual(db.get(498)["id"], 498)
        finally:
            db.close()

    def test_insert_after_reopen_does_not_overwrite(self):
        n = 500
        db = self._open()
        for i in range(n):
            db.insert({"id": i, "name": f"u{i}".encode(), "score": float(i)})
        db.delete(499)  # would corrupt high-water if reconstructed from liveness
        db.close()

        db = self._open()
        try:
            new_idx = db.insert({"id": 1000, "name": b"new", "score": 9.0})
            self.assertEqual(new_idx, n)          # continues past the high-water mark
            self.assertEqual(db.get(n)["id"], 1000)
            self.assertEqual(db.get(498)["id"], 498)   # neighbor not clobbered
        finally:
            db.close()

    def test_batch_insert_multi_slab_reopen(self):
        # batch_insert grows the file in one shot across many slabs; the result
        # must persist and stay consistent with single inserts.
        db = self._open()
        db.insert({"id": 0, "name": b"first", "score": -1.0})           # single
        db.batch_insert([{"id": i, "name": f"u{i}".encode(), "score": float(i)}
                         for i in range(1, 1000)])                       # spans slabs
        db.insert({"id": 1000, "name": b"last", "score": 1.0})          # single after batch
        self.assertGreater(len(db._slabs), 1)
        self.assertEqual(len(db), 1001)
        db.close()

        db = self._open()
        try:
            self.assertEqual(len(db), 1001)
            self.assertEqual(db.get(0)["name"], "first")
            self.assertEqual(db.get(500)["id"], 500)
            self.assertEqual(db.get(1000)["name"], "last")
            # appends after reopen land at the right high-water
            idx = db.insert({"id": 2000, "name": b"new", "score": 2.0})
            self.assertEqual(idx, 1001)
            self.assertEqual(db.get(1001)["id"], 2000)
        finally:
            db.close()

    def test_double_reopen_roundtrip(self):
        db = self._open()
        for i in range(300):
            db.insert({"id": i, "name": f"r{i}".encode(), "score": float(i)})
        db.close()
        db = self._open()
        db.insert({"id": 999, "name": b"last", "score": 1.0})
        db.close()
        db = self._open()
        try:
            self.assertEqual(db.get(0)["name"], "r0")
            self.assertEqual(db.get(300)["id"], 999)
            self.assertEqual(len(db), 301)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
