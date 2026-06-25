"""Tests for SnapDB v0.2.0 — Query Engine, Indexing, Transactions"""
import os
import tempfile
import unittest

from snapdb import SnapDB, Schema, ColumnDef


class TestSnapDBv02(unittest.TestCase):
    """Test v0.2.0 features."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.schema = Schema([
            ColumnDef("id", "i32"),
            ColumnDef("name", "bytes:16"),
            ColumnDef("score", "f32"),
        ])

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_query_engine(self):
        """Test SQL-like queries."""
        db = SnapDB(os.path.join(self.tmpdir, "test.snap"), self.schema)

        # Insert test data
        for i in range(20):
            db.insert({"id": i, "name": f"user{i}".encode(), "score": float(i * 10)})

        # Basic WHERE
        from query import query
        results = query(db).filter(score=100.0).execute()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1]["id"], 10)

        # ORDER BY + LIMIT
        results = query(db).order("score", desc=True).slice(3).execute()
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][1]["score"], 190.0)
        self.assertEqual(results[1][1]["score"], 180.0)

        # OFFSET
        results = query(db).order("score", desc=True).slice(3, offset=3).execute()
        self.assertEqual(results[0][1]["score"], 160.0)

        # RANGE query
        results = query(db).where_fn(lambda r: 50 <= r["score"] <= 100).execute()
        scores = [r["score"] for _, r in results]
        self.assertIn(50.0, scores)
        self.assertIn(100.0, scores)

        db.close()

    def test_hash_index(self):
        """Test O(1) index lookups."""
        db = SnapDB(os.path.join(self.tmpdir, "test.snap"), self.schema)

        for i in range(100):
            db.insert({"id": i, "name": f"user{i%10}".encode(), "score": float(i)})

        # Create index on "name"
        db.create_index("name")
        self.assertIn("name", db._indexes)

        # Lookup
        matches = db.find(name="user5".encode())
        self.assertEqual(len(matches), 10)  # user5 at indices 5, 15, 25...

        # Delete and verify index updated
        db.delete(5)
        matches = db.find(name="user5".encode())
        self.assertEqual(len(matches), 9)

        db.close()

    def test_transactions(self):
        """Test WAL transactions."""
        db = SnapDB(os.path.join(self.tmpdir, "test.snap"), self.schema)

        # Successful transaction
        with db.transaction():
            db.insert({"id": 1, "name": b"alice", "score": 100.0})
            db.insert({"id": 2, "name": b"bob", "score": 90.0})

        self.assertEqual(len(db), 2)

        # Failed transaction should rollback
        try:
            with db.transaction():
                db.insert({"id": 3, "name": b"charlie", "score": 80.0})
                raise RuntimeError("force rollback")
        except RuntimeError:
            pass

        # Charlie should not be present
        self.assertEqual(len(db), 2)

        db.close()

    def test_zero_copy_preserved(self):
        """Ensure v0.1.0 zero-copy reads still work."""
        db = SnapDB(os.path.join(self.tmpdir, "test.snap"), self.schema)
        db.insert({"id": 1, "name": b"test", "score": 99.9})

        raw = db.get_raw(0)
        self.assertIsInstance(raw, memoryview)
        decoded = db.get(0)
        self.assertEqual(decoded["id"], 1)

        db.close()

    def test_repr_shows_indexes(self):
        db = SnapDB(os.path.join(self.tmpdir, "test.snap"), self.schema)
        db.create_index("id")
        r = repr(db)
        self.assertIn("indexes=['id']", r)
        db.close()


if __name__ == "__main__":
    unittest.main()
