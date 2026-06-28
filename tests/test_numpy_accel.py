"""
Tests for the optional NumPy-accelerated aggregate path (issue #14).

The NumPy path must produce results identical (within float tolerance) to the
pure-Python path, across dtypes, nulls, negatives, and empty columns. Skipped
entirely when NumPy is not installed (the zero-dependency default still works).
"""
import importlib.util
import random
import unittest

from snapdb.columnar import ColumnarTable, _HAS_NUMPY

_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(_NUMPY, "numpy not installed")
class TestNumpyAggregateParity(unittest.TestCase):
    def setUp(self):
        self.rnd = random.Random(1234)

    def _both(self, t, col, agg):
        return t.aggregate(col, agg, use_numpy=True), t.aggregate(col, agg, use_numpy=False)

    def _assert_parity(self, t, col, dtype):
        for agg in ("sum", "min", "max", "avg"):
            npv, pyv = self._both(t, col, agg)
            if agg == "avg" or dtype.startswith("f"):
                if npv is None or pyv is None:
                    self.assertEqual(npv, pyv)
                else:
                    self.assertAlmostEqual(npv, pyv, places=3)
            else:
                self.assertEqual(npv, pyv, f"{dtype} {agg}: {npv} != {pyv}")

    def test_parity_all_int_dtypes(self):
        for dtype in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"):
            t = ColumnarTable("t", [("x", dtype)])
            signed = dtype.startswith("i")
            lo, hi = (-100, 100) if signed else (0, 200)
            t.batch_insert([{"x": self.rnd.randint(lo, hi)} for _ in range(500)])
            self._assert_parity(t, "x", dtype)

    def test_parity_floats(self):
        for dtype in ("f32", "f64"):
            t = ColumnarTable("t", [("x", dtype)])
            t.batch_insert([{"x": self.rnd.uniform(-1000, 1000)} for _ in range(500)])
            self._assert_parity(t, "x", dtype)

    def test_parity_with_nulls(self):
        t = ColumnarTable("t", [("x", "i32")])
        t.batch_insert([{"x": None if i % 4 == 0 else self.rnd.randint(-50, 50)}
                        for i in range(500)])
        self._assert_parity(t, "x", "i32")

    def test_empty_and_all_null(self):
        t = ColumnarTable("t", [("x", "i32")])
        t.batch_insert([{"x": None} for _ in range(10)])
        # all-null: sum 0, min/max None — both paths agree
        self.assertEqual(*self._both(t, "x", "sum"))
        self.assertEqual(t.aggregate("x", "sum", use_numpy=True), 0)
        self.assertIsNone(t.aggregate("x", "min", use_numpy=True))
        self.assertIsNone(t.aggregate("x", "min", use_numpy=False))

    def test_numpy_used_by_default_when_available(self):
        # The default (use_numpy=None) should engage NumPy when it's installed.
        self.assertTrue(_HAS_NUMPY)
        t = ColumnarTable("t", [("x", "f64")])
        t.batch_insert([{"x": float(i)} for i in range(1000)])
        self.assertAlmostEqual(t.aggregate("x", "sum"), sum(float(i) for i in range(1000)), places=3)

    def test_encoded_columns_fall_through_correctly(self):
        # delta / FOR columns can't use the buffer path; results stay correct.
        base = 1_700_000_000
        td = ColumnarTable("t", [("ts", "i64")], delta_columns=["ts"])
        td.batch_insert([{"ts": base + i} for i in range(200)])
        self.assertTrue(td.columns["ts"]._delta_mode)
        self.assertEqual(td.aggregate("ts", "sum"), sum(base + i for i in range(200)))
        self.assertEqual(td.aggregate("ts", "max"), base + 199)

        tf = ColumnarTable("t", [("v", "i32")], for_columns=["v"])
        tf.batch_insert([{"v": 1000 + (i % 50)} for i in range(200)])
        self.assertTrue(tf.columns["v"]._for_mode)
        self.assertEqual(tf.aggregate("v", "sum"), sum(1000 + (i % 50) for i in range(200)))

    def test_i64_sum_exact(self):
        # large i64 values: NumPy path defers to exact Python sum; verify exactness
        t = ColumnarTable("t", [("x", "i64")])
        vals = [self.rnd.randint(-10**15, 10**15) for _ in range(1000)]
        t.batch_insert([{"x": v} for v in vals])
        self.assertEqual(t.aggregate("x", "sum", use_numpy=True), sum(vals))
        self.assertEqual(t.aggregate("x", "sum", use_numpy=False), sum(vals))


if __name__ == "__main__":
    unittest.main(verbosity=2)
