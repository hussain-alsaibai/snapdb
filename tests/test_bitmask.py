"""
Tests for the v0.10.0 bitmask query engine (`ColumnarTable.select_bitmask`).

Covers:
  - Single-bit queries across one column (AND / OR / XOR)
  - Multi-bit queries on the same column (within-column AND)
  - Cross-column queries with AND / OR / XOR semantics
  - Column projection / limit / offset
  - Rejection of unknown columns and non-integer dtypes
  - Bitmask queries over encoded columns (delta, FOR, dict)
  - Bitmask queries over null rows (null rows are skipped, do not match)
  - `__buffer__` on ColumnarTable (PEP 688)
  - `to_dataframe()` (returns None when pandas is missing, DataFrame otherwise)
  - `compact()` reclaiming space after deletes and shrinking `_row_count`
"""
import importlib.util
import random
import unittest

from snapdb import ColumnarTable
from snapdb.columnar import _HAS_NUMPY


_PANDAS = importlib.util.find_spec("pandas") is not None


class TestSelectBitmaskBasic(unittest.TestCase):
    """Single-bit predicates across columns with AND / OR / XOR."""

    def setUp(self):
        self.t = ColumnarTable("t", [("flags", "i32"), ("status", "i32")])
        self.t.batch_insert([
            {"flags": 0b0001, "status": 0},  # bit 0 of flags only
            {"flags": 0b0010, "status": 1},  # bit 1 of flags, status bit 0
            {"flags": 0b0011, "status": 0},  # both bits of flags
            {"flags": 0b0000, "status": 1},  # nothing
        ])

    def test_single_bit_and(self):
        # flags bit 0 == 1 AND status bit 0 == 1
        result = self.t.select_bitmask({"flags": (0, 1), "status": (0, 1)})
        # flags bit 0 set: rows 0, 2; status bit 0 set: rows 1, 3
        # intersection: empty
        self.assertEqual(result, [])

    def test_single_bit_and_cross_column(self):
        result = self.t.select_bitmask({"flags": (1, 1), "status": (0, 1)})
        # flags bit 1 set: rows 1, 2; status bit 0 set: rows 1, 3
        # intersection: row 1
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["flags"], 0b0010)
        self.assertEqual(result[0]["status"], 1)

    def test_single_bit_or(self):
        result = self.t.select_bitmask(
            {"flags": (1, 1), "status": (0, 1)}, operator="OR"
        )
        # flags bit 1 set: rows 1, 2; status bit 0 set: rows 1, 3
        # union: rows 1, 2, 3
        self.assertEqual(len(result), 3)
        seen = {(r["flags"], r["status"]) for r in result}
        self.assertEqual(seen, {(2, 1), (3, 0), (0, 1)})

    def test_single_bit_xor(self):
        # XOR across columns: odd number of columns must match.
        result = self.t.select_bitmask(
            {"flags": (1, 1), "status": (0, 1)}, operator="XOR"
        )
        # row 0: flags bit 1 unset (no), status bit 0 unset (no) -> 0 matches -> no
        # row 1: flags bit 1 set (yes), status bit 0 set (yes) -> 2 -> no
        # row 2: flags bit 1 set (yes), status bit 0 unset (no) -> 1 -> yes
        # row 3: flags bit 1 unset (no), status bit 0 set (yes) -> 1 -> yes
        self.assertEqual(len(result), 2)
        seen = {(r["flags"], r["status"]) for r in result}
        self.assertEqual(seen, {(3, 0), (0, 1)})

    def test_match_all_default_is_and(self):
        result_default = self.t.select_bitmask({"flags": (1, 1), "status": (0, 1)})
        result_explicit = self.t.select_bitmask(
            {"flags": (1, 1), "status": (0, 1)}, match_all=True
        )
        self.assertEqual(result_default, result_explicit)

    def test_match_all_false_is_or(self):
        result = self.t.select_bitmask(
            {"flags": (1, 1), "status": (0, 1)}, match_all=False
        )
        result_or = self.t.select_bitmask(
            {"flags": (1, 1), "status": (0, 1)}, operator="OR"
        )
        self.assertEqual(result, result_or)

    def test_unknown_column_raises(self):
        with self.assertRaises(ValueError):
            self.t.select_bitmask({"missing": (0, 1)})

    def test_invalid_operator_raises(self):
        with self.assertRaises(ValueError):
            self.t.select_bitmask({"flags": (0, 1)}, operator="NAND")

    def test_empty_bitmask_returns_empty(self):
        self.assertEqual(self.t.select_bitmask({}), [])

    def test_non_integer_dtype_raises(self):
        t = ColumnarTable("t", [("name", "bytes:16")])
        t.insert({"name": "alice"})
        with self.assertRaises(TypeError):
            t.select_bitmask({"name": (0, 1)})


class TestSelectBitmaskMultiBit(unittest.TestCase):
    """Multiple bits per column (within-column AND combination)."""

    def test_multi_bit_within_column_and(self):
        t = ColumnarTable("t", [("flags", "i32")])
        t.batch_insert([
            {"flags": 0b0001},
            {"flags": 0b0010},
            {"flags": 0b0011},
            {"flags": 0b0000},
        ])
        result = t.select_bitmask({"flags": [(0, 1), (1, 1)]})
        # Within-column AND: row 2 has both bits set.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["flags"], 0b0011)

    def test_multi_bit_within_column_or(self):
        t = ColumnarTable("t", [("flags", "i32")])
        t.batch_insert([
            {"flags": 0b0001},
            {"flags": 0b0010},
            {"flags": 0b0011},
            {"flags": 0b0000},
        ])
        result = t.select_bitmask({"flags": [(0, 1), (1, 1)]}, operator="OR")
        # Single column -> OR == column matches -> row 2 only
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["flags"], 0b0011)

    def test_expected_zero(self):
        # Bit must be 0 (not 1).
        t = ColumnarTable("t", [("flags", "i32")])
        t.batch_insert([
            {"flags": 0b0001},
            {"flags": 0b0010},
            {"flags": 0b0011},
            {"flags": 0b0000},
        ])
        result = t.select_bitmask({"flags": (0, 0)})
        # flags bit 0 must be 0: rows 1 (0b0010) and 3 (0b0000).
        # row 0 (0b0001) and row 2 (0b0011) have bit 0 = 1.
        self.assertEqual(len(result), 2)
        self.assertEqual(sorted(r["flags"] for r in result), [0, 2])


class TestSelectBitmaskProjection(unittest.TestCase):
    """Column projection, limit, offset."""

    def test_column_projection(self):
        t = ColumnarTable("t", [("flags", "i32"), ("status", "i32"), ("name", "bytes:16")])
        t.batch_insert([
            {"flags": 1, "status": 1, "name": b"alice"},
            {"flags": 0, "status": 1, "name": b"bob"},
            {"flags": 1, "status": 0, "name": b"carol"},
        ])
        result = t.select_bitmask(
            {"flags": (0, 1)}, columns=["name"]
        )
        self.assertEqual(len(result), 2)
        # bytes columns round-trip through the str decoder during tolist().
        self.assertEqual({r["name"] for r in result}, {"alice", "carol"})

    def test_limit_and_offset(self):
        t = ColumnarTable("t", [("flags", "i32")])
        t.batch_insert([{"flags": 0b0011} for _ in range(10)])
        result = t.select_bitmask({"flags": (0, 1)}, limit=3, offset=2)
        self.assertEqual(len(result), 3)


class TestSelectBitmaskNullRows(unittest.TestCase):
    """Null rows are skipped and do not match the predicate."""

    def test_null_rows_skipped(self):
        t = ColumnarTable("t", [("flags", "i32")])
        t.insert({"flags": 0b0001})
        t.insert({"flags": None})
        t.insert({"flags": 0b0011})
        # live_mask excludes rows where every column is null; here only the
        # middle row is null (so it is dead), and the live rows are rows 0 and 2.
        result = t.select_bitmask({"flags": (0, 1)})
        self.assertEqual(len(result), 2)
        self.assertEqual([r["flags"] for r in result], [1, 3])

    def test_full_column_null(self):
        t = ColumnarTable("t", [("flags", "i32")])
        t.insert({"flags": None})
        t.insert({"flags": None})
        result = t.select_bitmask({"flags": (0, 1)})
        self.assertEqual(result, [])


class TestSelectBitmaskEncoded(unittest.TestCase):
    """Bitmask queries must work over encoded columns (delta, FOR, dict)."""

    def test_over_delta_encoded_column(self):
        t = ColumnarTable(
            "t",
            [("id", "i64")],
            delta_columns=["id"],
        )
        ids = [100 + i for i in range(50)]
        t.batch_insert([{"id": v} for v in ids])
        # 100 = 0b1100100, so bit 0 alternates starting at 0 (100 has bit 0 = 0,
        # 101 has bit 0 = 1, etc.). We get every odd index from 101..149.
        result = t.select_bitmask({"id": (0, 1)})
        bits = [int(r["id"]) & 1 for r in result]
        self.assertTrue(all(b == 1 for b in bits))
        ids_returned = [int(r["id"]) for r in result]
        self.assertEqual(len(ids_returned), 25)
        self.assertTrue(all(v & 1 for v in ids_returned))

    def test_over_for_encoded_column(self):
        t = ColumnarTable(
            "t",
            [("score", "u32")],
            for_columns=["score"],
        )
        # 0..100 range triggers FOR encoding.
        t.batch_insert([{"score": i % 101} for i in range(50)])
        result = t.select_bitmask({"score": (0, 1)})
        bits = [int(r["score"]) & 1 for r in result]
        self.assertTrue(all(b == 1 for b in bits))

    def test_over_dict_encoded_column(self):
        # Dict encoding is for bytes, but bitmask requires integer. The
        # engine should reject that with TypeError.
        t = ColumnarTable(
            "t",
            [("name", "bytes:16")],
            dict_columns=["name"],
        )
        t.insert({"name": b"alice"})
        with self.assertRaises(TypeError):
            t.select_bitmask({"name": (0, 1)})


class TestSelectBitmaskLarge(unittest.TestCase):
    """Stress test: 100K rows, scan is O(n) and correct."""

    def test_large_scan_correctness(self):
        rnd = random.Random(42)
        t = ColumnarTable("t", [("flags", "i32"), ("status", "i32")])
        rows = [{"flags": rnd.randint(0, 255), "status": rnd.randint(0, 1)} for _ in range(100_000)]
        t.batch_insert(rows)
        # Reference: list comprehension
        expected = [
            {"flags": r["flags"], "status": r["status"]}
            for r in rows
            if (r["flags"] >> 3) & 1 and r["status"] == 1
        ]
        result = t.select_bitmask({"flags": (3, 1), "status": (0, 1)})
        self.assertEqual(len(result), len(expected))


class TestBufferProtocol(unittest.TestCase):
    """PEP 688 __buffer__ on ColumnarTable returns a memoryview of column 0."""

    def test_buffer_returns_memoryview(self):
        t = ColumnarTable("t", [("id", "i32"), ("name", "bytes:16")])
        t.batch_insert([{"id": i, "name": f"u{i}".encode()} for i in range(5)])
        mv = t.__buffer__(0)
        self.assertIsInstance(mv, memoryview)
        # 5 i32 values = 20 bytes total.
        self.assertEqual(mv.nbytes, 5 * 4)
        # The element view should match the inserted values.
        self.assertEqual(list(mv), [0, 1, 2, 3, 4])

    def test_buffer_empty_table_returns_empty_view(self):
        t = ColumnarTable("t", [("id", "i32")])
        # An empty column's buffer is a 0-byte memoryview, not an error —
        # this matches the existing Column.buffer() semantics.
        mv = t.__buffer__(0)
        self.assertIsInstance(mv, memoryview)
        self.assertEqual(mv.nbytes, 0)


@unittest.skipUnless(_PANDAS, "pandas not installed")
class TestToDataframe(unittest.TestCase):
    """to_dataframe returns a pandas DataFrame when pandas is installed."""

    def test_returns_dataframe(self):
        import pandas as pd

        t = ColumnarTable("t", [("id", "i32"), ("name", "bytes:16")])
        t.batch_insert([{"id": i, "name": f"u{i}".encode()} for i in range(5)])
        df = t.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertEqual(set(df.columns), {"id", "name"})


class TestToDataframeNoPandas(unittest.TestCase):
    """to_dataframe returns None when pandas is missing (simulated)."""

    def test_returns_none_when_missing(self):
        t = ColumnarTable("t", [("id", "i32")])
        t.insert({"id": 1})
        # Real check: simulate missing pandas by patching the import.
        import sys
        original = sys.modules.get("pandas")
        sys.modules["pandas"] = None  # type: ignore
        try:
            self.assertIsNone(t.to_dataframe())
        finally:
            if original is not None:
                sys.modules["pandas"] = original
            else:
                sys.modules.pop("pandas", None)


class TestCompact(unittest.TestCase):
    """compact() rebuilds columns and shrinks _row_count."""

    def test_compact_no_deletes(self):
        t = ColumnarTable("t", [("flags", "i32")])
        t.batch_insert([{"flags": i} for i in range(10)])
        result = t.compact()
        self.assertEqual(result["rows_before"], 10)
        self.assertEqual(result["rows_after"], 10)
        self.assertEqual(result["rows_removed"], 0)
        self.assertEqual(result["bytes_freed"], 0)
        self.assertEqual(len(t), 10)

    def test_compact_after_deletes(self):
        t = ColumnarTable("t", [("flags", "i32"), ("status", "i32")])
        t.batch_insert([{"flags": i, "status": i % 2} for i in range(10)])
        t.delete(0)
        t.delete(2)
        t.delete(4)
        result = t.compact()
        self.assertEqual(result["rows_before"], 10)
        self.assertEqual(result["rows_after"], 7)
        self.assertEqual(result["rows_removed"], 3)
        self.assertGreater(result["bytes_freed"], 0)
        self.assertEqual(len(t), 7)
        # After compact, the rows are 1, 3, 5, 6, 7, 8, 9 (flags bit 0 = 1 in
        # 1, 3, 5, 7, 9; bit 0 = 0 in 6, 8).
        flags_bit_zero = sorted(r["flags"] for r in t.select_bitmask({"flags": (0, 1)}))
        flags_bit_one = sorted(r["flags"] for r in t.select_bitmask({"flags": (0, 0)}))
        self.assertEqual(flags_bit_zero, [1, 3, 5, 7, 9])
        self.assertEqual(flags_bit_one, [6, 8])

    def test_compact_over_bool_column(self):
        t = ColumnarTable("t", [("active", "bool")])
        t.batch_insert([{"active": i % 2 == 0} for i in range(10)])
        t.delete(1)
        t.delete(3)
        result = t.compact()
        self.assertEqual(result["rows_after"], 8)
        self.assertEqual(len(t), 8)

    def test_compact_over_delta_column(self):
        t = ColumnarTable(
            "t",
            [("id", "i64")],
            delta_columns=["id"],
        )
        ids = [100 + i for i in range(20)]
        t.batch_insert([{"id": v} for v in ids])
        t.delete(0)
        t.delete(5)
        t.delete(10)
        result = t.compact()
        self.assertEqual(result["rows_after"], 17)
        # Verify the bitmask query still works on the rebuilt delta column.
        result = t.select_bitmask({"id": (0, 1)})
        self.assertTrue(all(int(r["id"]) & 1 for r in result))

    def test_compact_over_for_column(self):
        t = ColumnarTable(
            "t",
            [("score", "u32")],
            for_columns=["score"],
        )
        t.batch_insert([{"score": i % 50} for i in range(20)])
        t.delete(0)
        t.delete(5)
        result = t.compact()
        self.assertEqual(result["rows_after"], 18)
        self.assertEqual(len(t), 18)

    def test_compact_over_bytes_column(self):
        t = ColumnarTable("t", [("name", "bytes:16")])
        rows = [{"name": f"user-{i}".encode()} for i in range(10)]
        t.batch_insert(rows)
        for i in range(0, 8, 2):
            t.delete(i)
        result = t.compact()
        self.assertEqual(result["rows_after"], 6)
        self.assertEqual(len(t), 6)

    def test_compact_over_dict_encoded_column(self):
        t = ColumnarTable(
            "t",
            [("name", "bytes:16")],
            dict_columns=["name"],
        )
        rows = [{"name": f"u{i}".encode()} for i in range(20)]
        t.batch_insert(rows)
        for i in range(0, 8, 2):
            t.delete(i)
        result = t.compact()
        self.assertEqual(result["rows_after"], 16)
        self.assertEqual(len(t), 16)


if __name__ == "__main__":
    unittest.main()
