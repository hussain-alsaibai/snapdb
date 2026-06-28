"""
Test Frame-of-Reference (FOR) encoding.

v0.6.0: FOR bit packing for bounded numeric ranges.

Expected behavior:
- Samples values during threshold period
- If range fits in <=16 bits, enables FOR mode
- Stores min once, bit-packs deltas into Python int
- Falls back to raw if range too large
- Correct readback via __getitem__ and iter_valid
"""

from __future__ import annotations

import sys
sys.path.insert(0, "snapdb")

from snapdb.columnar import Column, ColumnarTable


def test_for_threshold_sampling():
    """FOR samples values and decides after threshold."""
    col = Column("score", "i32", for_encode=True, for_threshold=5)
    # Insert 4 values (below threshold) - should still be raw sampling
    for v in [10, 20, 30, 40]:
        col.append(v)
    assert not col._for_mode
    assert col._for_stats is not None
    # 5th value triggers evaluation
    col.append(15)
    # Range = 40-10 = 30, bit_length = 5 <= 16 → FOR enabled
    assert col._for_mode
    assert col._for_min == 10
    # bits needed for 30 = 5
    assert col._for_bits == 5
    # Verify readback
    assert col[0] == 10
    assert col[1] == 20
    assert col[4] == 15
    print("✅ test_for_threshold_sampling")


def test_for_memory_reduction():
    """FOR should reduce memory for bounded ranges."""
    # Without FOR: 100 i32 values = 500 bytes (including nullmask)
    col_raw = Column("score", "i32", for_encode=False)
    for v in range(100, 200):
        col_raw.append(v)
    raw_mem = col_raw.memory_usage()

    # With FOR: 100 values, range=100, bit_length=7, packed = ~700 bits = ~88 bytes
    col_for = Column("score", "i32", for_encode=True, for_threshold=50)
    for v in range(100, 200):
        col_for.append(v)
    for_mem = col_for.memory_usage()

    assert for_mem < raw_mem, f"FOR mem ({for_mem}) should be < raw mem ({raw_mem})"
    print(f"✅ test_for_memory_reduction: raw={raw_mem}B, FOR={for_mem}B ({raw_mem/for_mem:.1f}x reduction)")


def test_for_fallback_range_too_large():
    """FOR falls back to raw if range exceeds 16 bits."""
    col = Column("id", "i32", for_encode=True, for_threshold=5)
    # Range = 100000 - 0 = 100000, bit_length = 17 > 16 → fallback
    for v in [0, 25000, 50000, 75000, 100000]:
        col.append(v)
    assert not col._for_mode
    assert col._for_fallback
    # Readback should still work via raw data
    assert col[0] == 0
    assert col[4] == 100000
    print("✅ test_for_fallback_range_too_large")


def test_for_with_nulls():
    """FOR handles null values correctly."""
    col = Column("score", "i32", for_encode=True, for_threshold=3)
    col.append(10)
    col.append(None)
    col.append(20)
    col.append(15)
    assert col._for_mode
    assert col[0] == 10
    assert col[1] is None
    assert col[2] == 20
    assert col[3] == 15
    print("✅ test_for_with_nulls")


def test_for_table_integration():
    """ColumnarTable with for_columns parameter."""
    table = ColumnarTable(
        "survey",
        schema=[
            ("user_id", "i32"),
            ("age", "i32"),
            ("rating", "i32"),
        ],
        for_columns=["age", "rating"],
        for_threshold=10,
    )

    # Insert 15 rows with bounded values
    for i in range(15):
        table.insert({"user_id": 1000 + i, "age": 20 + (i % 50), "rating": (i % 5) + 1})

    # Verify FOR enabled for age and rating
    assert table.columns["age"]._for_mode or table.columns["age"]._for_fallback
    assert table.columns["rating"]._for_mode or table.columns["rating"]._for_fallback

    # Readback
    row = table.get(5)
    assert row["user_id"] == 1005
    assert row["age"] == 25
    assert row["rating"] == 1

    # Select
    results = table.select(where=lambda r: r["rating"] == 5)
    assert len(results) == 3  # rows 4, 9, 14 have rating 5

    print("✅ test_for_table_integration")


def test_for_update_fallback():
    """Update on FOR column falls back to raw."""
    col = Column("score", "i32", for_encode=True, for_threshold=3)
    col.append(10)
    col.append(20)
    col.append(15)
    assert col._for_mode
    # Update triggers fallback
    col[1] = 100
    assert col._for_fallback
    assert col[1] == 100
    assert col[0] == 10
    assert col[2] == 15
    print("✅ test_for_update_fallback")


if __name__ == "__main__":
    test_for_threshold_sampling()
    test_for_memory_reduction()
    test_for_fallback_range_too_large()
    test_for_with_nulls()
    test_for_table_integration()
    test_for_update_fallback()
    print("\n✅ All FOR encoding tests passed!")
