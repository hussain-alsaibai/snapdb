"""Test delta encoding for SnapDB v0.5.0"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapdb import ColumnarTable

PASS = 0
FAIL = 0

def test(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def test_delta_encoding():
    print("\n━─ Test: Delta encoding for monotonic columns ──")

    schema = [
        ("id", "i32"),
        ("timestamp", "i64"),
        ("seq", "u32"),
        ("value", "f32"),
    ]

    # Create table with delta encoding on timestamp and seq
    db = ColumnarTable("events", schema, delta_columns=["timestamp", "seq"])

    # Insert monotonic data
    base_ts = 1700000000000  # some epoch ms
    rows = []
    for i in range(100):
        rows.append({
            "id": i,
            "timestamp": base_ts + i * 1000,  # +1s each row
            "seq": i,
            "value": float(i * 10.0),
        })

    db.batch_insert(rows)
    test("batch_insert 100 rows", len(db) == 100)

    # Verify delta mode is active
    ts_col = db.columns["timestamp"]
    seq_col = db.columns["seq"]
    test("timestamp delta mode active", ts_col._delta_mode and not ts_col._delta_fallback)
    test("seq delta mode active", seq_col._delta_mode and not seq_col._delta_fallback)

    # Verify data integrity
    row0 = db.get(0)
    test("row 0 timestamp", row0["timestamp"] == base_ts)
    test("row 0 seq", row0["seq"] == 0)

    row50 = db.get(50)
    test("row 50 timestamp", row50["timestamp"] == base_ts + 50 * 1000)
    test("row 50 seq", row50["seq"] == 50)

    row99 = db.get(99)
    test("row 99 timestamp", row99["timestamp"] == base_ts + 99 * 1000)
    test("row 99 seq", row99["seq"] == 99)

    # Verify non-delta column
    id_col = db.columns["id"]
    test("id not delta mode", not id_col._delta_mode)

    # Verify memory is lower than raw
    delta_mem = db.memory_usage()

    # Compare with non-delta table
    db2 = ColumnarTable("events2", schema)
    for row in rows:
        db2.insert(row)
    raw_mem = db2.memory_usage()

    test("delta memory < raw memory", delta_mem < raw_mem,
         f"delta={delta_mem}B raw={raw_mem}B")

    # Verify select works
    result = db.select(where=lambda r: r["seq"] > 50)
    test("select seq > 50", len(result) == 49)
    test("select correct", result[0]["seq"] == 51)

    # Verify iter_valid
    ts_vals = [v for _, v in ts_col.iter_valid()]
    test("iter_valid timestamps", len(ts_vals) == 100)
    test("iter_valid first", ts_vals[0] == base_ts)


def test_delta_fallback():
    print("\n━─ Test: Delta encoding fallback (non-monotonic) ──")

    schema = [("id", "u32"), ("score", "i32")]
    db = ColumnarTable("scores", schema, delta_columns=["score"])

    # Insert 60 rows with non-monotonic data (will trigger fallback after 50 samples)
    for i in range(60):
        # Alternating up and down
        db.insert({"id": i, "score": i if i % 2 == 0 else i - 5})

    score_col = db.columns["score"]
    test("fallback occurred", score_col._delta_fallback)

    # Verify data still correct after fallback
    for i in range(60):
        row = db.get(i)
        expected = i if i % 2 == 0 else i - 5
        test(f"fallback data row {i}", row["score"] == expected, f"got {row['score']} expected {expected}")


def test_delta_not_triggered():
    print("\n━─ Test: Delta not triggered (too few rows) ──")

    schema = [("ts", "i64")]
    db = ColumnarTable("small", schema, delta_columns=["ts"])

    # Insert only 10 rows (less than default delta_samples=50)
    for i in range(10):
        db.insert({"ts": 1000 + i * 100})

    ts_col = db.columns["ts"]
    # With only 10 rows, still in sampling phase (not delta yet)
    # Actually, we need to check - with the current implementation,
    # delta mode is enabled after reaching delta_samples
    test("still sampling", not ts_col._delta_mode)
    test("not fallback", not ts_col._delta_fallback)
    test("data correct", db.get(5)["ts"] == 1500)


if __name__ == "__main__":
    test_delta_encoding()
    test_delta_fallback()
    test_delta_not_triggered()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
