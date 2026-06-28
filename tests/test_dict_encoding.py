"""Test dictionary encoding for SnapDB v0.4.0"""
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


def test_dict_encoding():
    print("\n━─ Test: Dictionary encoding ──")

    schema = [
        ("id", "i32"),
        ("status", "bytes:20"),
        ("category", "bytes:20"),
        ("score", "f32"),
    ]

    # Create table with dict encoding on status and category
    db = ColumnarTable("users", schema, dict_columns=["status", "category"])

    # Insert rows with low-cardinality strings
    rows = [
        {"id": 1, "status": "active", "category": "electronics", "score": 95.5},
        {"id": 2, "status": "inactive", "category": "electronics", "score": 87.2},
        {"id": 3, "status": "active", "category": "books", "score": 92.1},
        {"id": 4, "status": "pending", "category": "electronics", "score": 78.0},
        {"id": 5, "status": "active", "category": "books", "score": 88.5},
    ]

    db.batch_insert(rows)
    test("batch_insert 5 rows", len(db) == 5)

    # Verify dict mode is active
    status_col = db.columns["status"]
    category_col = db.columns["category"]
    test("status dict mode active", status_col._dict_mode and not status_col._dict_fallback)
    test("category dict mode active", category_col._dict_mode and not category_col._dict_fallback)

    # Verify data integrity
    row0 = db.get(0)
    test("row 0 status", row0["status"] == "active")
    test("row 0 category", row0["category"] == "electronics")

    row1 = db.get(1)
    test("row 1 status", row1["status"] == "inactive")
    test("row 1 category", row1["category"] == "electronics")

    row3 = db.get(3)
    test("row 3 status", row3["status"] == "pending")
    test("row 3 category", row3["category"] == "electronics")

    # Verify unique count
    test("status unique count", status_col.unique_count() == 3)
    test("category unique count", category_col.unique_count() == 2)

    # Verify memory is lower than raw
    dict_mem = db.memory_usage()

    # Compare with non-dict table
    db2 = ColumnarTable("users2", schema)
    for row in rows:
        db2.insert(row)
    raw_mem = db2.memory_usage()

    test("dict memory < raw memory", dict_mem < raw_mem,
         f"dict={dict_mem}B raw={raw_mem}B")

    # Verify select works
    active = db.select(where=lambda r: r["status"] == "active")
    test("select active", len(active) == 3)
    test("select active row 0", active[0]["status"] == "active")

    # Verify update works with dict
    db.update(1, {"status": "active"})
    row1 = db.get(1)
    test("update status", row1["status"] == "active")

    # Verify iter_valid
    statuses = [v for _, v in status_col.iter_valid()]
    test("iter_valid", len(statuses) == 5)

    # Test dict fallback on threshold overflow
    print("\n  ━─ Test dict fallback ──")
    schema2 = [("id", "i32"), ("name", "bytes:50")]
    db3 = ColumnarTable("overflow", schema2, dict_columns=["name"], dict_threshold=3)

    # Insert 4 unique values (threshold=3, so should fallback)
    for i in range(4):
        db3.insert({"id": i, "name": f"user_{i}_very_long_name"})

    name_col = db3.columns["name"]
    test("fallback occurred", name_col._dict_fallback or not name_col._dict_mode)

    # Verify data still correct after fallback
    row = db3.get(2)
    test("fallback data correct", row["name"] == "user_2_very_long_name")


def test_dict_no_encode():
    print("\n━─ Test: Non-dict column (raw storage) ──")
    schema = [("id", "i32"), ("name", "bytes:20")]
    db = ColumnarTable("raw", schema)  # no dict_columns

    db.insert({"id": 1, "name": "alice"})
    db.insert({"id": 2, "name": "bob"})

    name_col = db.columns["name"]
    test("no dict mode", not name_col._dict_mode)
    test("raw data", db.get(0)["name"] == "alice")


if __name__ == "__main__":
    test_dict_encoding()
    test_dict_no_encode()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
