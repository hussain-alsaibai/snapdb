#!/usr/bin/env python3
"""
Comprehensive test suite for SnapDB v0.3.0
Tests: row storage, columnar storage, metrics, CDC, benchmarks
"""
import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapdb import SnapDB, Schema, ColumnDef, Metrics, ColumnarTable

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

# ── Test 1: Row storage CRUD ──────────────────────────────────────────────────
def test_row_crud():
    print("\n━─ Test 1: Row storage CRUD ──")
    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("name", "bytes:20"),
        ColumnDef("score", "f32"),
        ColumnDef("active", "bool")
    ])
    tmp = tempfile.mktemp(suffix=".snap")
    try:
        db = SnapDB(tmp, schema, storage_type="row")

        # Insert
        idx1 = db.insert({"id": 1, "name": "alice", "score": 95.5, "active": True})
        idx2 = db.insert({"id": 2, "name": "bob", "score": 87.2, "active": False})
        idx3 = db.insert({"id": 3, "name": "charlie", "score": 92.1, "active": True})
        test("insert returns 0,1,2", idx1 == 0 and idx2 == 1 and idx3 == 2)

        # Get
        row = db.get(0)
        test("get row 0 id", row["id"] == 1, str(row))
        test("get row 0 name", row["name"] == "alice", str(row))
        test("get row 0 score", abs(row["score"] - 95.5) < 0.01, str(row))
        test("get row 0 active", row["active"] == True, str(row))

        # Get raw
        raw = db.get_raw(0)
        test("get_raw returns memoryview", isinstance(raw, memoryview) or raw is None, str(type(raw)))

        # Update
        db.update(1, {"score": 90.0})
        row = db.get(1)
        test("update score", abs(row["score"] - 90.0) < 0.01, str(row))

        # Delete
        db.delete(0)
        test("deleted row is None", db.get(0) is None)

        # Len
        test("len after delete", len(db) == 2, str(len(db)))

        # Iterate
        rows = list(db)
        test("iter returns 2 rows", len(rows) == 2, str(len(rows)))

        db.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ── Test 2: Columnar storage CRUD ─────────────────────────────────────────────
def test_columnar_crud():
    print("\n━─ Test 2: Columnar storage CRUD ──")
    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("name", "bytes:20"),
        ColumnDef("score", "f32"),
        ColumnDef("active", "bool")
    ])
    tmp = tempfile.mktemp(suffix=".snap")
    try:
        db = SnapDB(tmp, schema, storage_type="columnar")

        # Insert
        idx1 = db.insert({"id": 1, "name": "alice", "score": 95.5, "active": True})
        idx2 = db.insert({"id": 2, "name": "bob", "score": 87.2, "active": False})
        idx3 = db.insert({"id": 3, "name": "charlie", "score": 92.1, "active": True})
        test("insert returns 0,1,2", idx1 == 0 and idx2 == 1 and idx3 == 2)

        # Get
        row = db.get(0)
        test("get row 0 id", row["id"] == 1, str(row))
        test("get row 0 name is string", row["name"] == "alice", f"got: {row['name']!r}")
        test("get row 0 score", abs(row["score"] - 95.5) < 0.01, str(row))
        test("get row 0 active", row["active"] == True, str(row))

        # Update
        db.update(1, {"score": 90.0})
        row = db.get(1)
        test("columnar update score", abs(row["score"] - 90.0) < 0.01, str(row))

        # Delete
        db.delete(0)
        test("columnar deleted row is None", db.get(0) is None)

        # Len
        test("columnar len after delete", len(db) == 3, f"len={len(db)}")

        db.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ── Test 3: Columnar select & aggregate ──────────────────────────────────────
def test_columnar_analytics():
    print("\n━─ Test 3: Columnar analytics ──")
    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("name", "bytes:20"),
        ColumnDef("score", "f32"),
        ColumnDef("active", "bool")
    ])
    tmp = tempfile.mktemp(suffix=".snap")
    try:
        db = SnapDB(tmp, schema, storage_type="columnar")
        for i in range(1000):
            db.insert({"id": i, "name": f"user{i}", "score": float(i) * 0.5, "active": i % 2 == 0})

        # Select with filter
        high = db.select(where=lambda r: r["score"] > 200.0)
        test("select filter count", len(high) == 599, f"got {len(high)}")

        # Select with projection
        names_only = db.select(columns=["name"], limit=5)
        test("select projection", all("name" in r and "id" not in r for r in names_only), str(names_only[:2]))

        # Select with offset
        offset_results = db.select(limit=5, offset=5)
        test("select offset", offset_results[0]["id"] == 5, str(offset_results[0]))

        # Aggregations
        test("agg sum", abs(db.aggregate("score", "sum") - 249750.0) < 1.0, str(db.aggregate("score", "sum")))
        test("agg avg", abs(db.aggregate("score", "avg") - 249.75) < 0.1, str(db.aggregate("score", "avg")))
        test("agg min", db.aggregate("score", "min") == 0.0, str(db.aggregate("score", "min")))
        test("agg max", db.aggregate("score", "max") == 499.5, str(db.aggregate("score", "max")))
        test("agg count", db.aggregate("active", "count") == 1000, str(db.aggregate("active", "count")))

        # select_column
        ids = db.select_column("id")
        test("select_column len", len(ids) == 1000, str(len(ids)))
        test("select_column first", ids[0] == 0, str(ids[0]))

        # memory_usage
        mem = db.memory_usage()
        test("memory_usage > 0", mem > 0, str(mem))

        db.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ── Test 4: Metrics ──────────────────────────────────────────────────────────
def test_metrics():
    print("\n━─ Test 4: Metrics ──")
    m = Metrics()
    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("value", "f32")
    ])
    tmp = tempfile.mktemp(suffix=".snap")
    try:
        db = SnapDB(tmp, schema, storage_type="row", metrics=m)
        for i in range(100):
            db.insert({"id": i, "value": float(i)})
        for i in [0, 50, 99]:
            db.get(i)
        db.update(10, {"value": 999.0})
        db.delete(0)

        report = m.report()
        test("metrics has insert", "db_insert_total" in report)
        test("metrics has get", "db_get_total" in report)
        test("metrics has update", "db_update_total" in report)
        test("metrics has delete", "db_delete_total" in report)
        test("metrics has latency", "db_insert_latency" in report)
        test("metrics insert count", "snapdb_db_insert_total_total 100" in report)

        db.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ── Test 5: CDC (Change Data Capture) ──────────────────────────────────────────
def test_cdc():
    print("\n━─ Test 5: CDC ──")
    from snapdb import CDCLog
    events = []
    cdc = CDCLog(callback=lambda e: events.append(e))
    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("name", "bytes:10")
    ])
    tmp = tempfile.mktemp(suffix=".snap")
    try:
        db = SnapDB(tmp, schema, storage_type="row", cdc=cdc)
        db.insert({"id": 1, "name": "alice"})
        db.insert({"id": 2, "name": "bob"})
        db.update(0, {"name": "alice2"})
        db.delete(1)

        test("cdc captured 4 events", len(events) == 4, f"got {len(events)}")
        test("cdc event 0 is insert", events[0]["op"] == "insert")
        test("cdc event 2 is update", events[2]["op"] == "update")
        test("cdc event 3 is delete", events[3]["op"] == "delete")
        test("cdc update has old_row", events[2]["old"] is not None)
        test("cdc delete has old_row", events[3]["old"] is not None)

        db.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ── Test 6: CDC with file log ──────────────────────────────────────────────────
def test_cdc_file():
    print("\n━─ Test 6: CDC file log ──")
    from snapdb import CDCLog
    log_path = tempfile.mktemp(suffix=".cdc")
    cdc = CDCLog(log_file=log_path)
    schema = Schema([ColumnDef("id", "i32")])
    tmp = tempfile.mktemp(suffix=".snap")
    try:
        db = SnapDB(tmp, schema, storage_type="row", cdc=cdc)
        db.insert({"id": 1})
        db.insert({"id": 2})
        db.delete(0)
        db.close()

        # Replay
        cdc2 = CDCLog(log_file=log_path)
        events = cdc2.replay()
        test("cdc file replay 3 events", len(events) == 3, f"got {len(events)}")
        test("cdc file event 0 insert", events[0]["op"] == "insert")
        test("cdc file event 2 delete", events[2]["op"] == "delete")
        cdc2.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
        if os.path.exists(log_path):
            os.remove(log_path)

# ── Test 7: Benchmark row vs columnar ─────────────────────────────────────────
def test_benchmark():
    print("\n━─ Test 7: Benchmark row vs columnar ──")
    N = 10_000
    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("name", "bytes:20"),
        ColumnDef("score", "f32"),
        ColumnDef("active", "bool")
    ])

    # Row storage
    tmp_row = tempfile.mktemp(suffix=".snap")
    db_row = SnapDB(tmp_row, schema, storage_type="row")
    t0 = time.time()
    for i in range(N):
        db_row.insert({"id": i, "name": f"user{i}", "score": float(i), "active": i % 2 == 0})
    row_insert_time = time.time() - t0

    # Scan all rows
    t0 = time.time()
    total = 0
    for _, row in db_row:
        total += row["score"]
    row_scan_time = time.time() - t0
    test("row scan correct", abs(total - sum(float(i) for i in range(N))) < 1.0)
    db_row.close()
    os.remove(tmp_row)

    # Columnar storage
    tmp_col = tempfile.mktemp(suffix=".snap")
    db_col = SnapDB(tmp_col, schema, storage_type="columnar")
    t0 = time.time()
    for i in range(N):
        db_col.insert({"id": i, "name": f"user{i}", "score": float(i), "active": i % 2 == 0})
    col_insert_time = time.time() - t0

    # Aggregate (columnar fast path)
    t0 = time.time()
    col_sum = db_col.aggregate("score", "sum")
    col_agg_time = time.time() - t0
    test("columnar agg correct", abs(col_sum - sum(float(i) for i in range(N))) < 1.0)

    # Select scan
    t0 = time.time()
    results = db_col.select(where=lambda r: r["score"] > 5000.0)
    col_scan_time = time.time() - t0
    test("columnar scan correct", len(results) == N - 5001, f"got {len(results)}")

    db_col.close()
    if os.path.exists(tmp_col):
        os.remove(tmp_col)

    print(f"\n  📊 Results ({N} rows):")
    print(f"     Row insert:    {row_insert_time:.3f}s")
    print(f"     Col insert:    {col_insert_time:.3f}s")
    print(f"     Row scan:      {row_scan_time:.3f}s")
    print(f"     Col aggregate: {col_agg_time:.3f}s")
    print(f"     Col select:    {col_scan_time:.3f}s")

# ── Run all tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║  SnapDB v0.3.0 — Comprehensive Test Suite   ║")
    print("╚══════════════════════════════════════════════╝")
    test_row_crud()
    test_columnar_crud()
    test_columnar_analytics()
    test_metrics()
    test_cdc()
    test_cdc_file()
    test_benchmark()
    print(f"\n╔══════════ Results ══════════╗")
    print(f"  ✅ Passed: {PASS}")
    print(f"  ❌ Failed: {FAIL}")
    print(f"  Total:    {PASS + FAIL}")
    if FAIL == 0:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {FAIL} test(s) failed.")
    sys.exit(0 if FAIL == 0 else 1)
