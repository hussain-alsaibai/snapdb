#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for SnapDB v0.3.1
Compares: SnapDB (row), SnapDB (columnar), SQLite (in-memory), DuckDB (in-memory), pure dict

Workloads:
  1. Bulk insert (10K, 50K, 100K rows)
  2. Point lookup (random by ID)
  3. Full scan (sum aggregation)
  4. Filtered scan (WHERE clause equivalent)
  5. Update (random rows)
  6. Delete (random rows)
  7. Memory usage comparison

Measures: throughput (ops/sec), latency (ms), total time (s)
"""

import sys
import os
import time
import json
import sqlite3
import tempfile
import tracemalloc
from typing import Any, Dict, List, Tuple, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snapdb import SnapDB, Schema, ColumnDef
from columnar import ColumnarTable

try:
    import duckdb
    import pandas as pd
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    print("⚠️  duckdb not installed — skipping DuckDB benchmarks")

# ── Configuration ──────────────────────────────────────────────────────────────

SIZES = [10_000, 50_000, 100_000]
LOOKUPS = 10_000
UPDATES = 5_000
DELETES = 2_000

COLUMNS = [
    ("id", "i32"),
    ("name", "bytes:20"),
    ("score", "f32"),
    ("active", "bool"),
]

def make_row(i: int) -> Dict[str, Any]:
    return {
        "id": i,
        "name": f"user_{i:06d}",
        "score": float(i) * 0.5,
        "active": i % 3 == 0,
    }

def make_rows(n: int) -> List[Dict[str, Any]]:
    return [make_row(i) for i in range(n)]

# ── Timing helpers ─────────────────────────────────────────────────────────────

def fmt_ops_sec(elapsed: float, count: int) -> str:
    if elapsed == 0:
        return "inf"
    return f"{count / elapsed:,.0f}"

def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}ms"

# ── SnapDB Row ─────────────────────────────────────────────────────────────────

def snapdb_row_insert(n: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="row")
    t0 = time.perf_counter()
    for i in range(n):
        db.insert(make_row(i))
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

def snapdb_row_batch_insert(n: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="row")
    rows = make_rows(n)
    t0 = time.perf_counter()
    db.batch_insert(rows)
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

def snapdb_row_get(n: int, lookups: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="row")
    for i in range(n):
        db.insert(make_row(i))
    t0 = time.perf_counter()
    for i in range(lookups):
        idx = (i * 7919) % n
        db.get(idx)
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

def snapdb_row_scan(n: int) -> Tuple[float, float]:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="row")
    for i in range(n):
        db.insert(make_row(i))
    t0 = time.perf_counter()
    total = 0.0
    for _, row in db:
        total += row["score"]
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed, total

def snapdb_row_update(n: int, updates: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="row")
    for i in range(n):
        db.insert(make_row(i))
    t0 = time.perf_counter()
    for i in range(updates):
        idx = (i * 7919) % n
        db.update(idx, {"score": float(i) * 1.5})
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

def snapdb_row_delete(n: int, deletes: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="row")
    for i in range(n):
        db.insert(make_row(i))
    t0 = time.perf_counter()
    for i in range(deletes):
        idx = (i * 7919) % n
        try:
            db.delete(idx)
        except (KeyError, IndexError):
            pass
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

# ── SnapDB Columnar ────────────────────────────────────────────────────────────

def snapdb_col_insert(n: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="columnar")
    t0 = time.perf_counter()
    for i in range(n):
        db.insert(make_row(i))
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

def snapdb_col_batch_insert(n: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="columnar")
    rows = make_rows(n)
    t0 = time.perf_counter()
    db.batch_insert(rows)
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

def snapdb_col_get(n: int, lookups: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="columnar")
    for i in range(n):
        db.insert(make_row(i))
    t0 = time.perf_counter()
    for i in range(lookups):
        idx = (i * 7919) % n
        db.get(idx)
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

def snapdb_col_scan(n: int) -> Tuple[float, float]:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="columnar")
    for i in range(n):
        db.insert(make_row(i))
    t0 = time.perf_counter()
    total = db.aggregate("score", "sum")
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed, total

def snapdb_col_select(n: int) -> Tuple[float, int]:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="columnar")
    for i in range(n):
        db.insert(make_row(i))
    threshold = float(n) * 0.25
    t0 = time.perf_counter()
    results = db.select(where=lambda r: r["score"] > threshold)
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed, len(results)

def snapdb_col_update(n: int, updates: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="columnar")
    for i in range(n):
        db.insert(make_row(i))
    t0 = time.perf_counter()
    for i in range(updates):
        idx = (i * 7919) % n
        db.update(idx, {"score": float(i) * 1.5})
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

def snapdb_col_delete(n: int, deletes: int) -> float:
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="columnar")
    for i in range(n):
        db.insert(make_row(i))
    t0 = time.perf_counter()
    for i in range(deletes):
        idx = (i * 7919) % n
        try:
            db.delete(idx)
        except (KeyError, IndexError):
            pass
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

# ── SQLite In-Memory ───────────────────────────────────────────────────────────

def sqlite_insert_batch(n: int) -> float:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT, score REAL, active INTEGER)")
    rows = [(i, f"user_{i:06d}", float(i) * 0.5, 1 if i % 3 == 0 else 0) for i in range(n)]
    t0 = time.perf_counter()
    conn.execute("BEGIN")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", rows)
    conn.execute("COMMIT")
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

def sqlite_get(n: int, lookups: int) -> float:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT, score REAL, active INTEGER)")
    conn.execute("BEGIN")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)",
                     [(i, f"user_{i:06d}", float(i) * 0.5, 1 if i % 3 == 0 else 0) for i in range(n)])
    conn.execute("COMMIT")
    conn.execute("CREATE INDEX idx_id ON t(id)")
    t0 = time.perf_counter()
    for i in range(lookups):
        idx = (i * 7919) % n
        conn.execute("SELECT * FROM t WHERE id = ?", (idx,)).fetchone()
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

def sqlite_scan(n: int) -> Tuple[float, float]:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT, score REAL, active INTEGER)")
    conn.execute("BEGIN")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)",
                     [(i, f"user_{i:06d}", float(i) * 0.5, 1 if i % 3 == 0 else 0) for i in range(n)])
    conn.execute("COMMIT")
    t0 = time.perf_counter()
    row = conn.execute("SELECT SUM(score) FROM t").fetchone()
    total = row[0] if row else 0.0
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed, total

def sqlite_select(n: int) -> Tuple[float, int]:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT, score REAL, active INTEGER)")
    conn.execute("BEGIN")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)",
                     [(i, f"user_{i:06d}", float(i) * 0.5, 1 if i % 3 == 0 else 0) for i in range(n)])
    conn.execute("COMMIT")
    threshold = float(n) * 0.25
    t0 = time.perf_counter()
    results = conn.execute("SELECT COUNT(*) FROM t WHERE score > ?", (threshold,)).fetchone()
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed, results[0] if results else 0

def sqlite_update(n: int, updates: int) -> float:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT, score REAL, active INTEGER)")
    conn.execute("BEGIN")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)",
                     [(i, f"user_{i:06d}", float(i) * 0.5, 1 if i % 3 == 0 else 0) for i in range(n)])
    conn.execute("COMMIT")
    conn.execute("CREATE INDEX idx_id ON t(id)")
    t0 = time.perf_counter()
    conn.execute("BEGIN")
    for i in range(updates):
        idx = (i * 7919) % n
        conn.execute("UPDATE t SET score = ? WHERE id = ?", (float(i) * 1.5, idx))
    conn.execute("COMMIT")
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

def sqlite_delete(n: int, deletes: int) -> float:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT, score REAL, active INTEGER)")
    conn.execute("BEGIN")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)",
                     [(i, f"user_{i:06d}", float(i) * 0.5, 1 if i % 3 == 0 else 0) for i in range(n)])
    conn.execute("COMMIT")
    conn.execute("CREATE INDEX idx_id ON t(id)")
    t0 = time.perf_counter()
    conn.execute("BEGIN")
    for i in range(deletes):
        idx = (i * 7919) % n
        conn.execute("DELETE FROM t WHERE id = ?", (idx,))
    conn.execute("COMMIT")
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

# ── DuckDB In-Memory ────────────────────────────────────────────────────────────

def _duckdb_create(n: int):
    """Create a DuckDB in-memory table with n rows. Returns (conn, n)."""
    conn = duckdb.connect(":memory:")
    conn.execute("PRAGMA threads=1")
    conn.execute("CREATE TABLE t (id INTEGER, name VARCHAR, score FLOAT, active BOOLEAN)")
    df = pd.DataFrame({
        "id": list(range(n)),
        "name": [f"user_{i:06d}" for i in range(n)],
        "score": [float(i) * 0.5 for i in range(n)],
        "active": [i % 3 == 0 for i in range(n)],
    })
    conn.register("src_df", df)
    conn.execute("INSERT INTO t SELECT * FROM src_df")
    return conn

def duckdb_insert(n: int) -> float:
    """DuckDB bulk insert via DataFrame registration."""
    conn = duckdb.connect(":memory:")
    conn.execute("PRAGMA threads=1")
    conn.execute("CREATE TABLE t (id INTEGER, name VARCHAR, score FLOAT, active BOOLEAN)")
    df = pd.DataFrame({
        "id": list(range(n)),
        "name": [f"user_{i:06d}" for i in range(n)],
        "score": [float(i) * 0.5 for i in range(n)],
        "active": [i % 3 == 0 for i in range(n)],
    })
    conn.register("src_df", df)
    t0 = time.perf_counter()
    conn.execute("INSERT INTO t SELECT * FROM src_df")
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

def duckdb_insert_executemany(n: int) -> float:
    """DuckDB executemany (slower path, for comparison)."""
    conn = duckdb.connect(":memory:")
    conn.execute("PRAGMA threads=1")
    conn.execute("CREATE TABLE t (id INTEGER, name VARCHAR, score FLOAT, active BOOLEAN)")
    rows = [(i, f"user_{i:06d}", float(i) * 0.5, i % 3 == 0) for i in range(n)]
    t0 = time.perf_counter()
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", rows)
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

def duckdb_get(n: int, lookups: int) -> float:
    conn = _duckdb_create(n)
    conn.execute("CREATE INDEX idx_id ON t(id)")
    t0 = time.perf_counter()
    for i in range(lookups):
        idx = (i * 7919) % n
        conn.execute("SELECT * FROM t WHERE id = ?", [idx]).fetchone()
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

def duckdb_scan(n: int) -> Tuple[float, float]:
    conn = _duckdb_create(n)
    t0 = time.perf_counter()
    row = conn.execute("SELECT SUM(score) FROM t").fetchone()
    total = row[0] if row else 0.0
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed, total

def duckdb_select(n: int) -> Tuple[float, int]:
    conn = _duckdb_create(n)
    threshold = float(n) * 0.25
    t0 = time.perf_counter()
    results = conn.execute("SELECT COUNT(*) FROM t WHERE score > ?", [threshold]).fetchone()
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed, results[0] if results else 0

def duckdb_update(n: int, updates: int) -> float:
    conn = _duckdb_create(n)
    conn.execute("CREATE INDEX idx_id ON t(id)")
    t0 = time.perf_counter()
    conn.execute("BEGIN")
    for i in range(updates):
        idx = (i * 7919) % n
        conn.execute("UPDATE t SET score = ? WHERE id = ?", [float(i) * 1.5, idx])
    conn.execute("COMMIT")
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

def duckdb_delete(n: int, deletes: int) -> float:
    conn = _duckdb_create(n)
    conn.execute("CREATE INDEX idx_id ON t(id)")
    t0 = time.perf_counter()
    conn.execute("BEGIN")
    for i in range(deletes):
        idx = (i * 7919) % n
        conn.execute("DELETE FROM t WHERE id = ?", [idx])
    conn.execute("COMMIT")
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

# ── Pure Dict (baseline) ───────────────────────────────────────────────────────

def dict_insert(n: int) -> float:
    store: Dict[int, Dict[str, Any]] = {}
    t0 = time.perf_counter()
    for i in range(n):
        store[i] = make_row(i)
    elapsed = time.perf_counter() - t0
    return elapsed

def dict_get(n: int, lookups: int) -> float:
    store: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        store[i] = make_row(i)
    t0 = time.perf_counter()
    for i in range(lookups):
        idx = (i * 7919) % n
        _ = store.get(idx)
    elapsed = time.perf_counter() - t0
    return elapsed

def dict_scan(n: int) -> Tuple[float, float]:
    store: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        store[i] = make_row(i)
    t0 = time.perf_counter()
    total = sum(v["score"] for v in store.values())
    elapsed = time.perf_counter() - t0
    return elapsed, total

def dict_select(n: int) -> Tuple[float, int]:
    store: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        store[i] = make_row(i)
    threshold = float(n) * 0.25
    t0 = time.perf_counter()
    count = sum(1 for v in store.values() if v["score"] > threshold)
    elapsed = time.perf_counter() - t0
    return elapsed, count

def dict_update(n: int, updates: int) -> float:
    store: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        store[i] = make_row(i)
    t0 = time.perf_counter()
    for i in range(updates):
        idx = (i * 7919) % n
        if idx in store:
            store[idx]["score"] = float(i) * 1.5
    elapsed = time.perf_counter() - t0
    return elapsed

def dict_delete(n: int, deletes: int) -> float:
    store: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        store[i] = make_row(i)
    t0 = time.perf_counter()
    for i in range(deletes):
        idx = (i * 7919) % n
        store.pop(idx, None)
    elapsed = time.perf_counter() - t0
    return elapsed

# ── Memory measurement ─────────────────────────────────────────────────────────

def measure_memory(n: int) -> Dict[str, int]:
    """Measure memory usage of each storage engine."""
    results: Dict[str, int] = {}

    # SnapDB row
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    tracemalloc.start()
    db = SnapDB(tmp, schema, storage_type="row")
    for i in range(n):
        db.insert(make_row(i))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["snapdb_row"] = peak
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)

    # SnapDB columnar
    tmp = tempfile.mktemp(suffix=".snap")
    tracemalloc.start()
    db = SnapDB(tmp, schema, storage_type="columnar")
    for i in range(n):
        db.insert(make_row(i))
    results["snapdb_col"] = db.memory_usage()
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)

    # SQLite
    tracemalloc.start()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT, score REAL, active INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)",
                     [(i, f"user_{i:06d}", float(i) * 0.5, 1 if i % 3 == 0 else 0) for i in range(n)])
    conn.execute("SELECT * FROM t").fetchall()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["sqlite"] = peak
    conn.close()

    # DuckDB
    if HAS_DUCKDB:
        tracemalloc.start()
        conn = _duckdb_create(n)
        conn.execute("SELECT * FROM t").fetchall()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results["duckdb"] = peak
        conn.close()

    # Dict
    tracemalloc.start()
    store: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        store[i] = make_row(i)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["dict"] = peak

    return results

# ── Main benchmark runner ─────────────────────────────────────────────────────

def run_benchmarks():
    engines = ["SnapDB Row", "SnapDB Row (batch)", "SnapDB Col", "SnapDB Col (batch)",
               "SQLite (batch)", "DuckDB (DataFrame)", "Pure Dict"]

    print("=" * 90)
    print("  SnapDB v0.3.1 — Comprehensive Benchmark Suite")
    print("  SnapDB Row | SnapDB Columnar | SQLite | DuckDB | Pure Dict")
    print("=" * 90)

    for n in SIZES:
        print(f"\n{'─' * 90}")
        print(f"  📊 Dataset: {n:,} rows | Lookups: {LOOKUPS:,} | Updates: {UPDATES:,} | Deletes: {DELETES:,}")
        print(f"{'─' * 90}")

        # ── Insert ──
        print(f"\n  1️⃣  INSERT ({n:,} rows)")
        results = {
            "SnapDB Row": snapdb_row_insert(n),
            "SnapDB Row (batch)": snapdb_row_batch_insert(n),
            "SnapDB Col": snapdb_col_insert(n),
            "SnapDB Col (batch)": snapdb_col_batch_insert(n),
            "SQLite (batch)": sqlite_insert_batch(n),
        }
        if HAS_DUCKDB:
            results["DuckDB (DataFrame)"] = duckdb_insert(n)
        results["Pure Dict"] = dict_insert(n)

        for label, elapsed in results.items():
            print(f"     {label:25s} {fmt_ops_sec(elapsed, n):>12} ops/sec  ({elapsed:.3f}s)")

        # ── Point Lookup ──
        print(f"\n  2️⃣  POINT LOOKUP ({LOOKUPS:,} random gets)")
        results = {
            "SnapDB Row": snapdb_row_get(n, LOOKUPS),
            "SnapDB Col": snapdb_col_get(n, LOOKUPS),
            "SQLite (idx)": sqlite_get(n, LOOKUPS),
        }
        if HAS_DUCKDB:
            results["DuckDB (idx)"] = duckdb_get(n, LOOKUPS)
        results["Pure Dict"] = dict_get(n, LOOKUPS)

        for label, elapsed in results.items():
            print(f"     {label:25s} {fmt_ops_sec(elapsed, LOOKUPS):>12} ops/sec  ({fmt_ms(elapsed)})")

        # ── Full Scan / Aggregate ──
        print(f"\n  3️⃣  AGGREGATE SUM (full scan, {n:,} rows)")
        t_row, s_row = snapdb_row_scan(n)
        t_col, s_col = snapdb_col_scan(n)
        t_sqlite, s_sqlite = sqlite_scan(n)
        t_dict, s_dict = dict_scan(n)

        expected = sum(float(i) * 0.5 for i in range(n))
        correct = all(abs(s - expected) < 1.0 for s in [s_row, s_col, s_sqlite, s_dict])

        print(f"     {'SnapDB Row':25s} {fmt_ops_sec(t_row, n):>12} rows/sec ({t_row:.3f}s)")
        print(f"     {'SnapDB Col':25s} {fmt_ops_sec(t_col, n):>12} rows/sec ({t_col:.3f}s)")
        print(f"     {'SQLite':25s} {fmt_ops_sec(t_sqlite, n):>12} rows/sec ({t_sqlite:.3f}s)")
        if HAS_DUCKDB:
            t_duck, s_duck = duckdb_scan(n)
            correct = correct and abs(s_duck - expected) < 1.0
            print(f"     {'DuckDB':25s} {fmt_ops_sec(t_duck, n):>12} rows/sec ({t_duck:.3f}s)")
        print(f"     {'Pure Dict':25s} {fmt_ops_sec(t_dict, n):>12} rows/sec ({t_dict:.3f}s)")
        print(f"     ✅ Correctness:  {'PASS' if correct else 'FAIL'}")

        # ── Filtered Scan ──
        print(f"\n  4️⃣  FILTERED SELECT (score > threshold, {n:,} rows)")
        t_col, count_col = snapdb_col_select(n)
        t_sqlite, count_sqlite = sqlite_select(n)
        t_dict, count_dict = dict_select(n)

        print(f"     {'SnapDB Col':25s} {fmt_ops_sec(t_col, n):>12} rows/sec ({t_col:.3f}s)  matches={count_col}")
        print(f"     {'SQLite':25s} {fmt_ops_sec(t_sqlite, n):>12} rows/sec ({t_sqlite:.3f}s)  matches={count_sqlite}")
        if HAS_DUCKDB:
            t_duck, count_duck = duckdb_select(n)
            print(f"     {'DuckDB':25s} {fmt_ops_sec(t_duck, n):>12} rows/sec ({t_duck:.3f}s)  matches={count_duck}")
        print(f"     {'Pure Dict':25s} {fmt_ops_sec(t_dict, n):>12} rows/sec ({t_dict:.3f}s)  matches={count_dict}")

        # ── Update ──
        print(f"\n  5️⃣  UPDATE ({UPDATES:,} random rows)")
        results = {
            "SnapDB Row": snapdb_row_update(n, UPDATES),
            "SnapDB Col": snapdb_col_update(n, UPDATES),
            "SQLite (idx)": sqlite_update(n, UPDATES),
        }
        if HAS_DUCKDB:
            results["DuckDB (idx)"] = duckdb_update(n, UPDATES)
        results["Pure Dict"] = dict_update(n, UPDATES)

        for label, elapsed in results.items():
            print(f"     {label:25s} {fmt_ops_sec(elapsed, UPDATES):>12} ops/sec  ({fmt_ms(elapsed)})")

        # ── Delete ──
        print(f"\n  6️⃣  DELETE ({DELETES:,} random rows)")
        results = {
            "SnapDB Row": snapdb_row_delete(n, DELETES),
            "SnapDB Col": snapdb_col_delete(n, DELETES),
            "SQLite (idx)": sqlite_delete(n, DELETES),
        }
        if HAS_DUCKDB:
            results["DuckDB (idx)"] = duckdb_delete(n, DELETES)
        results["Pure Dict"] = dict_delete(n, DELETES)

        for label, elapsed in results.items():
            print(f"     {label:25s} {fmt_ops_sec(elapsed, DELETES):>12} ops/sec  ({fmt_ms(elapsed)})")

    # ── Memory Usage ──
    print(f"\n  7️⃣  MEMORY USAGE ({SIZES[1]:,} rows)")
    mem = measure_memory(SIZES[1])
    print(f"     {'SnapDB Row':25s} {mem['snapdb_row'] / 1024 / 1024:>10.1f} MB")
    print(f"     {'SnapDB Col':25s} {mem['snapdb_col'] / 1024 / 1024:>10.1f} MB")
    print(f"     {'SQLite':25s} {mem['sqlite'] / 1024 / 1024:>10.1f} MB")
    if HAS_DUCKDB:
        print(f"     {'DuckDB':25s} {mem['duckdb'] / 1024 / 1024:>10.1f} MB")
    print(f"     {'Pure Dict':25s} {mem['dict'] / 1024 / 1024:>10.1f} MB")

    print(f"\n{'=' * 90}")
    print("  ✅ Benchmark complete.")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    run_benchmarks()
