#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for SnapDB v0.3.0
Compares: SnapDB (row), SnapDB (columnar), SQLite (in-memory), pure dict

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

# ── Configuration ──────────────────────────────────────────────────────────────

SIZES = [10_000, 50_000, 100_000]
WARMUP = 1_000

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

# ── Timing helpers ─────────────────────────────────────────────────────────────

def bench(label: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Run a function and return timing stats."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "elapsed": elapsed,
        "result": result,
    }

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
    """Row storage with batch_insert"""
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="row")
    rows = [make_row(i) for i in range(n)]
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
    # Random lookups (deterministic pattern)
    t0 = time.perf_counter()
    for i in range(lookups):
        idx = (i * 7919) % n  # prime stride for pseudo-random
        db.get(idx)
    elapsed = time.perf_counter() - t0
    db.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    return elapsed

def snapdb_row_scan(n: int) -> Tuple[float, float]:
    """Returns (scan_time, sum)"""
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
    """Columnar with batch_insert"""
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="columnar")
    rows = [make_row(i) for i in range(n)]
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
    """Columnar aggregate (fast path)"""
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
    """Filtered select: score > threshold"""
    schema = Schema([ColumnDef(*c) for c in COLUMNS])
    tmp = tempfile.mktemp(suffix=".snap")
    db = SnapDB(tmp, schema, storage_type="columnar")
    for i in range(n):
        db.insert(make_row(i))
    threshold = float(n) * 0.25  # 75% pass
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

def sqlite_insert(n: int) -> float:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT, score REAL, active INTEGER)")
    t0 = time.perf_counter()
    conn.execute("BEGIN")
    for i in range(n):
        conn.execute("INSERT INTO t VALUES (?, ?, ?, ?)",
                     (i, f"user_{i:06d}", float(i) * 0.5, 1 if i % 3 == 0 else 0))
    conn.execute("COMMIT")
    elapsed = time.perf_counter() - t0
    conn.close()
    return elapsed

def sqlite_insert_batch(n: int) -> float:
    """SQLite with batch insert (executemany)"""
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
    print("=" * 80)
    print("  SnapDB v0.3.0 — Comprehensive Benchmark Suite")
    print("  Comparing: SnapDB Row | SnapDB Columnar | SQLite (in-memory) | Pure Dict")
    print("=" * 80)

    lookups = 10_000
    updates = 5_000
    deletes = 2_000

    for n in SIZES:
        print(f"\n{'─' * 80}")
        print(f"  📊 Dataset: {n:,} rows | Lookups: {lookups:,} | Updates: {updates:,} | Deletes: {deletes:,}")
        print(f"{'─' * 80}")

        # ── Insert ──
        print(f"\n  1️⃣  INSERT ({n:,} rows)")
        t_snap_row = snapdb_row_insert(n)
        t_snap_row_b = snapdb_row_batch_insert(n)
        t_snap_col = snapdb_col_insert(n)
        t_snap_col_b = snapdb_col_batch_insert(n)
        t_sqlite = sqlite_insert_batch(n)
        t_dict = dict_insert(n)

        print(f"     SnapDB Row:        {fmt_ops_sec(t_snap_row, n):>12} ops/sec  ({t_snap_row:.3f}s)")
        print(f"     SnapDB Row(batch): {fmt_ops_sec(t_snap_row_b, n):>12} ops/sec  ({t_snap_row_b:.3f}s)")
        print(f"     SnapDB Col:        {fmt_ops_sec(t_snap_col, n):>12} ops/sec  ({t_snap_col:.3f}s)")
        print(f"     SnapDB Col(batch): {fmt_ops_sec(t_snap_col_b, n):>12} ops/sec  ({t_snap_col_b:.3f}s)")
        print(f"     SQLite (batch):    {fmt_ops_sec(t_sqlite, n):>12} ops/sec  ({t_sqlite:.3f}s)")
        print(f"     Pure Dict:         {fmt_ops_sec(t_dict, n):>12} ops/sec  ({t_dict:.3f}s)")

        # ── Point Lookup ──
        print(f"\n  2️⃣  POINT LOOKUP ({lookups:,} random gets)")
        t_snap_row = snapdb_row_get(n, lookups)
        t_snap_col = snapdb_col_get(n, lookups)
        t_sqlite = sqlite_get(n, lookups)
        t_dict = dict_get(n, lookups)

        print(f"     SnapDB Row:      {fmt_ops_sec(t_snap_row, lookups):>12} ops/sec  ({fmt_ms(t_snap_row)})")
        print(f"     SnapDB Col:      {fmt_ops_sec(t_snap_col, lookups):>12} ops/sec  ({fmt_ms(t_snap_col)})")
        print(f"     SQLite (idx):    {fmt_ops_sec(t_sqlite, lookups):>12} ops/sec  ({fmt_ms(t_sqlite)})")
        print(f"     Pure Dict:       {fmt_ops_sec(t_dict, lookups):>12} ops/sec  ({fmt_ms(t_dict)})")

        # ── Full Scan / Aggregate ──
        print(f"\n  3️⃣  AGGREGATE SUM (full scan, {n:,} rows)")
        t_snap_row, s_row = snapdb_row_scan(n)
        t_snap_col, s_col = snapdb_col_scan(n)
        t_sqlite, s_sqlite = sqlite_scan(n)
        t_dict, s_dict = dict_scan(n)

        # Verify correctness
        expected = sum(float(i) * 0.5 for i in range(n))
        correct = all(abs(s - expected) < 1.0 for s in [s_row, s_col, s_sqlite, s_dict])

        print(f"     SnapDB Row:      {fmt_ops_sec(t_snap_row, n):>12} rows/sec ({t_snap_row:.3f}s)  sum={s_row:.1f}")
        print(f"     SnapDB Col:      {fmt_ops_sec(t_snap_col, n):>12} rows/sec ({t_snap_col:.3f}s)  sum={s_col:.1f}")
        print(f"     SQLite:          {fmt_ops_sec(t_sqlite, n):>12} rows/sec ({t_sqlite:.3f}s)  sum={s_sqlite:.1f}")
        print(f"     Pure Dict:       {fmt_ops_sec(t_dict, n):>12} rows/sec ({t_dict:.3f}s)  sum={s_dict:.1f}")
        print(f"     ✅ Correctness:  {'PASS' if correct else 'FAIL'}")

        # ── Filtered Scan ──
        print(f"\n  4️⃣  FILTERED SELECT (score > threshold, {n:,} rows)")
        t_snap_col, count_col = snapdb_col_select(n)
        t_sqlite, count_sqlite = sqlite_select(n)
        t_dict, count_dict = dict_select(n)

        print(f"     SnapDB Col:      {fmt_ops_sec(t_snap_col, n):>12} rows/sec ({t_snap_col:.3f}s)  matches={count_col}")
        print(f"     SQLite:          {fmt_ops_sec(t_sqlite, n):>12} rows/sec ({t_sqlite:.3f}s)  matches={count_sqlite}")
        print(f"     Pure Dict:       {fmt_ops_sec(t_dict, n):>12} rows/sec ({t_dict:.3f}s)  matches={count_dict}")

        # ── Update ──
        print(f"\n  5️⃣  UPDATE ({updates:,} random rows)")
        t_snap_row = snapdb_row_update(n, updates)
        t_snap_col = snapdb_col_update(n, updates)
        t_sqlite = sqlite_update(n, updates)
        t_dict = dict_update(n, updates)

        print(f"     SnapDB Row:      {fmt_ops_sec(t_snap_row, updates):>12} ops/sec  ({fmt_ms(t_snap_row)})")
        print(f"     SnapDB Col:      {fmt_ops_sec(t_snap_col, updates):>12} ops/sec  ({fmt_ms(t_snap_col)})")
        print(f"     SQLite (idx):    {fmt_ops_sec(t_sqlite, updates):>12} ops/sec  ({fmt_ms(t_sqlite)})")
        print(f"     Pure Dict:       {fmt_ops_sec(t_dict, updates):>12} ops/sec  ({fmt_ms(t_dict)})")

        # ── Delete ──
        print(f"\n  6️⃣  DELETE ({deletes:,} random rows)")
        t_snap_row = snapdb_row_delete(n, deletes)
        t_snap_col = snapdb_col_delete(n, deletes)
        t_sqlite = sqlite_delete(n, deletes)
        t_dict = dict_delete(n, deletes)

        print(f"     SnapDB Row:      {fmt_ops_sec(t_snap_row, deletes):>12} ops/sec  ({fmt_ms(t_snap_row)})")
        print(f"     SnapDB Col:      {fmt_ops_sec(t_snap_col, deletes):>12} ops/sec  ({fmt_ms(t_snap_col)})")
        print(f"     SQLite (idx):    {fmt_ops_sec(t_sqlite, deletes):>12} ops/sec  ({fmt_ms(t_sqlite)})")
        print(f"     Pure Dict:       {fmt_ops_sec(t_dict, deletes):>12} ops/sec  ({fmt_ms(t_dict)})")

    # ── Memory Usage ──
    print(f"\n  7️⃣  MEMORY USAGE ({SIZES[1]:,} rows)")
    mem = measure_memory(SIZES[1])
    print(f"     SnapDB Row:      {mem['snapdb_row'] / 1024 / 1024:>10.1f} MB")
    print(f"     SnapDB Col:      {mem['snapdb_col'] / 1024 / 1024:>10.1f} MB")
    print(f"     SQLite:          {mem['sqlite'] / 1024 / 1024:>10.1f} MB")
    print(f"     Pure Dict:       {mem['dict'] / 1024 / 1024:>10.1f} MB")

    print(f"\n{'=' * 80}")
    print("  ✅ Benchmark complete.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_benchmarks()
