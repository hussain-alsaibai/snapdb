"""
SnapDB Benchmark — Prove it's lightning fast

Usage: python benchmark.py

Tests:
1. Insert speed (rows/sec)
2. Random read speed (rows/sec) — zero-copy vs decoded
3. Sequential scan speed (rows/sec)
4. Memory footprint
5. Comparison with dict-of-dicts baseline
"""

import random
import struct
import time
import tracemalloc
from pathlib import Path

from snapdb import SnapDB, Schema, ColumnDef


N = 1_000_000  # rows for benchmark
WARMUP = 1000


def benchmark_insert(db, n):
    start = time.perf_counter()
    for i in range(n):
        db.insert({
            "id": i,
            "sensor_id": i % 256,
            "temperature": random.gauss(25.0, 5.0),
            "humidity": random.gauss(60.0, 10.0),
            "active": i % 7 == 0,
        })
    elapsed = time.perf_counter() - start
    return n / elapsed


def benchmark_random_read(db, n):
    max_idx = len(db) - 1
    start = time.perf_counter()
    for _ in range(n):
        idx = random.randint(0, max_idx)
        _ = db.get(idx)
    elapsed = time.perf_counter() - start
    return n / elapsed


def benchmark_random_read_raw(db, n):
    max_idx = len(db) - 1
    schema = db.schema
    start = time.perf_counter()
    for _ in range(n):
        idx = random.randint(0, max_idx)
        raw = db.get_raw(idx)
        if raw is not None:
            # Zero-copy: decode only what we need (temperature is at offset 3)
            temp = struct.unpack_from("<f", raw, schema.offset("temperature"))[0]
    elapsed = time.perf_counter() - start
    return n / elapsed


def benchmark_dict_insert(data, n):
    start = time.perf_counter()
    for i in range(n):
        data[i] = {
            "id": i,
            "sensor_id": i % 256,
            "temperature": random.gauss(25.0, 5.0),
            "humidity": random.gauss(60.0, 10.0),
            "active": i % 7 == 0,
        }
    elapsed = time.perf_counter() - start
    return n / elapsed


def benchmark_dict_read(data, n):
    keys = list(data.keys())
    start = time.perf_counter()
    for _ in range(n):
        k = random.choice(keys)
        _ = data[k]
    elapsed = time.perf_counter() - start
    return n / elapsed


def main():
    print("=" * 60)
    print("SnapDB Benchmark")
    print(f"Rows: {N:,}")
    print("=" * 60)

    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("sensor_id", "u8"),
        ColumnDef("temperature", "f32"),
        ColumnDef("humidity", "f32"),
        ColumnDef("active", "bool"),
    ])

    db_path = Path("/tmp/snapdb_benchmark.snap")
    if db_path.exists():
        db_path.unlink()

    # --- Insert ---
    db = SnapDB(db_path, schema)
    print("\n--- Insert ---")
    rate = benchmark_insert(db, WARMUP)
    print(f"  Warmup {WARMUP:,}: {rate:,.0f} rows/sec")
    
    rate = benchmark_insert(db, N)
    print(f"  Insert {N:,}:   {rate:,.0f} rows/sec")

    # --- Random Read (decoded) ---
    print("\n--- Random Read (decoded dict) ---")
    rate = benchmark_random_read(db, 100_000)
    print(f"  100K reads: {rate:,.0f} rows/sec")

    # --- Random Read (zero-copy raw) ---
    print("\n--- Random Read (zero-copy memoryview) ---")
    rate = benchmark_random_read_raw(db, 100_000)
    print(f"  100K reads: {rate:,.0f} rows/sec")

    # --- Sequential Scan ---
    print("\n--- Sequential Scan ---")
    start = time.perf_counter()
    count = 0
    for idx, row in db:
        count += 1
    elapsed = time.perf_counter() - start
    print(f"  Scanned {count:,} rows in {elapsed:.3f}s = {count/elapsed:,.0f} rows/sec")

    # --- Memory ---
    print("\n--- Memory Footprint ---")
    tracemalloc.start()
    db2 = SnapDB(db_path, schema)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Current: {current / 1024 / 1024:.2f} MB")
    print(f"  Peak:    {peak / 1024 / 1024:.2f} MB")
    file_size = db_path.stat().st_size
    print(f"  File:    {file_size / 1024 / 1024:.2f} MB")
    theoretical = N * schema.row_width
    print(f"  Overhead: {(file_size - theoretical) / theoretical * 100:.1f}%")

    # --- Dict Baseline ---
    print("\n--- Dict-of-Dicts Baseline ---")
    data = {}
    rate = benchmark_dict_insert(data, WARMUP)
    print(f"  Warmup insert: {rate:,.0f} rows/sec")
    
    # Memory for dict
    tracemalloc.start()
    data = {}
    for i in range(N):
        data[i] = {
            "id": i,
            "sensor_id": i % 256,
            "temperature": random.gauss(25.0, 5.0),
            "humidity": random.gauss(60.0, 10.0),
            "active": i % 7 == 0,
        }
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Dict memory: {peak / 1024 / 1024:.2f} MB")
    
    rate = benchmark_dict_read(data, 100_000)
    print(f"  100K reads: {rate:,.0f} rows/sec")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  SnapDB file size:  {file_size / 1024 / 1024:.2f} MB")
    print(f"  Dict memory:       {peak / 1024 / 1024:.2f} MB")
    print(f"  Space savings:     {(1 - file_size/peak)*100:.1f}%")
    print("=" * 60)

    db.close()


if __name__ == "__main__":
    main()
