"""Benchmark delta encoding for monotonic columns."""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapdb import ColumnarTable

def benchmark_delta_encoding():
    print("━─ Delta Encoding Benchmark ──\n")

    n = 100_000
    schema = [
        ("id", "i32"),
        ("timestamp", "i64"),
        ("seq", "u32"),
        ("value", "f32"),
    ]

    # Monotonic time-series data
    base_ts = 1700000000000
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "timestamp": base_ts + i * 1000,  # +1s each row
            "seq": i,
            "value": float(i % 100),
        })

    # Raw storage (no delta)
    print("Inserting into RAW storage...")
    raw = ColumnarTable("raw", schema)
    t0 = time.time()
    raw.batch_insert(rows)
    t_raw = time.time() - t0
    raw_mem = raw.memory_usage()

    # Delta-encoded storage
    print("Inserting into DELTA-encoded storage...")
    delta = ColumnarTable("delta", schema, delta_columns=["timestamp", "seq"])
    t0 = time.time()
    delta.batch_insert(rows)
    t_delta = time.time() - t0
    delta_mem = delta.memory_usage()

    # Results
    print(f"\n{'='*50}")
    print(f"  Rows: {n:,}")
    print(f"  Raw insert:     {t_raw:.3f}s")
    print(f"  Delta insert:   {t_delta:.3f}s")
    print(f"  Raw memory:     {raw_mem:,} B ({raw_mem/1024/1024:.2f} MB)")
    print(f"  Delta memory:   {delta_mem:,} B ({delta_mem/1024/1024:.2f} MB)")
    print(f"  Memory savings: {(raw_mem - delta_mem):,} B ({(raw_mem - delta_mem)/1024/1024:.2f} MB)")
    print(f"  Reduction:      {raw_mem/delta_mem:.1f}×")
    print(f"{'='*50}")

    # Verify data integrity
    print("\nVerifying data integrity...")
    for i in [0, n//2, n-1]:
        r_raw = raw.get(i)
        r_delta = delta.get(i)
        assert r_raw["timestamp"] == r_delta["timestamp"], f"Timestamp mismatch at {i}"
        assert r_raw["seq"] == r_delta["seq"], f"Seq mismatch at {i}"
    print("  ✅ Data integrity verified (sampled)")

    # Delta status
    ts_col = delta.columns["timestamp"]
    seq_col = delta.columns["seq"]
    print(f"\n  timestamp delta: {'YES' if ts_col._delta_mode else 'NO'}")
    print(f"  seq delta:       {'YES' if seq_col._delta_mode else 'NO'}")
    print(f"  timestamp bytes saved: {(ts_col._data.itemsize * n) - (ts_col._deltas.itemsize * n + 8):,}")


if __name__ == "__main__":
    benchmark_delta_encoding()
