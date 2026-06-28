"""Benchmark dictionary encoding vs raw storage."""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapdb import ColumnarTable

def benchmark_dict_encoding():
    print("━─ Dictionary Encoding Benchmark ──\n")

    n = 100_000
    schema = [("id", "i32"), ("status", "bytes:20"), ("category", "bytes:20"), ("score", "f32")]

    # Low-cardinality data
    statuses = ["active", "inactive", "pending", "suspended"]
    categories = ["electronics", "books", "clothing", "food", "toys"]

    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "status": statuses[i % 4],
            "category": categories[i % 5],
            "score": float(i % 100),
        })

    # Raw storage (no dict)
    print("Inserting into RAW storage...")
    raw = ColumnarTable("raw", schema)
    t0 = time.time()
    raw.batch_insert(rows)
    t_raw = time.time() - t0
    raw_mem = raw.memory_usage()

    # Dict-encoded storage
    print("Inserting into DICT-encoded storage...")
    dict_table = ColumnarTable("dict", schema, dict_columns=["status", "category"])
    t0 = time.time()
    dict_table.batch_insert(rows)
    t_dict = time.time() - t0
    dict_mem = dict_table.memory_usage()

    # Results
    print(f"\n{'='*50}")
    print(f"  Rows: {n:,}")
    print(f"  Raw insert:     {t_raw:.3f}s")
    print(f"  Dict insert:    {t_dict:.3f}s")
    print(f"  Raw memory:     {raw_mem:,} B ({raw_mem/1024/1024:.2f} MB)")
    print(f"  Dict memory:    {dict_mem:,} B ({dict_mem/1024/1024:.2f} MB)")
    print(f"  Memory savings: {(raw_mem - dict_mem):,} B ({(raw_mem - dict_mem)/1024/1024:.2f} MB)")
    print(f"  Reduction:      {raw_mem/dict_mem:.1f}×")
    print(f"{'='*50}")

    # Verify data integrity
    print("\nVerifying data integrity...")
    for i in [0, n//2, n-1]:
        r_raw = raw.get(i)
        r_dict = dict_table.get(i)
        assert r_raw["status"] == r_dict["status"], f"Status mismatch at {i}"
        assert r_raw["category"] == r_dict["category"], f"Category mismatch at {i}"
    print("  ✅ Data integrity verified (sampled)")

    # Unique counts
    print(f"\n  status unique:   {dict_table.columns['status'].unique_count()}")
    print(f"  category unique: {dict_table.columns['category'].unique_count()}")
    print(f"  status dict:     {'YES' if dict_table.columns['status']._dict_mode else 'NO'}")


if __name__ == "__main__":
    benchmark_dict_encoding()
