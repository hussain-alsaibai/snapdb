"""Quick benchmark — 100K rows."""

import time
import tempfile
from pathlib import Path

from snapdb import SnapDB, Schema, ColumnDef


def main():
    N = 100_000

    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("sensor_id", "u8"),
        ColumnDef("temperature", "f32"),
        ColumnDef("humidity", "f32"),
        ColumnDef("active", "bool"),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bench.snap"
        db = SnapDB(db_path, schema)

        # Insert
        t0 = time.perf_counter()
        for i in range(N):
            db.insert({
                "id": i,
                "sensor_id": i % 256,
                "temperature": 20.0 + (i % 30),
                "humidity": 50.0 + (i % 20),
                "active": i % 7 == 0,
            })
        t1 = time.perf_counter()
        insert_rate = N / (t1 - t0)
        print(f"Insert {N:,}: {insert_rate:,.0f} rows/sec ({t1-t0:.3f}s)")

        # Random read (decoded)
        import random
        t0 = time.perf_counter()
        for _ in range(50_000):
            idx = random.randint(0, N - 1)
            _ = db.get(idx)
        t1 = time.perf_counter()
        read_rate = 50_000 / (t1 - t0)
        print(f"Read decoded 50K: {read_rate:,.0f} rows/sec ({t1-t0:.3f}s)")

        # Random read (zero-copy raw)
        t0 = time.perf_counter()
        for _ in range(50_000):
            idx = random.randint(0, N - 1)
            _ = db.get_raw(idx)
        t1 = time.perf_counter()
        raw_rate = 50_000 / (t1 - t0)
        print(f"Read raw 50K: {raw_rate:,.0f} rows/sec ({t1-t0:.3f}s)")

        # Sequential scan
        t0 = time.perf_counter()
        count = 0
        for idx, row in db:
            count += 1
        t1 = time.perf_counter()
        scan_rate = count / (t1 - t0)
        print(f"Scan {count:,}: {scan_rate:,.0f} rows/sec ({t1-t0:.3f}s)")

        # Query
        t0 = time.perf_counter()
        active = list(db.query(lambda r: r["active"]))
        t1 = time.perf_counter()
        print(f"Query active: {len(active):,} rows in {t1-t0:.3f}s")

        db.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()
