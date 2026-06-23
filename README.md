# SnapDB

**Extremely Lightweight Lightning-Fast In-Memory Database for Python**

A single-file, zero-dependency, pure-Python in-memory database using memory-mapped files, memoryview zero-copy reads, and slab-oriented storage.

## Key Innovations

- **Slab-oriented storage**: Each segment holds all columns for N rows contiguously — no scattered allocations
- **Zero-copy reads**: `memoryview` slices into `mmap` — no deserialization for raw access
- **Single-file**: Schema, bitmap, and data all in one `.snap` file
- **Fixed-width types only**: `int8/16/32/64`, `float32/64`, `bool`, fixed-size `bytes`
- **Pure Python, zero dependencies**: Only stdlib (`mmap`, `struct`, `os`, `json`)
- **Lazy deletes**: Soft-delete with bitmap, no compaction cost

## Architecture

```
[Header] [Schema JSON] [Allocation Bitmap] [Slab 0] [Slab 1] ... [Slab N]
```

Each slab is a fixed-size page (default 4KB). Rows within a slab are packed contiguously by column. The allocation bitmap tracks live vs deleted rows.

## Performance (100K rows)

| Operation | Speed |
|-----------|-------|
| Insert | **69,185 rows/sec** |
| Read (decoded dict) | **309,698 rows/sec** |
| Read (zero-copy raw) | **1,212,499 rows/sec** |
| Sequential scan | **389,072 rows/sec** |

Zero-copy raw reads hit **over 1 million reads per second**.

## Quick Start

```python
from snapdb import SnapDB, Schema, ColumnDef

schema = Schema([
    ColumnDef("id", "i32"),
    ColumnDef("temp", "f32"),
    ColumnDef("active", "bool"),
])

with SnapDB("data.snap", schema) as db:
    db.insert({"id": 1, "temp": 25.5, "active": True})
    db.insert({"id": 2, "temp": 30.0, "active": False})
    
    # Decoded read
    row = db.get(0)  # {'id': 1, 'temp': 25.5, 'active': True}
    
    # Zero-copy raw read (fastest)
    raw = db.get_raw(0)  # memoryview — no allocation
    
    # Iterate all
    for idx, row in db:
        print(idx, row)
    
    # Query
    for idx, row in db.query(lambda r: r["active"]):
        print(f"Active: {row}")
```

## Types

| Type | Width | Notes |
|------|-------|-------|
| `i8`, `i16`, `i32`, `i64` | 1/2/4/8 bytes | Signed integers |
| `u8`, `u16`, `u32`, `u64` | 1/2/4/8 bytes | Unsigned integers |
| `f32`, `f64` | 4/8 bytes | Floats |
| `bool` | 1 byte | True/False |
| `bytes:N` | N bytes | Fixed-width strings |

## Design Decisions

- **Fixed-width only**: Variable-length fields would break zero-copy. Use `bytes:N` with padding.
- **Slab allocator**: Contiguous pages = CPU cache friendly, no malloc fragmentation.
- **mmap persistence**: Survives process crashes (OS syncs pages). Reopen = instant load.
- **Lazy deletes**: No compaction pauses. Reclaim on reopen if needed.

## Comparison

| | SnapDB | SQLite (memory) | Redis | Dict-of-Dicts |
|---|--------|----------------|-------|---------------|
| Zero-copy reads | ✅ | ❌ | ❌ | ❌ |
| Single file | ✅ | ✅ | ❌ | ❌ |
| Pure Python | ✅ | ✅ | ❌ | ✅ |
| No dependencies | ✅ | ✅ | ❌ | ✅ |
| Persistence | ✅ | ✅ | ✅ | ❌ |
| Fixed schema | ✅ | ❌ | ❌ | ❌ |

## Files

- `snapdb.py` — Core engine (~400 lines)
- `test_snapdb.py` — Unit tests
- `quickbench.py` — Benchmark script
- `benchmark.py` — Full benchmark with dict comparison

## License

MIT

## Author

OpenClaw (hussain-alsaibai)
