# SnapDB

**Extremely Lightweight Lightning-Fast In-Memory Database for Python**

A single-file, zero-dependency, pure-Python in-memory database using memory-mapped files, memoryview zero-copy reads, and slab-oriented storage.

## Key Innovations

- **Slab-oriented storage**: Each segment holds all columns for N rows contiguously
- **Zero-copy reads**: `memoryview` slices into `mmap` — no deserialization
- **Single-file**: Schema, bitmap, and data all in one `.snap` file
- **Fixed-width types only**: `int8/16/32/64`, `float32/64`, `bool`, fixed-size `bytes`
- **Pure Python, zero dependencies**: Only stdlib
- **Query engine**: SQL-like `WHERE`, `ORDER BY`, `LIMIT`, `OFFSET`
- **Hash indexes**: O(1) lookups, auto-maintained
- **Transactions**: WAL-based `begin()`/`commit()`/`rollback()`

## Quick Start

```python
from snapdb import SnapDB, Schema, ColumnDef

schema = Schema([
    ColumnDef("id", "i32"),
    ColumnDef("email", "bytes:32"),
    ColumnDef("score", "f32"),
])

with SnapDB("data.snap", schema) as db:
    db.insert({"id": 1, "email": b"alice@test.com", "score": 100.0})
    
    # Query engine
    from query import query
    top5 = query(db).order("score", desc=True).slice(5).execute()
    
    # Index for O(1) lookups
    db.create_index("email")
    matches = db.find(email=b"alice@test.com")
    
    # Transactions
    with db.transaction():
        db.insert({"id": 2, "email": b"bob@test.com", "score": 90.0})
        db.update(0, {"id": 1, "email": b"alice@test.com", "score": 99.0})
```

## Performance (100K rows)

| Operation | Speed |
|-----------|-------|
| Insert | **69,185 rows/sec** |
| Read (decoded dict) | **309,698 rows/sec** |
| Read (zero-copy raw) | **1,212,499 rows/sec** |
| Sequential scan | **389,072 rows/sec** |

## Files

- `snapdb.py` — Core engine with indexing + transactions
- `query.py` — SQL-like query builder
- `index.py` — Hash index implementation
- `wal.py` — Write-ahead log
- `test_snapdb.py` — v0.1.0 tests
- `test_v02.py` — v0.2.0 tests (all passing ✅)
- `benchmark.py` — Performance benchmark

## Changelog

**v0.2.0** (2026-06-25) — Query engine, indexing, transactions
- SQL-like query builder
- Hash indexes with auto-maintenance
- WAL transactions with rollback

**v0.1.0** (2026-06-23) — Initial release
- Slab-oriented mmap storage
- Zero-copy reads
- Fixed-width schema
- Pure Python, zero dependencies

## License

MIT

## Author

OpenClaw (hussain-alsaibai)
