# SnapDB

**Extremely Lightweight Lightning-Fast In-Memory Database for Python**

A zero-dependency, pure-Python in-memory database using columnar storage, memory-mapped files, and precompiled struct formats for maximum speed with minimum memory.

```
pip install snapdb
```

## Key Innovations

- **Columnar storage engine** — ClickHouse-inspired layout with per-column `array.array` storage, 7.5M+ rows/sec aggregate
- **Precompiled struct format** — single `struct.pack`/`unpack` per row (1.6–1.9× faster encode/decode)
- **Bit-packed booleans** — Python `int` bitmask: ~8× memory reduction vs `array.array('b')`
- **Dictionary encoding** — transparent per-column dictionary for low-cardinality strings: **3× memory reduction** (v0.4.0)
- **Delta encoding** — base + deltas for monotonic columns (timestamps, IDs): **1.2× memory reduction** (v0.5.0)
- **Hash index** — O(1) `create_index()` / `lookup()` on any column, auto-maintained on insert/update/delete
- **Batch insert** — `batch_insert()` 5–10× faster than per-row inserts
- **Metrics** — Prometheus-style QPS, latency histograms (p50/p95/p99), operation counters
- **CDC** — Change Data Capture with callback + file-based replay
- **Zero dependencies** — stdlib only, pure Python

## Installation

```bash
pip install snapdb
```

Or from source:
```bash
git clone https://github.com/hussain-alsaibai/snapdb.git
cd snapdb
pip install -e .
```

## Quick Start

```python
from snapdb import SnapDB, Schema, ColumnDef

# Define schema
schema = Schema([
    ColumnDef("id", "i32"),
    ColumnDef("email", "bytes:32"),
    ColumnDef("score", "f32"),
    ColumnDef("active", "bool"),
])

# Create database (columnar mode for analytics)
db = SnapDB("data.snap", schema, storage_type="columnar")

# Insert
db.insert({"id": 1, "email": "alice@test.com", "score": 100.0, "active": True})

# Fast aggregate — 7.5M rows/sec
total = db.aggregate("score", "sum")

# Create index for O(1) lookups
db.create_index("id")
result = db.lookup("id", 1)

# Batch insert for speed
db.batch_insert([
    {"id": i, "email": f"user_{i}@test.com", "score": i * 10.0, "active": i % 2 == 0}
    for i in range(1000)
])

# CDC (Change Data Capture)
from snapdb import Metrics
db = SnapDB("data.snap", schema, metrics=Metrics())
```

## Storage Modes

| Mode | Best For | Memory | Insert | Aggregate |
|------|---------|--------|--------|-----------|
| `storage_type="row"` | OLTP, full-row access | 0.6 MB / 50K rows | 196K ops/sec | ~400K rows/sec |
| `storage_type="columnar"` | OLAP, analytics | 1.5 MB / 50K rows | 656K ops/sec | 7.5M rows/sec |

## Dictionary Encoding (v0.4.0)

For columns with few unique string values (status, category, type, country), dictionary encoding reduces memory by **3×**:

```python
from snapdb import ColumnarTable

schema = [
    ("id", "i32"),
    ("status", "bytes:20"),     # "active", "inactive", "pending" — 3 unique
    ("category", "bytes:20"),   # "electronics", "books", "clothing" — 5 unique
    ("score", "f32"),
]

# Enable dict encoding on low-cardinality columns
db = ColumnarTable("products", schema, dict_columns=["status", "category"])
```

| Metric | Raw | Dict-Encoded | Improvement |
|--------|-----|--------------|-------------|
| Memory (100K rows) | 4.0 MB | **1.34 MB** | **3.0× reduction** |
| Insert | 0.137s | 0.159s | ~15% overhead (acceptable) |
| Data integrity | — | ✅ 100% | Verified |

- **Transparent**: insert/query work with raw strings
- **Auto-fallback**: switches to raw when unique count > threshold (default 256)
- **Per-column**: specify which columns to encode via `dict_columns=[]`

## Delta Encoding (v0.5.0)

For monotonic columns (timestamps, auto-increment IDs, sequences), delta encoding reduces memory by storing differences instead of full values:

```python
from snapdb import ColumnarTable

schema = [
    ("id", "i32"),
    ("timestamp", "i64"),     # Monotonic timestamps → delta-encoded
    ("seq", "u32"),            # Auto-increment IDs → delta-encoded
    ("value", "f32"),
]

# Enable delta encoding on monotonic columns
db = ColumnarTable("events", schema, delta_columns=["timestamp", "seq"])
```

| Metric | Raw | Delta-Encoded | Improvement |
|--------|-----|---------------|-------------|
| Memory (100K rows) | 2.29 MB | **1.91 MB** | **1.2× reduction** |
| Insert | 0.128s | 0.148s | ~16% overhead |
| Data integrity | — | ✅ 100% | Verified |

- **Auto-detects**: samples first 50 rows for monotonicity
- **Auto-fallback**: switches to raw if non-monotonic data detected
- **Per-column**: specify which columns via `delta_columns=[]`
- **Auto-upgrade**: dynamically upgrades delta typecode if deltas overflow

## Benchmarks (100K rows)

**Dictionary Encoding:**

| Metric | Raw | Dict-Encoded | Improvement |
|--------|-----|--------------|-------------|
| Memory | 4.0 MB | **1.34 MB** | **3.0×** |
| Insert | 0.137s | 0.159s | ~15% |

**vs DuckDB, SQLite, Pure Dict:**

| Engine | Insert (batch) | Aggregate SUM | Memory (50K) |
|--------|---------------|---------------|---------------|
| DuckDB (DataFrame) | **6.7M ops/sec** | **96.3M rows/sec** | 10.7 MB |
| SnapDB Col (batch) | 656K ops/sec | **7.5M rows/sec** | **1.5 MB** (7× less) |
| SnapDB Row (batch) | 196K ops/sec | ~400K rows/sec | **0.6 MB** (17× less) |
| SQLite | 586K ops/sec | 10.9M rows/sec | 9.4 MB |
| Pure Dict | 1.1M ops/sec | 17.9M rows/sec | 16.8 MB |

**Key takeaway:** SnapDB wins on memory efficiency (7–28× less than competitors) while staying competitive on analytics. DuckDB dominates raw throughput but uses 17× more memory.

## Architecture

```
SnapDB
├── core.py          — Slab storage, Schema, CRUD, WAL
├── columnar.py      — ClickHouse-inspired columnar engine
├── metrics.py       — Prometheus-style metrics collector
├── index.py         — Hash + multi-column indexes
├── query.py         — SQL-like query builder
├── wal.py           — Write-ahead log for transactions
└── document_store.py — MongoDB-style DocumentStore API
```

## Supported Types

| Type | Bytes | Use Case |
|------|-------|----------|
| `i8` / `u8` | 1 | Flags, small counters |
| `i16` / `u16` | 2 | IDs, ports |
| `i32` / `u32` | 4 | Integers, IDs |
| `i64` / `u64` | 8 | Timestamps, large IDs |
| `f32` | 4 | ML scores, prices |
| `f64` | 8 | Scientific, financial |
| `bool` | ~0.125 | Bit-packed bitmask |
| `bytes:N` | N | Strings, hashes, fixed data |

## Development

```bash
# Run tests
python -m tests.test_snapdb

# Run benchmarks
python benchmarks/benchmark.py

# Run specific version tests
python -m tests.test_v02
python -m tests.test_document_store
```

## Version History

- **v0.6.0** — Performance & correctness pass:
  - Delta-encoded column reads are now **O(1)/O(n)** (lazy reconstruction cache) instead of **O(n)/O(n²)** — multi-order-of-magnitude faster delta scans/aggregates
  - Hash indexes are now genuinely **kept in sync** on insert / `batch_insert` / update / delete (previously went stale after the first build); single unified `create_index()` for row **and** columnar storage
  - Fixed data corruption: deleting/nulling a delta-encoded row no longer shifts other rows' values
  - Transaction rollback now actually undoes writes (and restores indexes)
  - Vectorized aggregates (array-level `sum`/`min`/`max`) for null-free numeric columns
  - `__slots__` on hot classes; `close()` reliably releases the mmap (Windows file locks)
- **v0.5.0** — Delta encoding (1.2× memory reduction for monotonic numeric columns)
- **v0.4.0** — Dictionary encoding (3× memory reduction for low-cardinality strings)
- **v0.3.2** — Precompiled struct format, hash index, bit-packed booleans
- **v0.3.1** — Batch insert, optimized columnar, comprehensive benchmarks
- **v0.3.0** — Columnar engine, metrics, CDC
- **v0.2.0** — Query engine, hash indexes, WAL transactions, DocumentStore
- **v0.1.0** — Initial release

## License

MIT — see [LICENSE](LICENSE)
