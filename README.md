# SnapDB

**Extremely Lightweight, Lightning-Fast In-Memory Database for Python**

[![CI](https://github.com/hussain-alsaibai/snapdb/actions/workflows/ci.yml/badge.svg)](https://github.com/hussain-alsaibai/snapdb/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/snapdb.svg)](https://pypi.org/project/snapdb/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **zero-dependency, pure-Python** embedded database with a columnar analytics
engine and a row store, memory-mapped files, lightweight column compression, and
precompiled struct codecs — built for **maximum speed at minimum memory**.

```bash
pip install snapdb
```

## Contents

- [Key Innovations](#key-innovations)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Storage Modes](#storage-modes)
- [Dictionary Encoding](#dictionary-encoding-v040) · [Delta Encoding](#delta-encoding-v050)
- [Vectorized Filtering](#vectorized-filtering-v060) · [Auto-Indexing](#auto-indexing-v060) · [NumPy / Zero-Copy Export](#numpy--zero-copy-export-v060)
- [Benchmarks](#benchmarks)
- [Architecture](#architecture) · [Supported Types](#supported-types)
- [Development](#development)
- [Roadmap & Known Limitations](#roadmap--known-limitations)
- [License](#license)

## Key Innovations

- **Columnar engine** — column-oriented per-column `array.array` storage; full-scan aggregation **~27× faster than SQLite** at a fraction of the memory
- **NumPy-accelerated aggregates** *(optional, v0.8.0)* — when NumPy is installed, `aggregate()` runs over the zero-copy column buffer (**~530M rows/s**, on par with pandas); pure-Python remains the zero-dependency default
- **NumPy-accelerated filters** *(optional, v0.9.0)* — `select_where()` builds masks vectorially; `count_where()` (filtered count, no row materialization) hits **~314M rows/s** on numeric predicates (~166× the pure-Python path)
- **Lowest memory footprint of the field** — ~2.2 MB / 100K rows vs SQLite 2.9 MB, pandas 11 MB, plain `dict` 22 MB ([benchmarks](#benchmarks))
- **Vectorized multi-condition filters** *(v0.6.0)* — `select_where()` combines per-column bitmasks with C-speed big-integer `AND`/`OR` (**~2× faster** selective `WHERE`)
- **O(1) delta-encoded reads** *(v0.6.0)* — lazy reconstruction cache turns delta scans from O(n²) into O(n) (orders of magnitude faster)
- **Auto-indexing** *(v0.6.0)* — `auto_index=True` builds a hash index for a column once it's queried often enough
- **Zero-copy NumPy export** *(v0.6.0)* — `to_numpy()` / `buffer()` (PEP 688) share raw column memory with NumPy without copying
- **Dictionary encoding** — transparent per-column dictionary for low-cardinality strings: **~3× memory reduction** (v0.4.0)
- **Delta encoding** — base + deltas for monotonic columns (timestamps, IDs) (v0.5.0)
- **Bit-packed booleans** — Python `int` bitmask: ~8× smaller than `array('b')`
- **Hash index** — `create_index()` / `lookup()` / `find()`, **kept in sync** on every insert / update / delete
- **Durable writes** — write-ahead log with real transaction rollback; CDC stream; Prometheus-style metrics
- **Zero dependencies** — stdlib only (NumPy is optional, only for zero-copy export)

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

# Fast columnar aggregate (~59M rows/sec full scan)
total = db.aggregate("score", "sum")

# Vectorized multi-condition filter (v0.6.0)
hot = db.select_where([("score", ">", 90.0), ("active", "==", True)])

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

| Mode | Best For | Strengths |
|------|---------|-----------|
| `storage_type="columnar"` | OLAP / analytics | Fast full-scan aggregation (~59M rows/s), vectorized filters, column compression, lowest memory |
| `storage_type="row"` | OLTP / full-row point access | Zero-copy `get_raw()`, WAL transactions, hash indexes, CDC |

See [Benchmarks](#benchmarks) for measured throughput and memory.

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

## Frame-of-Reference Encoding (v0.7.0)

For numeric columns with bounded ranges (ages 0-120, scores 0-100, ratings 1-5), Frame-of-Reference (FOR) stores the minimum value once, then bit-packs deltas into the minimum required bits. **4–8× memory reduction**:

```python
from snapdb import ColumnarTable

schema = [
    ("user_id", "i32"),
    ("age", "i32"),          # Ages 18-65 → 6 bits per value
    ("rating", "i32"),       # Ratings 1-5 → 3 bits per value
    ("score", "i32"),        # Scores 0-100 → 7 bits per value
]

# Enable FOR encoding on bounded numeric columns
db = ColumnarTable("survey", schema, for_columns=["age", "rating", "score"])
```

| Metric | Raw | FOR-Encoded | Improvement |
|--------|-----|-------------|-------------|
| Memory (100K rows, range 0-100) | 400 KB | **~88 KB** | **4.5× reduction** |
| Memory (100K rows, range 0-120) | 400 KB | **~103 KB** | **3.9× reduction** |
| Insert overhead | — | ~10% | Sampling cost |
| Data integrity | — | ✅ 100% | Verified |

- **Auto-detects**: samples first N rows (default 50) to measure range
- **Auto-fallback**: switches to raw if range exceeds 16 bits (saves <50%)
- **Per-column**: specify which columns via `for_columns=[]`
- **Bit-packed**: Python `int` bitmask (same technique as v0.3.2 booleans)
- **Transparent**: reads return full values, no API changes

## Vectorized Filtering (v0.6.0, NumPy-accelerated in v0.9.0)

`select_where()` evaluates each condition column-at-a-time into a mask and
combines them with `AND`/`OR`. With NumPy installed the masks are built
vectorially over the column buffers (pure-Python big-integer masks otherwise).
For filtered counts, `count_where()` skips row materialization entirely and runs
at **~314M rows/s** on numeric predicates (~166× the pure-Python path).

```python
db = SnapDB("events.snap", schema, storage_type="columnar")

# (column, op, value) triples — op ∈ eq/ne/gt/gte/lt/lte/in/between
rows = db.select_where(
    [("age", ">", 30), ("status", "==", b"active")],
    columns=["id", "age"], limit=100,
)

# OR semantics, ranges and membership
db.select_where([("age", "<", 18), ("age", ">", 65)], combine="or")
db.select_where([("age", "between", (30, 40)), ("country", "in", [b"US", b"CA"])])

# dict shorthand
db.select_where({"status": b"active", "age": {"gte": 21}})

# fast filtered count — no rows materialized (NumPy-accelerated)
db.count_where([("age", ">", 30), ("temp", "<", 35.0)])
```

## Auto-Indexing (v0.6.0)

Let SnapDB index the columns you actually query, so you never forget a
`create_index()` for a hot path:

```python
db = SnapDB("users.snap", schema, auto_index=True, auto_index_threshold=8)
# after the 8th equality query on a column, a hash index is built automatically
for uid in stream:
    db.find(email=uid)          # transparently O(1) once the index materializes
```

`find()` also works **without** any index (scan fallback), so correctness never
depends on remembering to index.

## NumPy / Zero-Copy Export (v0.6.0)

Hand raw column memory to NumPy without copying (PEP 688 buffer protocol). NumPy
is an **optional** dependency — only needed if you call these methods.

```python
col = db.to_numpy("temperature")              # safe copy (works for any column)
view = db.to_numpy("temperature", zero_copy=True)   # shares memory, no copy
mv = db.column_buffer("temperature")          # raw memoryview for advanced use
```

Plain numeric columns export a true zero-copy view; encoded columns
(dictionary/delta) transparently fall back to a materialized copy.

## Benchmarks

SnapDB's headline strength is memory efficiency — the columnar store is the
lightest engine in this comparison while staying fully analytical:

<p align="center">
  <img src="docs/memory-efficiency.svg" alt="Memory footprint for 100,000 rows: SnapDB columnar 2.2 MB, sqlite3 in-memory 2.9 MB, pandas 11.0 MB, dict baseline 22.5 MB — lower is better" width="720">
</p>

<p align="center"><em>~5× lighter than pandas and ~10× lighter than a plain <code>dict</code> — with zero dependencies.</em></p>

Reproduce locally (numbers below are from the environment noted in the table):

```bash
python benchmarks/bench_suite.py --rows 100000 --markdown bench.md
```

<!-- BENCH:START -->
_100,000 rows · 50,000 point reads · best of 5 · Python 3.13 · win32 (NumPy installed → accelerated aggregate). Higher is better except Memory (lower is better)._

| Workload | Unit | SnapDB (columnar) | SnapDB (row) | sqlite3 (:memory:) | pandas | dict (baseline) |
|---|---|---|---|---|---|---|
| Bulk insert | rows/s | 467,309 | 287,230 | 770,788 | 794,461 | 11,139,083 |
| Point read (PK) | ops/s | 86,243 | 87,836 | 370,698 | 32,296 | 5,494,807 |
| Full scan + SUM | rows/s | 529,660,985 | 483,067 | 19,910,403 | 513,874,544 | 19,488,619 |
| 3-cond filter | rows/s | 2,259,928 | 470,223 | 11,842,168 | 19,827,894 | 13,811,773 |
| Memory footprint | MB | 2.2 | n/a | 2.9 | 11.0 | 22.5 |
<!-- BENCH:END -->

**Where SnapDB wins (honestly):**

- **Memory** — the columnar store is the **lightest** here: ~5× smaller than pandas and ~10× smaller than a plain `dict`, with zero dependencies.
- **Full-scan aggregation** — **on par with pandas (~530M rows/s)** and ~27× faster than in-memory SQLite. With NumPy installed, `aggregate()` runs over the zero-copy column buffer (issue #14); without NumPy the pure-Python path still does ~58M rows/s (~3× SQLite).
- **Embeddable** — a single mmap-backed file, no server, no C extensions.

**Where it doesn't (also honestly):** pandas still wins multi-condition
filtering (vectorized `WHERE` acceleration is the next item, [#14](https://github.com/hussain-alsaibai/snapdb/issues/14)), and SQLite's
B-tree wins indexed point reads. SnapDB targets the lightweight-embedded-
analytics niche. Encoding memory wins for low-cardinality / monotonic columns
are shown above.

> CI runs this suite on every push and publishes a fresh table to the workflow
> run summary (Actions → CI → Benchmark).

### Encoding memory (100K rows)

| Encoding | Raw | Encoded | Reduction |
|----------|-----|---------|-----------|
| Frame-of-Reference (bounded numeric) | 400 KB | **~88 KB** | **~4.5×** |
| Dictionary (low-cardinality strings) | 4.0 MB | **1.34 MB** | **~3.0×** |
| Delta (monotonic integers) | 2.29 MB | **1.91 MB** | **~1.2×** |

## Architecture

```
SnapDB
├── core.py          — Slab storage, Schema, CRUD, WAL
├── columnar.py      — column-oriented analytical engine
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
# Install with dev + optional extras
pip install -e ".[dev,numpy]"

# Lint (same config CI uses)
ruff check .

# Unit tests
pytest tests/ -q

# Legacy script-style suites (encoding/codec checks)
python tests/test_delta_encoding.py
python tests/test_dict_encoding.py

# Benchmark suite (writes a Markdown table you can drop into the README)
python benchmarks/bench_suite.py --rows 100000 --json bench.json --markdown bench.md
```

Continuous integration (`.github/workflows/ci.yml`) runs ruff, the test matrix
on Linux (3.9–3.13) and Windows, and the benchmark on every push and PR.

## Version History

- **v0.10.0** — Fast row-store bulk insert ([#13](https://github.com/hussain-alsaibai/snapdb/issues/13)):
  - `batch_insert()` now grows the backing file in a **single** truncate + remap for the whole batch instead of one per slab — **~26× faster** (100K rows: ~5.8s → ~0.29s, now in the same ballpark as SQLite/pandas). On-disk format and durability guarantees unchanged
- **v0.9.0** — NumPy-accelerated filters ([#14](https://github.com/hussain-alsaibai/snapdb/issues/14)):
  - `select_where()` builds condition masks vectorially over the column buffers when NumPy is installed (~2× faster); `use_numpy=False` forces the pure-Python path
  - New `count_where()` — filtered row count with no materialization, **~314M rows/s** on numeric predicates (~166×). Exact parity with the pure-Python path verified
  - Bytes/encoded conditions fall back to the Python mask; mixed queries still accelerate their numeric conditions
- **v0.8.0** — Optional NumPy-accelerated aggregates ([#14](https://github.com/hussain-alsaibai/snapdb/issues/14)):
  - `aggregate()` runs `sum`/`min`/`max`/`avg` over the zero-copy column buffer with NumPy when it's installed — **~13–27× faster** (full-scan SUM ~530M rows/s, on par with pandas)
  - Auto-enabled when NumPy is present; `use_numpy=False` forces the pure-Python path; exact parity verified (integers exact, floats within tolerance)
  - Zero-dependency default unchanged; encoded (delta/FOR) and 64-bit-int-sum cases fall through to the exact Python path
- **v0.7.0** — Frame-of-Reference encoding:
  - **New:** Frame-of-Reference (FOR) + bit packing for bounded numeric columns (ages, scores, ratings): **4–8× memory reduction**
  - Auto-detects after sampling threshold (default 50 rows), auto-fallback when range exceeds 16 bits
  - Per-column via `for_columns=[]`, transparent API, update fallback to raw
  - 6 new tests, zero regressions
- **v0.6.0** — Performance, correctness & features:
  - **New:** vectorized multi-condition `select_where()` (bitmask `AND`/`OR`), auto-indexing (`auto_index=True`), zero-copy NumPy export (`to_numpy()`/`buffer()`, PEP 688)
  - Delta-encoded column reads are now **O(1)/O(n)** (lazy reconstruction cache) instead of **O(n)/O(n²)** — orders of magnitude faster delta scans/aggregates
  - Hash indexes are genuinely **kept in sync** on insert / `batch_insert` / update / delete (previously went stale after the first build); single unified `create_index()` for row **and** columnar storage; `find()` gained a scan fallback
  - Fixed data corruption: deleting/nulling a delta-encoded row no longer shifts other rows' values
  - Transaction rollback now actually undoes writes (and restores indexes)
  - **Durability fix:** multi-slab row databases now survive `close()`/reopen — the on-disk bitmap geometry and slab high-water marks are persisted correctly (previously reopening a >1-slab database lost data)
  - Vectorized aggregates (array-level `sum`/`min`/`max`) for null-free numeric columns
  - `__slots__` on hot classes; `close()` reliably releases the mmap (Windows file locks)
  - Tooling: reproducible benchmark suite, GitHub Actions CI (ruff + test matrix + benchmark), `ruff`-clean codebase
- **v0.5.0** — Delta encoding (1.2× memory reduction for monotonic numeric columns)
- **v0.4.0** — Dictionary encoding (3× memory reduction for low-cardinality strings)
- **v0.3.2** — Precompiled struct format, hash index, bit-packed booleans
- **v0.3.1** — Batch insert, optimized columnar, comprehensive benchmarks
- **v0.3.0** — Columnar engine, metrics, CDC
- **v0.2.0** — Query engine, hash indexes, WAL transactions, DocumentStore
- **v0.1.0** — Initial release

## Roadmap & Known Limitations

Tracked as GitHub issues:

- [#11](https://github.com/hussain-alsaibai/snapdb/issues/11) — Frame-of-Reference (FOR) encoding for bounded numeric ranges
- [#12](https://github.com/hussain-alsaibai/snapdb/issues/12) — Low-overhead query profiler via `sys.monitoring` (PEP 669)
- [#14](https://github.com/hussain-alsaibai/snapdb/issues/14) — Optional NumPy-accelerated filters & aggregates (keeping the zero-dependency default)

## License

MIT — see [LICENSE](LICENSE)
