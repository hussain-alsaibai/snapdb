# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.18.0] - 2026-10-18

### Added

- **Index-aware query planning** — `Query.filter()` now consults
  `HashIndex` (equality) and `RangeIndex` (`gt`/`gte`/`lt`/`lte`) before
  falling back to a full table scan. A single indexed column turns a
  filter from O(n) into O(log n + k); measured 31.5× speedup on a 50K-row
  narrow range scan.
- `Query.between(column, low, high, include_low=True, include_high=True)`
  — readable shortcut for an inclusive range filter; consults the same
  `RangeIndex` as the underlying `filter({col: {"gte": low, "lte": high}})`.
- `Query.explain()` — returns a one-line description of the chosen
  execution plan (`HashIndex[col = v]`, `RangeIndex[col (low, high)]`,
  `Scan(filter=[...])`, `Scan(full)`). Never executes the predicate;
  useful for verifying plans before the expensive part of a query.
- `_plan_query(db, conditions)` — public helper that returns
  `(plan, predicate)`. Useful for tooling that wants to inspect plans
  without running the query.

### Changed

- `Query` carries an internal `where_conditions` dict so the planner can
  re-plan on every `execute()` (queries are immutable; no cache needed).
- `Query.where_fn()` clears the `where_conditions` cache — custom
  predicates cannot be served from any index, so the planner falls back
  to a scan.
- `_iter_matching_values` (used by `sum`/`min`/`max`/`avg`/
  `count_distinct`) is now index-aware when the index planner finds a
  single-column range/equality condition.

### Performance

- 31.5× speedup on `range` predicate (50K rows, 500 matches).
- 27.1× speedup on `eq` predicate via HashIndex lookup.

## [0.17.0] - 2026-09-25

### Added

- **Vector similarity search** (`VectorIndex`) — flat (brute-force) nearest-
  neighbor index with three zero-dependency distance metrics: `cosine`
  (default), `dot`, and `euclidean`. Public primitives `cosine_similarity`,
  `dot_product`, `euclidean_distance` are also exported for standalone use.
  - `add(id, vector)`, `extend([(id, vec), ...])`, `remove(id)`, `clear()`
  - `search(query, k=5)` returns top-k `(id, score)` sorted descending
    (for `euclidean`, distances are negated internally so a single
    descending sort works for all three metrics).
  - Optional `normalize_on_add=True` L2-normalizes vectors on insert so
    `dot` behaves identically to cosine on unit vectors (skipping the
    per-query norm computation when callers pass raw model outputs).
  - Optional `dim=N` validates vector length at insert time.
  - Duplicate ids REPLACE the prior vector (idempotent upsert semantics).
  - Thread-safe for concurrent reads after bulk inserts (RLock around
    writes).
  - `to_dict()` / `from_dict()` JSON-safe serialization for disk persistence
    or network transport.
  - Zero-vector queries return `0.0` rather than `NaN` (deterministic).

  Designed for small-to-medium corpora (≤ ~50k vectors per index) where
  the simplicity of a flat scan beats the complexity of HNSW / IVF. For
  larger corpora, swap in any external FAISS / OpenSearch backend while
  keeping the same API.

  Use case: agent memory retrieval, semantic cache lookup, RAG embedding
  nearest-neighbor, deduplication of tool results. Composes with
  `tiny-prompt-cache`, `tiny-semantic-cache`, `tiny-tool-cache`.

### Tests

- New `tests/test_v017_vector.py` — 36 tests covering distance primitives,
  metric validation, dim checking, search ranking for all three metrics,
  replacement / bulk-add / remove / clear, normalize-on-add round-trip,
  JSON serialization round-trip, concurrent adds (4 threads × 100 vectors),
  and a real-world agent-memory retrieval pattern. Full suite now passes.

## [0.16.1] - 2026-09-20

### Added

- **Aggregations** (`ColumnarTable.aggregate`) — `sum`, `min`, `max`, `avg`,
  `count_distinct` over numeric columns with optional `where=` predicate.
  Returns a flat dict (`{column: value}`) or `None` for empty filtered
  sets.
- **CHANGELOG hygiene** — cleanup of stray `:memory:` artifacts from git
  index.

## [0.16.0] - 2026-09-08

### Added

- **`llms.txt` manifest** at the repo root for agent-native discovery.

## [0.15.0.dev0] - 2026-07-23

### Added

- **Bitmask query engine** (`ColumnarTable.select_bitmask`) — bit-level
  predicates on integer columns: `(value >> position) & 1` against an
  expected bit value, combined with AND / OR / XOR across columns.
  Supports single-bit (`{col: (pos, val)}`) and multi-bit
  (`{col: [(pos, val), ...]}`) forms. Bypasses the comparison operators
  used by `select_where` for queries that are naturally expressed as
  bitmask checks.
- **PEP 688 `__buffer__`** on `ColumnarTable` — exposes the first column's
  raw buffer for zero-copy NumPy access without going through the
  per-column `buffer()` accessor.
- **`to_dataframe()`** — export the table as a pandas `DataFrame` (returns
  `None` if pandas is not installed).
- **`compact()`** — reclaim space after many deletes by rebuilding columns
  with dead rows dropped and shrinking `_row_count` to the live total.
  Returns `{rows_before, rows_after, rows_removed, bytes_freed}`.

### Internal

- `_rebuild_column_alive` helper handles plain numeric, delta, FOR,
  bool, dict-encoded, and bytes columns uniformly; the columnar engine
  version header bumped to v0.10.0.

## [0.14.0] - 2026-07-13

### Fixed

- Derive the WAL sidecar path by suffix (`Path.with_suffix`) instead of
  `str.replace(".snap", ".wal")` so a non-`.snap` database path no longer
  aliases the WAL onto the database file itself.
- `DocumentStore` rebuilds its inferred field map on reopen (inserts /
  updates no longer `KeyError` after a restart); string fields are
  JSON-encoded so numeric-looking strings round-trip as strings; unknown
  query operators raise instead of silently matching everything.
- Tolerate torn / partial trailing WAL records on replay instead of
  making the database unopenable.
- Reject negative row indices in `get` / `get_raw` / `update` / `delete`.
- Columnar: a legitimately-null first column no longer hides a live row
  from scans / aggregates; aggregate `count` honors the `where` predicate;
  delta encoding widens or falls back to raw for out-of-range / negative
  deltas.
- `batch_insert` emits CDC events and returns the inserted-row count for
  both engines; a failed `open` no longer leaks the in-process file lock.

### Speed (measured vs the pre-change code)

- `open` uses `bytearray.count(1)` for the live-row popcount (~73× faster
  open).
- `range_find` prunes whole slabs via per-slab min/max zone maps built
  lazily on first use (~220× on a warm narrow range query).
- Delta / FOR aggregates reduce a cached reconstruction with C-level
  `sum` / `min` / `max` (~14× on repeated delta sums).
- `Query.filter` compiles conditions into one evaluated predicate, values
  passed via namespace (no interpolation / no injection surface) (~2.6×).
- Full slab scans batch-decode via `struct.iter_unpack`; bool columns use
  a mutable `bytearray` bitset (O(1) append); the NumPy filter paths keep
  a C-level vectorized liveness mask.

### Added

- **Named snapshots** — `snapshot()` / `list_snapshots()` /
  `open_snapshot()` / `drop_snapshot()` give restorable point-in-time
  history via a JSON manifest beside the database (full flush + copy v1,
  not copy-on-write); the manifest is written atomically. Opening a
  snapshot returns an independent `SnapDB`.

### Internal

- Consolidate the shared dtype tables, query-value normalization, and the
  XOR-stream cipher into `snapdb/_util.py` (was duplicated across
  core / columnar / index / wal).

### Tests

- Full suite 98 passing (adds `tests/test_v014_features.py` covering
  zone-map `range_find`, snapshots, the compiled filter, and the
  columnar nullable-first-column liveness fix). `ruff` clean.

[0.15.0.dev0]: https://github.com/hussain-alsaibai/snapdb/compare/v0.14.0...HEAD
[0.14.0]: https://github.com/hussain-alsaibai/snapdb/releases/tag/v0.14.0
