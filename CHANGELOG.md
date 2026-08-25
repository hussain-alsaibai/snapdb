# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
