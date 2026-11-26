# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.5] - 2026-11-26

### Fixed

- **`DocumentStore.update()` was a whole-record write with patch syntax.**
  `update(idx, doc)` wrote `doc` as the entire row. Because the row format is
  fixed-width and the decoder omits the null placeholders, every field the
  caller did not name came back *absent*:

  ```python
  db.insert({"name": "alice", "age": 30, "role": "admin"})
  db.update(0, {"age": 31})
  db.get(0)   # -> {"age": 31}    name and role are gone
  ```

  Nothing raised, the row still existed, and the field the caller meant to
  change was right. The rest of the record was erased, and the surviving row
  reads as if it never had those fields -- the worst failure for a store used
  as an agent audit trail.

  `update` is now a patch: named fields replace, unnamed fields keep their
  stored value, and an explicit `None` still clears a field. `replace(idx,
  doc)` provides the old whole-record behaviour for callers who want it.

  Updating a row that does not exist now raises `IndexError` instead of
  writing a placeholder row. Use `insert` to create a row.

  Thirteen new tests; ten are red against 0.18.4. Every existing test wrote
  *every* field in the update payload, which is the one spelling where a
  rewrite and a patch are indistinguishable.

### Known limitation (unchanged, now documented)

An integer or float field set to a real `0` on a nullable column still reads
back as absent: the empty slot and the value zero are the same bytes, so the
decoder cannot tell "never written" from "written as 0". Strings and booleans
are unaffected (`""` and `False` round-trip). Fixing this needs a per-column
presence sentinel, which changes the on-disk row width -- a schema-level change
rather than a decode tweak. Tracked, not fixed here.

## [0.18.4] - 2026-11-23

### Fixed

- **A document store whose first record lacked a field dropped that field
  forever.** `DocumentStore._coerce` coerced every document against the schema
  inferred from record one, so `insert({"text": ..., "ts": 3})` against a store
  whose first document was `{"text": ...}` wrote the text and silently
  discarded `ts` — no error, no warning. The field map now grows when a
  document names a new key, and **`SnapDB.replace_schema(new_schema)`** (new
  public API) performs the on-disk widening, because the row layout is
  positional and a slab decodes with the schema object it was built with.
  A widened schema survives close/reopen and keeps every earlier row.

- **An absent field came back as `0` / `""` / `False`.** Combined with the
  above, pre-widening rows were re-encoded with the type default, so a
  `ts` column claimed every pre-existing record happened at the unix epoch.
  Nullable columns now store an empty slot that decodes to a *missing key*, so
  a caller can tell "no value recorded" from "the value was zero". Absence
  also sorts **last** in both directions rather than first, so
  `query(sort="ts", limit=10)` never spends its first slot on an empty record.

- **`query(sort=...)` raised `TypeError` on a string field with any absent
  value, and silently mis-ordered a float field.** The sort key mixed the
  absent `0` with real values: strings raised
  `'<' not supported between 'str' and 'int'`, floats interleaved with `0.0`
  and produced a plausible-looking wrong order. The sort key is now a total
  order (`_sort_key`) and `desc` reverses only the value band.

- **`sort` was applied after `limit`/`offset`.** `query(sort="ts", limit=2)`
  limited the *unsorted* scan, so the two rows it returned were the first two
  on disk, not the two smallest. Sorting now precedes slicing.

- **`search(term, limit=0)` returned 1 hit; a negative limit meant "no
  limit".** `limit <= 0` now returns nothing, matching `query()`.

### Changed

- `replace_schema()` re-packs live rows densely: ids after a deleted row shift
  down. Read the docstring before holding a row id across a schema change.

## [0.18.3] - 2026-11-19

### Fixed

- **`from snapdb import query` gave you the module, not the function.** The
  README has documented this import since v0.18.0 and it had never worked.
  `snapdb/query.py` defines a function also called `query()`, and
  `__init__.py` did not import it. Python's `from package import name` falls
  back to loading the submodule when the attribute is absent, so the name
  bound to the **module** and every documented call raised:

  ```
  TypeError: 'module' object is not callable
  ```

  Nothing raised at import time, so the failure surfaced only in user code,
  where it reads as the user's mistake. It survived four releases because
  every test imported from the defining module (`from snapdb.query import
  query`), which has always worked. `__init__.py` now re-exports both `query`
  and `Query`, and six regression tests assert the package-level name is
  callable, is in `__all__`, and that the README's own example runs end to
  end. Five of the six fail against 0.18.2.

  Internal call sites that relied on the accidental module binding used
  `import snapdb.query as qmod`; two test files did, and now use
  `importlib.import_module("snapdb.query")` so the distinction is explicit.

### Added

- `examples/agent_memory.py` — the README's "SnapDB for AI Agent Memory"
  argument as a runnable script: 20k-record audit trail, named snapshot,
  point reads, indexed vs. scanned range query with `explain()`, and a
  restore that checks the row count. Asserts the index changes the plan and
  not the result.
- CI runs the example's assertions through the test suite
  (`tests/test_package_query_export.py`), so it cannot rot the way a README
  snippet does.

## [0.18.1] - 2026-11-11

### Fixed

- **The published sdist could not pass its own test suite.** Two defects,
  both invisible in a checkout because a checkout has files an sdist does
  not:
  - `tests/test_version_consistency.py` read `CHANGELOG.md` and
    `package.json` from the repo root, and neither was in the tarball →
    2 `FileNotFoundError` failures for anyone who ran
    `pip download pysnapdb --no-binary :all: && pytest`.
  - `tests/test_delta_encoding.py` and `tests/test_dict_encoding.py` defined a
    module-level helper `def test(name, ok, detail="")`. pytest collects that
    as a test with a fixture called `name` → `fixture 'name' not found` on
    every run, including ours. Renamed to `check()` (39 call sites).

### Added

- `MANIFEST.in` — ships `CHANGELOG.md` and `package.json` in the sdist.
- `tests/test_module_level_api.py` — asserts that the files the suite reads
  are in the sdist, that the sdist version matches `PKG-INFO`, and that no
  test module defines a pytest-collectable helper named `test()`. The first
  two skip in a checkout (they can only be true of a tarball); the third runs
  everywhere.
- Repo-metadata checks now `skipTest` instead of `FileNotFoundError` when the
  file is absent from the sdist — the checkout still enforces them.

## [0.18.2] - 2026-11-16

### Fixed

- **The selectivity guard only applied to `execute()`. `count()` and every
  aggregation bypassed it.** v0.18.2 (this release) added the guard, wired it
  into `execute()`, and left three other paths calling the raw planner. Both
  directions were wrong:

  | query | `execute()` | `count()` / `sum()` / `avg()` / `min()` / `max()` |
  |---|---|---|
  | range matching **1%** of rows | 1.1 ms (index) | **37 ms** (full scan) |
  | range matching **62%** of rows | scan | scan **plus** a 162K-element candidate set |

  A selective `count()` was ~34× slower than `len(execute())` on the same
  predicate, and it scanned a table whose index already named the matching
  rows. A wide aggregation scanned *and* materialised the plan `explain()` had
  just reported as rejected - on a 200K-row table, `_execute_plan` built a
  162K-element set in 706 ms, paid again by each of four aggregations.

  The guard now lives in a single helper, `_plan_or_scan()`, used by
  `execute()`, `count()` and `_iter_matching_values()`. A regression test
  asserts that exactly two call sites may reach the raw `_plan_query()`, so a
  future consumer cannot silently skip the guard.

- **`min()` and `max()` raised `TypeError` when a query matched exactly one
  row.** `min(first, *rest)` lowers to `min(first)` when `rest` is empty, and
  `min` of a number is not iterable. A one-row result is the common case for a
  selective index query, so this broke the queries the index made fast. The
  first value is now chained back into the iterator instead of splatted.

### Changed

- `Query.count()` walks the index candidate set instead of the whole table.
  Correctness is unchanged; membership testing against a set no longer forces
  a decode of every row.


### Performance

- **An index plan is now only used when it is the cheaper plan.** `RangeIndex`
  was used for every range predicate that had one, regardless of how many rows
  the range covered. Serving k rows from an index costs k random decodes
  through `db.get`, while a scan decodes sequentially, so past a few tens of
  percent of the table the index was **slower than the scan it replaced**.
  Measured on a 50K-row `f32` column (median of 9 runs):

  | rows matched | scan | index | index speedup |
  |---|---|---|---|
  | 30 (0.1%) | 24.7 ms | 0.15 ms | **169×** |
  | 505 (1%) | 25.7 ms | 0.97 ms | 26× |
  | 2,505 (5%) | 25.4 ms | 4.6 ms | 5.5× |
  | 10,005 (20%) | 27.3 ms | 19.3 ms | 1.4× |
  | 20,005 (40%) | 31.7 ms | 29.7 ms | 1.1× |
  | 40,005 (80%) | 37.8 ms | 35.7 ms | 1.1× |

  (The last two rows are post-fix: the guard hands those queries back to the
  scan, which is why they now match scan speed. Before the fix they were
  **0.9× and 0.4×** — actively slower with the index in place.)

  `Query.execute()` now estimates the rows a range plan would touch with two
  bisects and drops the plan above `_INDEX_SCAN_RATIO` (0.30) of the table.
  Equality plans are unchanged. Results are identical either way — the guard
  changes how rows are found, never which.

- `Query.explain()` reports the plan `execute()` will actually use, and names
  the reason when an available index was rejected:
  `Scan(filter=['age']) [index rejected: 62/100 rows]`. Previously it
  advertised `RangeIndex[...]` for a query that fell back to a scan.

### Docs

- The 0.18.0 "31.5× speedup on a 50K-row narrow range" claim could not be
  reproduced. Re-measuring the stated shape (50K rows, ~500 matches) gives
  **26×**, and the number swings from 5.5× to 169× depending on selectivity
  and on whether the comparison includes query-construction overhead. The
  README and the 0.18.0 entry now carry the table above instead of a single
  scalar.

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
