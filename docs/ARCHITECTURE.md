# SnapDB Architecture

This document describes the internal architecture of SnapDB, focusing on the
columnar engine's storage layout, encoding stack, and query path. It is aimed
at contributors who want to understand the design tradeoffs before changing
the query / encoding code.

The row-store engine is the same single-process transactional log used since
v0.1.0 — see `docs/memory-efficiency.svg` for a layout diagram of the
columnar side, and `snapdb/core.py` for the row-store implementation.

## Columnar Storage

`ColumnarTable` holds a fixed schema (defined at construction time) and a
per-column `Column` object. Each column is laid out in a single contiguous
`array.array` for the raw data (numeric) or a Python `list` / `bytes` mapping
(string-id), with a parallel `array.array("b")` nullmask. The engine never
reallocates the column arrays during inserts — it appends, which keeps the
hot path allocation-free.

The liveness model is intentionally simple: a row is "deleted" by setting
the nullmask bit on every column. A row is live iff at least one column's
nullmask bit is zero. `_live_mask()` rebuilds the liveness bitmap from the
column nullmasks on demand; `_live_mask_np()` does the same in NumPy for
the vectorized filter paths.

## Encoding Stack

Columns can opt into encoding to shrink memory at the cost of read-decoding
overhead. Each encoding is auto-selected based on the column's dtype and the
values seen during a sampling window:

| Encoding | Eligible dtypes | Win | Tradeoff |
|----------|------------------|-----|----------|
| Plain numeric (`array.array`) | all numeric | baseline | none |
| Bit-packed bool | `bool` | ~8× | bit-twiddling read |
| Frame-of-Reference (FOR) | `i32`, `i64`, `u32`, `u64` | up to ~4.5× | bounded range required |
| Dictionary | `bytes:N` | ~3× | bounded cardinality |
| Delta | `i32`, `i64`, `u32`, `u64` | ~1.2× | monotonic values only |

Each encoding has a `_mode` / `_fallback` flag pair: `_mode = True` means the
column is currently using the encoding, `_fallback = True` means the encoder
turned the encoding off (range too wide, non-monotonic, etc.) and the column
is reading from the raw layout. Both flags persist on the column object so
the engine can introspect the live state.

### FOR + bit packing

The FOR encoder stores the column's min and a bit-packed integer window of
`(value - min)` with the smallest bit-width that fits the range. A range of
0..100 takes 7 bits/value instead of 32 bits/value — a ~4.5× shrink.
The accessor path is `>> bit & mask`, which is cheaper than a full dict
lookup but more expensive than a plain array read; the columnar engine
amortizes this by caching the full reconstruction (`_for_cache`) on first
read after a write, so repeated scans reduce over the cache instead of
decoding through the bit-window.

### Delta encoding

The delta encoder stores the first value as the base and subsequent values
as the difference from the previous value. A monotonic column of uint64
timestamps will fit in an int8 array most of the time. The decoder
incrementally accumulates the deltas in O(n) on first read and caches the
result; later scans use the cached array.

### Bitmask queries over encoded columns

`_rebuild_column_alive` and the bitmask query engine read encoded columns
through the column's standard `tolist()` accessor, which transparently
decodes delta / FOR / dict-encoded columns. This means a bitmask query is
O(n) per encoded column it touches, with the decode amortized across the
materialised list.

## Bitmask Query Engine

The bitmask query engine is the columnar engine's high-throughput path for
predicates that are naturally expressed as bit-level checks on integer
columns — `flags`, booleans packed into `int`s, parity masks, etc. It
bypasses the comparison-operator dispatch used by `select_where()` and
evaluates `(value >> pos) & 1` directly against the expected bit value.

### API

```python
db.select_bitmask(
    {'flags': (0, 1), 'type': (5, 1)},
    operator='AND',  # or 'OR', 'XOR'
)
# returns rows where bit 0 of 'flags' == 1 AND bit 5 of 'type' == 1
```

Multiple bits per column are passed as a list; within a column they're
combined with AND (every listed bit must match for that column to
contribute):

```python
db.select_bitmask({'flags': [(0, 1), (3, 1)]}, operator='XOR')
# returns rows where exactly one of bits 0/3 of 'flags' is set
```

### Combination semantics

The combination operator is applied across the columns that have a
non-null value at the row. AND requires every column to match; OR requires
any column to match; XOR requires an odd number of columns to match
(parity):

| Operator | Match condition |
|----------|-----------------|
| `AND` (default when `match_all=True`) | every column's bit predicate matches |
| `OR` (default when `match_all=False`) | any column's bit predicate matches |
| `XOR` | an odd number of columns' bit predicates match |

### Implementation

1. **Validation.** Each entry in the bitmask dict is validated against the
   table's column list; missing columns raise `ValueError`, non-integer
   dtypes raise `TypeError`. The dict value is normalised to a `list[(pos,
   val)]` so a single `(pos, val)` tuple and a list of them share the
   evaluate path.

2. **Materialisation.** Every column referenced by the bitmask is
   materialised via `Column.tolist()` once per query. The materialised
   list is shared across rows, so the total work is O(n × k) for k
   bitmask columns — encodings (delta / FOR / dict) decode on the first
   pass through `tolist()` and pay no further cost.

3. **Per-row evaluation.** For each live row, the engine evaluates
   `(value >> pos) & 1` against the expected bit value for every
   listed bit, AND-combines the bits within each column, and then
   combines the column match results with the supplied operator. A
   all-bits-zero spec still matches rows whose bits are zero (the
   predicate is "bit N must be v", not "bit N must be 1").

4. **Projection.** The projected columns are materialised once and the
   matched rows are projected with index lookup. `limit` and `offset`
   apply to the result list, not to the column scan.

5. **Encoded columns.** The decoder path is the column's standard
   `tolist()` accessor — encoded columns pay the decode cost once and
   the bitmask evaluate runs over the cached full reconstruction. This
   keeps the implementation uniform across the encoding stack.

### Interaction with auto-indexing

Bitmask queries do not consult the auto-index (`v0.6.0`). The engine
target is high-selectivity scans over a bitmask-encoded column, where
the index would just add a hash-lookup per row. If the engine sees a
hot bitmask query path in the future, the index could be keyed by the
column's full value rather than by bit position; for now the scan path
is the right tradeoff.

### When to use it

Use `select_bitmask` when the predicate is naturally bit-level:

- `flags` / permission masks (`bit 0 = READ`, `bit 1 = WRITE`, …)
- parity predicates (`XOR` over a set of bits)
- presence masks (`bit 0 = has_email`, `bit 1 = has_phone`, …)

Use `select_where` for the comparison-operator path (`==`, `>`, `<`,
`in`, `between`). The two paths share the same row-projection code, so
combining them is straightforward.

## PEP 688 `__buffer__`

`ColumnarTable.__buffer__(flags)` returns the first column's buffer as a
`memoryview`. NumPy can hand this straight to `np.frombuffer` with no
copying. The lifetime constraint is the same as `Column.__buffer__`: while
the returned view is alive, the column's `array.array` cannot grow, so
release the view before further inserts.

For multi-column tables, prefer `column_buffer(name)` to address the
column explicitly.

## `to_dataframe()` and `compact()`

`to_dataframe()` is a thin pandas wrapper that calls `tolist()` on each
column and hands the dict to `pd.DataFrame`. Returns `None` if pandas is
not installed, so the zero-dependency default is unchanged.

`compact()` rebuilds every column's value / nullmask arrays with the
dead rows dropped, and lowers `_row_count` to the live row count. It
returns the rows / bytes delta so callers can log the actual reclaim.
The rebuild path uses `_rebuild_column_alive`, which handles plain
numeric, delta, FOR, bool, dict-encoded, and bytes columns uniformly —
the only column type that needs special handling is dict (the dictionary
is rebuilt so the codes stay dense after the drop).

## Versioning

The columnar engine header (top of `snapdb/columnar.py`) tracks the
encoding-stack version. The package version in `pyproject.toml` and
`snapdb/__init__.py` tracks the feature version. They are not necessarily
identical: the columnar header increments when the encoding layout
changes in a way that affects the on-disk format, while the package
version increments when the public API gains new methods.
