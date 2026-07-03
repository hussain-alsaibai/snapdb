"""
SnapDB — Extremely Lightweight Lightning-Fast In-Memory Database

A single-file, zero-dependency, pure-Python in-memory database using
memory-mapped files, memoryview zero-copy reads, and slab-oriented storage.

v0.3.0: Added columnar storage engine, metrics, and CDC support.

Architecture:
    [Header] [Schema JSON] [Allocation Bitmap] [Slab 0] [Slab 1] ... [Slab N]

Author: OpenClaw (hussain-alsaibai)
License: MIT
"""

from __future__ import annotations

import json
import mmap
import os
import pickle
import shutil
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union
from contextlib import contextmanager, nullcontext

from ._util import (_type_size, _struct_code,
                    _norm_value, _derive_key, _xor_stream)

# v0.2.0 imports (optional)
try:
    from .index import HashIndex, MultiIndex, RangeIndex
except ImportError:
    HashIndex = None  # type: ignore
    MultiIndex = None  # type: ignore
    RangeIndex = None  # type: ignore
try:
    from .wal import WAL
except ImportError:
    WAL = None  # type: ignore

# v0.3.0 imports (optional)
try:
    from .columnar import ColumnarTable
except ImportError:
    ColumnarTable = None  # type: ignore
try:
    from .metrics import Metrics
except ImportError:
    Metrics = None  # type: ignore


# ── Header ─────────────────────────────────────────────────────────────────────

_HEADER_FMT = "<4sHHIQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAGIC = b"SNAP"
_VERSION = 1
_DEFAULT_PAGE_SIZE = 4096


def _pack_header(schema_offset: int, page_size: int = _DEFAULT_PAGE_SIZE) -> bytes:
    return struct.pack(_HEADER_FMT, _MAGIC, _VERSION, page_size, 0, schema_offset)


def _unpack_header(buf: bytes) -> Tuple[int, int, int, int]:
    magic, version, page_size, flags, schema_offset = struct.unpack(_HEADER_FMT, buf)
    if magic != _MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")
    if version != _VERSION:
        raise ValueError(f"Unsupported version: {version}")
    return version, page_size, flags, schema_offset


# ── Schema ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColumnDef:
    name: str
    dtype: str
    unique: bool = False
    primary_key: bool = False
    not_null: bool = True
    width: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "width", _type_size(self.dtype))
        if self.primary_key:
            object.__setattr__(self, "unique", True)
            object.__setattr__(self, "not_null", True)


class Schema:
    """Table schema definition."""

    def __init__(self, columns: List[ColumnDef]) -> None:
        self.columns = tuple(columns)
        self._name_to_idx = {c.name: i for i, c in enumerate(columns)}
        self._row_width = sum(c.width for c in columns)
        self._offsets = [0]
        for c in columns[:-1]:
            self._offsets.append(self._offsets[-1] + c.width)
        # Precompile struct format for fast encode/decode
        self._compile_format()

    def _compile_format(self) -> None:
        """Precompute struct format string for batch pack/unpack."""
        self._struct_fmt = "<" + "".join(_struct_code(c.dtype) for c in self.columns)
        self._struct_size = struct.calcsize(self._struct_fmt)
        # Precomputed for iter_unpack_rows: column names in order, and which
        # positions need the bytes rstrip+decode step.
        self._col_names_tuple = tuple(c.name for c in self.columns)
        self._bytes_positions = tuple(i for i, c in enumerate(self.columns)
                                      if c.dtype.startswith("bytes"))
        # Columns eligible for per-slab zone maps (min/max pruning): ordered,
        # comparable, fixed-width scalars. Bytes columns ARE lexically
        # comparable but their zone-map value would be the decoded str (not
        # the raw bytes), which range_find compares against; skip them to
        # keep the maintenance cost predictable — numeric range queries are
        # the overwhelmingly common case zone maps target.
        self._zone_columns = tuple(c.name for c in self.columns
                                   if not c.dtype.startswith("bytes") and c.dtype != "bool")

    @property
    def struct_fmt(self) -> str:
        return self._struct_fmt

    @property
    def struct_size(self) -> int:
        return self._struct_size

    @property
    def row_width(self) -> int:
        return self._row_width

    def offset(self, name: str) -> int:
        return self._offsets[self._name_to_idx[name]]

    def index(self, name: str) -> int:
        return self._name_to_idx[name]

    def to_json(self) -> List[Dict[str, str]]:
        return [
            {
                "name": c.name,
                "dtype": c.dtype,
                "unique": c.unique,
                "primary_key": c.primary_key,
                "not_null": c.not_null,
            }
            for c in self.columns
        ]

    @classmethod
    def from_json(cls, data: List[Dict[str, str]]) -> "Schema":
        return cls([
            ColumnDef(
                c["name"],
                c["dtype"],
                unique=bool(c.get("unique", False)),
                primary_key=bool(c.get("primary_key", False)),
                not_null=bool(c.get("not_null", True)),
            )
            for c in data
        ])

    def to_columnar_schema(self) -> List[Tuple[str, str]]:
        """Convert to list of (name, dtype) tuples for ColumnarTable."""
        return [(c.name, c.dtype) for c in self.columns]

    def encode_row(self, row: Dict[str, Any]) -> bytes:
        # Use precompiled struct format for speed
        values = []
        for col in self.columns:
            if col.name not in row:
                raise KeyError(f"Missing required column: {col.name}")
            val = row[col.name]
            if val is None and col.not_null:
                raise ValueError(f"Column {col.name!r} cannot be None")
            if col.dtype == "bool":
                values.append(bool(val))
            elif col.dtype.startswith("bytes"):
                raw = val if isinstance(val, bytes) else str(val).encode("utf-8")
                values.append(raw[:col.width].ljust(col.width, b"\x00"))
            else:
                values.append(val)
        return struct.pack(self._struct_fmt, *values)

    def decode_row(self, buf: Union[bytes, memoryview]) -> Dict[str, Any]:
        # struct.unpack accepts memoryview directly — avoid the bytes() copy.
        unpacked = struct.unpack(self._struct_fmt, buf)
        row: Dict[str, Any] = {}
        for col, val in zip(self.columns, unpacked):
            if col.dtype.startswith("bytes"):
                row[col.name] = val.rstrip(b"\x00").decode("utf-8", errors="replace")
            else:
                row[col.name] = val
        return row

    def iter_unpack_rows(self, buf: bytes) -> Iterator[Dict[str, Any]]:
        """Batch-decode a contiguous buffer of whole rows.

        struct.iter_unpack does the per-row unpacking in a single C-level call
        sequence instead of N Python-level struct.unpack calls — the dominant
        cost of a full scan. ``len(buf)`` must be an exact multiple of
        ``struct_size`` (true for a slab's [0, count) byte range). Only used
        for unencrypted slabs; encrypted rows are decrypted individually and
        go through decode_row instead.
        """
        names = self._col_names_tuple
        bytes_positions = self._bytes_positions
        if not bytes_positions:
            for unpacked in struct.iter_unpack(self._struct_fmt, buf):
                yield dict(zip(names, unpacked))
            return
        for unpacked in struct.iter_unpack(self._struct_fmt, buf):
            row: Dict[str, Any] = dict(zip(names, unpacked))
            for pos in bytes_positions:
                name = names[pos]
                row[name] = row[name].rstrip(b"\x00").decode("utf-8", errors="replace")
            yield row


# ── Slab (Segment) ───────────────────────────────────────────────────────────

class Slab:
    """Memory-mapped slab holding rows in a contiguous buffer."""

    __slots__ = ("schema", "_mm", "_offset", "_capacity", "_count", "_live", "_crypt",
                "_zone_min", "_zone_max", "_zone_built")

    def __init__(self, schema: Schema, mm: mmap.mmap, offset: int, capacity: int,
                 crypt: Optional[Callable[[bytes, int], bytes]] = None) -> None:
        self.schema = schema
        self._mm = mm
        self._offset = offset
        self._capacity = capacity
        self._count = 0
        self._live = bytearray(capacity)
        self._crypt = crypt
        # Zone map (min/max per column) for range-query slab pruning. A
        # freshly-constructed slab has no rows yet, so it starts "built" —
        # every future row arrives through insert()/batch_insert(), which
        # maintain it incrementally. A slab reconstructed by _load() from an
        # existing file is different: it already has rows nothing here has
        # seen, so the caller (SnapDB._load) marks it unbuilt; ensure_zone_map
        # then does one lazy full scan on first use instead of paying that
        # cost for every reopen regardless of whether range queries happen.
        self._zone_min: Dict[str, Any] = {}
        self._zone_max: Dict[str, Any] = {}
        self._zone_built = True

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count >= self._capacity

    def _row_offset(self, idx: int) -> int:
        return self._offset + idx * self.schema.row_width

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx >= self._capacity or not self._live[idx]:
            return None
        start = self._row_offset(idx)
        end = start + self.schema.row_width
        raw = bytes(memoryview(self._mm)[start:end])
        if self._crypt is not None:
            raw = self._crypt(raw, start)
        return self.schema.decode_row(raw)

    def get_raw(self, idx: int) -> Optional[memoryview]:
        if idx >= self._capacity or not self._live[idx]:
            return None
        start = self._row_offset(idx)
        end = start + self.schema.row_width
        return memoryview(self._mm)[start:end]

    def _zone_widen(self, row: Dict[str, Any]) -> None:
        zmin, zmax = self._zone_min, self._zone_max
        for name in self.schema._zone_columns:
            v = row.get(name)
            if v is None:
                continue
            cur_min = zmin.get(name)
            if cur_min is None or v < cur_min:
                zmin[name] = v
            cur_max = zmax.get(name)
            if cur_max is None or v > cur_max:
                zmax[name] = v

    def ensure_zone_map(self) -> None:
        """Build the zone map if it hasn't been built yet (only true right
        after a reload — see __init__). Deferred so opening a file stays
        O(slabs) instead of O(rows); the first range query per slab pays a
        one-time scan and every query after it (this session) is pruned."""
        if self._zone_built:
            return
        if self._count > 0 and self.schema._zone_columns:
            for _, row in self.iter_rows():
                self._zone_widen(row)
        self._zone_built = True

    def zone_overlaps(self, column: str, low: Any, high: Any,
                      include_low: bool = True, include_high: bool = True) -> bool:
        """True if this slab's rows COULD contain a value in [low, high].

        Conservative: bounds only ever widen (never shrink on delete/update),
        so False means definitely no match; True means the caller must still
        filter row-by-row. Missing zone info (unbuilt, or an all-null column)
        also means True — safe default, just no pruning."""
        zmin = self._zone_min.get(column)
        zmax = self._zone_max.get(column)
        if zmin is None or zmax is None:
            return True
        if high is not None:
            if (zmin > high) if include_high else (zmin >= high):
                return False
        if low is not None:
            if (zmax < low) if include_low else (zmax <= low):
                return False
        return True

    def insert(self, row: Dict[str, Any]) -> int:
        if self.is_full:
            raise RuntimeError("Slab is full")
        idx = self._count
        self._count += 1
        self._live[idx] = 1
        raw = self.schema.encode_row(row)
        start = self._row_offset(idx)
        if self._crypt is not None:
            raw = self._crypt(raw, start)
        mv = memoryview(self._mm)[start : start + len(raw)]
        mv[:] = raw
        # Only maintain incrementally once the map reflects everything prior
        # (see ensure_zone_map) — otherwise this row's bounds would look
        # authoritative while older, not-yet-scanned rows are excluded.
        if self._zone_built and self.schema._zone_columns:
            self._zone_widen(row)
        return idx

    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        """Insert multiple rows at once. Returns index of first inserted row."""
        n = len(rows)
        if self._count + n > self._capacity:
            n = self._capacity - self._count
            rows = rows[:n]
        if n <= 0:
            raise RuntimeError("Slab is full")
        start_idx = self._count
        row_width = self.schema.row_width
        base_offset = self._row_offset(start_idx)
        # Encode all rows into a single buffer, then write in one shot
        buf = bytearray(len(rows) * row_width)
        widen = self._zone_built and self.schema._zone_columns
        for i, row in enumerate(rows):
            raw = self.schema.encode_row(row)
            if self._crypt is not None:
                raw = self._crypt(raw, base_offset + i * row_width)
            buf[i * row_width : (i + 1) * row_width] = raw
            self._live[start_idx + i] = 1
            if widen:
                self._zone_widen(row)
        self._mm[base_offset : base_offset + len(buf)] = buf
        self._count += len(rows)
        return start_idx

    def update(self, idx: int, row: Dict[str, Any]) -> None:
        if idx >= self._capacity or not self._live[idx]:
            raise KeyError(f"Row {idx} not found")
        raw = self.schema.encode_row(row)
        start = self._row_offset(idx)
        if self._zone_built and self.schema._zone_columns:
            self._zone_widen(row)
        if self._crypt is not None:
            raw = self._crypt(raw, start)
        mv = memoryview(self._mm)[start : start + len(raw)]
        mv[:] = raw

    def delete(self, idx: int) -> None:
        if idx < self._capacity:
            self._live[idx] = 0

    def iter_rows(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        count = self._count
        if count == 0:
            return
        row_width = self.schema.row_width
        live = self._live
        if self._crypt is None:
            # Whole-slab batch decode: one contiguous byte-copy + one
            # struct.iter_unpack pass over all `count` rows (live and dead),
            # instead of a Python-level struct.unpack call per live row. Dead
            # rows still get unpacked but that's a cheap C-level no-op compared
            # to the interpreter overhead this replaces.
            buf = bytes(memoryview(self._mm)[self._offset:self._offset + count * row_width])
            for i, row in enumerate(self.schema.iter_unpack_rows(buf)):
                if live[i]:
                    yield i, row
            return
        # Encrypted slab: nonce is derived per absolute offset, so rows must
        # be decrypted individually before they can be unpacked.
        offset = self._offset
        mm = self._mm
        crypt = self._crypt
        for i in range(count):
            if live[i]:
                start = offset + i * row_width
                raw: bytes = bytes(memoryview(mm)[start:start + row_width])
                raw = crypt(raw, start)
                yield i, self.schema.decode_row(raw)

    def iter_raw(self) -> Iterator[Tuple[int, memoryview]]:
        for i in range(self._count):
            if self._live[i]:
                yield i, self.get_raw(i)


# ── CDC (Change Data Capture) ──────────────────────────────────────────────────

class CDCLog:
    """
    Simple Change Data Capture log.
    Streams insert/update/delete events to a callback or file.
    """

    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 log_file: Optional[str] = None) -> None:
        self._callback = callback
        self._log_file = log_file
        self._file = None
        if log_file:
            self._file = open(log_file, "a")

    def append(self, op: str, idx: int, row: Optional[Dict[str, Any]] = None,
               old_row: Optional[Dict[str, Any]] = None) -> None:
        event = {"op": op, "idx": idx, "row": row, "old": old_row,
                 "ts": time.time()}
        if self._callback:
            self._callback(event)
        if self._file:
            self._file.write(json.dumps(event) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()

    def replay(self) -> List[Dict[str, Any]]:
        """Replay events from log file."""
        events = []
        if self._log_file and os.path.exists(self._log_file):
            with open(self._log_file, "r") as f:
                for line in f:
                    events.append(json.loads(line))
        return events


# ── Metric timing helpers ──────────────────────────────────────────────────────

_NULL_CTX = nullcontext()


@contextmanager
def _metric_timer(db: "SnapDB", name: str):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        db._m_inc(f"db_{name}_total")
        db._m_lat(f"db_{name}_latency", elapsed)


# ── SnapDB Engine ──────────────────────────────────────────────────────────────

class SnapDB:
    """
    Single-file in-memory database with zero-copy reads.

    Usage:
        db = SnapDB("data.snap", schema)
        db.insert({"id": 1, "name": "alice", "score": 95.5})

    Columnar mode:
        db = SnapDB("data.snap", schema, storage_type="columnar")

    With metrics:
        from metrics import Metrics
        m = Metrics()
        db = SnapDB("data.snap", schema, metrics=m)
    """

    _open_paths: set[Path] = set()
    _open_paths_lock = threading.Lock()

    def __init__(self, path: Union[str, Path], schema: Schema,
                 page_size: int = _DEFAULT_PAGE_SIZE,
                 storage_type: str = "row",
                 metrics: Optional[Metrics] = None,
                 cdc: Optional[CDCLog] = None,
                 auto_index: bool = False,
                 auto_index_threshold: int = 8,
                 dict_columns: Optional[List[str]] = None,
                 delta_columns: Optional[List[str]] = None,
                 encryption_key: Union[str, bytes, None] = None) -> None:
        self.path = Path(path)
        self.schema = schema
        self.page_size = page_size
        self._storage_type = storage_type
        self._metrics = metrics
        self._cdc = cdc
        self._lock = threading.RLock()
        self._closed = False
        self._recovering = False
        self._file_lock_handle: Optional[BinaryIO] = None
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._crypto_key = _derive_key(encryption_key)
        self._acquire_file_lock()
        self._unique_columns = [c.name for c in schema.columns if c.unique or c.primary_key]
        self._unique_values: Dict[str, Dict[Any, int]] = {c: {} for c in self._unique_columns}

        # Auto-indexing (issue #6): once a column has been queried by equality
        # this many times, a hash index is built for it automatically.
        self._auto_index = auto_index
        self._auto_index_threshold = max(1, auto_index_threshold)
        self._access_counts: Dict[str, int] = {}

        if storage_type == "columnar":
            if ColumnarTable is None:
                self._release_file_lock()
                raise RuntimeError("Columnar storage not available (columnar.py missing)")
            self._table = ColumnarTable(
                "columnar_store", schema.to_columnar_schema(),
                dict_columns=dict_columns, delta_columns=delta_columns)
            self._indexes = None
            self._wal = None
            self._slabs = None
            self._mm = None
            self._file = None
            self._total_rows = 0
            self._in_tx = False
            self._tx_state: List[Tuple[str, int, Optional[Dict]]] = []
            try:
                if self.path.exists() and self.path.stat().st_size > 0:
                    self._load_columnar()
                self._rebuild_unique_constraints()
            except Exception:
                # A failed open (corrupt/encrypted file) must not leak the
                # process-wide lock, or the path could never be reopened.
                self._release_file_lock()
                raise
            return

        # Row-oriented storage
        self._slabs: List[Slab] = []
        self._rows_per_slab = page_size // schema.row_width
        self._total_rows = 0
        self._mm: Optional[mmap.mmap] = None
        self._file: Optional[BinaryIO] = None

        self._in_tx = False
        self._tx_state: List[Tuple[str, int, Optional[Dict]]] = []
        self._indexes: Optional["MultiIndex"] = None
        self._range_indexes: Dict[str, "RangeIndex"] = {}
        self._wal: Optional["WAL"] = None
        if MultiIndex is not None:
            self._indexes = MultiIndex()
        if WAL is not None:
            # Suffix-based sidecar derivation (like the .lock file above). The
            # old str.replace(".snap", ".wal") aliased the WAL onto the database
            # file itself for paths without ".snap" (WAL appends/clears would
            # then corrupt or delete the database).
            if self.path.suffix == ".snap":
                wal_path = self.path.with_suffix(".wal")
            else:
                wal_path = self.path.with_suffix(self.path.suffix + ".wal")
            self._wal = WAL(str(wal_path), encryption_key=encryption_key)

        if page_size > 65535:
            self._release_file_lock()
            raise ValueError("page_size must be <= 65535 bytes")
        if self._rows_per_slab < 1:
            self._release_file_lock()
            raise ValueError(f"Row size ({schema.row_width}) exceeds page size ({page_size})")

        try:
            if self.path.exists() and self.path.stat().st_size >= _HEADER_SIZE:
                self._load()
            else:
                self._create()
            self._recover_wal()
            self._unique_columns = [c.name for c in self.schema.columns if c.unique or c.primary_key]
            self._unique_values = {c: {} for c in self._unique_columns}
            self._rebuild_unique_constraints()
        except Exception:
            # A failed open (bad magic, corrupt schema, WAL/constraint errors)
            # must not leak the process-wide lock, or the path could never be
            # reopened in this process.
            self._close_mapping_only()
            self._release_file_lock()
            raise

    def _acquire_file_lock(self) -> None:
        """Prevent two writable SnapDB instances from opening one file.

        The in-process registry catches same-process double-open; the advisory
        sidecar lock catches separate processes on platforms that support it.
        """
        resolved = self.path.resolve()
        with self._open_paths_lock:
            if resolved in self._open_paths:
                raise RuntimeError(f"Database is already open in this process: {self.path}")
            self._open_paths.add(resolved)
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_lock_handle = open(self._lock_path, "a+b")
            if os.name == "nt":
                import msvcrt
                self._file_lock_handle.seek(0)
                try:
                    msvcrt.locking(self._file_lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RuntimeError(f"Database is locked by another process: {self.path}") from exc
            else:
                import fcntl
                try:
                    fcntl.flock(self._file_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RuntimeError(f"Database is locked by another process: {self.path}") from exc
        except Exception:
            with self._open_paths_lock:
                self._open_paths.discard(resolved)
            if self._file_lock_handle is not None:
                self._file_lock_handle.close()
                self._file_lock_handle = None
            raise

    def _release_file_lock(self) -> None:
        resolved = self.path.resolve()
        handle = self._file_lock_handle
        if handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
                self._file_lock_handle = None
        with self._open_paths_lock:
            self._open_paths.discard(resolved)

    def _crypt_row(self, data: bytes, absolute_offset: int) -> bytes:
        if self._crypto_key is None:
            return data
        nonce = absolute_offset.to_bytes(16, "little", signed=False)
        return _xor_stream(self._crypto_key, nonce, data)

    def _encrypt_blob(self, data: bytes) -> bytes:
        if self._crypto_key is None:
            return data
        nonce = os.urandom(16)
        encrypted = _xor_stream(self._crypto_key, nonce, data)
        return b"SNAPENC1" + nonce + encrypted

    def _decrypt_blob(self, data: bytes) -> bytes:
        if not data.startswith(b"SNAPENC1"):
            return data
        if self._crypto_key is None:
            raise ValueError("Database is encrypted; pass encryption_key to open it")
        nonce = data[8:24]
        return _xor_stream(self._crypto_key, nonce, data[24:])

    def _check_row_shape(self, row: Dict[str, Any]) -> None:
        for col in self.schema.columns:
            if col.name not in row:
                raise KeyError(f"Missing required column: {col.name}")
            if row[col.name] is None and col.not_null:
                raise ValueError(f"Column {col.name!r} cannot be None")

    def _unique_key(self, value: Any) -> Any:
        return _norm_value(value)

    def _validate_unique(self, row: Dict[str, Any], existing_idx: Optional[int] = None) -> None:
        for col in self._unique_columns:
            key = self._unique_key(row.get(col))
            owner = self._unique_values[col].get(key)
            if owner is not None and owner != existing_idx:
                raise ValueError(f"UNIQUE constraint failed: {col}={row.get(col)!r}")

    def _unique_insert(self, idx: int, row: Dict[str, Any]) -> None:
        for col in self._unique_columns:
            self._unique_values[col][self._unique_key(row.get(col))] = idx

    def _unique_delete(self, idx: int, row: Optional[Dict[str, Any]]) -> None:
        if row is None:
            return
        for col in self._unique_columns:
            key = self._unique_key(row.get(col))
            if self._unique_values[col].get(key) == idx:
                del self._unique_values[col][key]

    def _unique_update(self, idx: int, old_row: Dict[str, Any], new_row: Dict[str, Any]) -> None:
        self._unique_delete(idx, old_row)
        try:
            self._validate_unique(new_row, existing_idx=idx)
            self._unique_insert(idx, new_row)
        except Exception:
            self._unique_insert(idx, old_row)
            raise

    def _rebuild_unique_constraints(self) -> None:
        self._unique_values = {c: {} for c in self._unique_columns}
        if not self._unique_columns:
            # No unique/primary-key columns — skip the full-table scan that
            # would otherwise run on every open/fsck/compact.
            return
        for idx, row in self:
            self._validate_unique(row, existing_idx=idx)
            self._unique_insert(idx, row)

    def _create(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        schema_json = json.dumps(self.schema.to_json()).encode("utf-8")
        schema_offset = _HEADER_SIZE + len(schema_json)
        bitmap_size = self._rows_per_slab

        with open(self.path, "wb") as f:
            f.write(_pack_header(schema_offset, self.page_size))
            f.write(schema_json)
            f.write(b"\x00" * bitmap_size)
            f.write(b"\x00" * self.page_size)
            f.flush()
            os.fsync(f.fileno())

        self._load()

    def _load(self) -> None:
        file_size = os.path.getsize(self.path)
        if file_size < _HEADER_SIZE:
            raise ValueError(f"File too small ({file_size} bytes) — not a valid SnapDB")

        self._file = open(self.path, "r+b")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)

        version, page_size, flags, schema_offset = _unpack_header(self._mm[:_HEADER_SIZE])
        self.page_size = page_size

        schema_json = json.loads(self._mm[_HEADER_SIZE:schema_offset].decode("utf-8"))
        self.schema = Schema.from_json(schema_json)
        self._rows_per_slab = page_size // self.schema.row_width

        # On-disk layout after the header+schema is a repeating unit of
        # [bitmap (rows_per_slab bytes)][slab (page_size bytes)], appended once
        # per slab by _create/_expand. Read it back with that exact geometry.
        bitmap_size = self._rows_per_slab
        unit = bitmap_size + self.page_size
        file_size = len(self._mm)
        slab_count = (file_size - schema_offset) // unit

        # Remember the geometry so close()/flush() can persist liveness bitmaps.
        self._schema_offset = schema_offset
        self._bitmap_size = bitmap_size
        self._unit = unit

        # `_count` is the high-water insert position (NOT the live count): a
        # deleted row keeps its slot. Slabs fill in order and a slab only grows
        # once full, so every slab except the last is at capacity; the last
        # slab's high-water is persisted in the header `flags` field.
        for i in range(slab_count):
            bitmap_start = schema_offset + i * unit
            slab_off = bitmap_start + bitmap_size
            slab = Slab(self.schema, self._mm, slab_off, self._rows_per_slab,
                        crypt=self._crypt_row if self._crypto_key is not None else None)
            slab._live = bytearray(self._mm[bitmap_start : bitmap_start + bitmap_size])
            if i < slab_count - 1:
                slab._count = self._rows_per_slab
            else:
                slab._count = min(flags, self._rows_per_slab)
            # This slab's rows came straight from disk, not through
            # insert()/batch_insert(), so its zone map (built lazily on first
            # range query — see Slab.ensure_zone_map) knows nothing yet.
            slab._zone_built = False
            # bytearray.count(1) is a single C-level call — the previous
            # Python for-loop popcount was the dominant cost of opening a
            # large file (O(total capacity) interpreter iterations).
            self._total_rows += slab._live.count(1)
            self._slabs.append(slab)

    def _recover_wal(self) -> None:
        """Replay committed transactional row operations that were not checkpointed."""
        if self.is_columnar() or self._wal is None:
            return
        pending: List[Dict[str, Any]] = []
        committed: List[Dict[str, Any]] = []
        for record in self._wal.replay() or ():
            op = record.get("op")
            if op == "begin":
                pending = []
            elif op == "commit":
                committed.extend(pending)
                pending = []
            elif op == "rollback":
                pending = []
            elif pending is not None:
                pending.append(record)
        if not committed:
            return
        # Net out per-index operations first: an insert/update superseded by a
        # later delete of the same idx must not be replayed at all — replaying
        # the insert would append at the high-water mark and resurrect the
        # deleted row at a new index (get(idx) can't tell "never applied" from
        # "applied then deleted").
        netted: List[Dict[str, Any]] = []
        for record in committed:
            if record.get("op") == "delete":
                idx = record.get("idx")
                netted = [r for r in netted if r.get("idx") != idx]
            netted.append(record)
        committed = netted
        self._recovering = True
        try:
            for record in committed:
                op = record.get("op")
                if op == "insert":
                    idx = record.get("idx")
                    if idx is None or self.get(idx) is None:
                        self.insert(record["row"])
                elif op == "update":
                    idx = record["idx"]
                    if self.get(idx) is not None:
                        self.update(idx, record["row"])
                elif op == "delete":
                    idx = record["idx"]
                    if self.get(idx) is not None:
                        self.delete(idx)
            self.flush()
            self._wal.clear()
        finally:
            self._recovering = False

    def _load_columnar(self) -> None:
        with open(self.path, "rb") as f:
            payload = pickle.loads(self._decrypt_blob(f.read()))
        rows = payload.get("rows", [])
        expected = 0
        for idx, row in rows:
            while expected < idx:
                hole = self._table.insert({})
                self._table.delete(hole)
                expected += 1
            self._table.insert(row)
            expected = idx + 1

    def _persist_columnar(self) -> None:
        if not self.is_columnar():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [(idx, row) for idx, row in self]
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            payload = pickle.dumps({"rows": rows}, protocol=pickle.HIGHEST_PROTOCOL)
            f.write(self._encrypt_blob(payload))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _grow_to(self, new_slabs: int) -> None:
        """Append ``new_slabs`` empty slabs to the file in a SINGLE truncate +
        remap, instead of one truncate/reopen per slab. Each slab adds an
        interleaved [bitmap][slab-page] unit (the layout _load expects)."""
        if new_slabs <= 0:
            return
        self._mm.flush()
        old_size = len(self._mm)
        bitmap_size = self._rows_per_slab
        unit = bitmap_size + self.page_size
        new_size = old_size + new_slabs * unit

        self._file.close()
        with open(self.path, "r+b") as f:
            f.truncate(new_size)

        self._file = open(self.path, "r+b")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)

        # Re-point existing slabs at the freshly-mapped buffer so every slab
        # reads/writes through the same (current) mapping.
        for s in self._slabs:
            s._mm = self._mm

        zero = b"\x00" * bitmap_size
        for j in range(new_slabs):
            unit_off = old_size + j * unit
            self._mm[unit_off:unit_off + bitmap_size] = zero  # liveness bitmap
            slab = Slab(self.schema, self._mm, unit_off + bitmap_size, self._rows_per_slab,
                        crypt=self._crypt_row if self._crypto_key is not None else None)
            self._slabs.append(slab)

    def _expand(self) -> None:
        self._grow_to(1)

    def is_columnar(self) -> bool:
        return self._storage_type == "columnar"

    # ── Metrics helpers ──────────────────────────────────────────────────────

    def _m_inc(self, metric: str, value: int = 1) -> None:
        if self._metrics is not None:
            self._metrics.inc(metric, value)

    def _m_lat(self, metric: str, seconds: float) -> None:
        if self._metrics is not None:
            self._metrics.add_latency(metric, seconds)

    def _m_time(self, metric_name: str):
        """Context manager for timing operations (no-op without metrics)."""
        if self._metrics is None:
            return _NULL_CTX
        return _metric_timer(self, metric_name)

    # ── CDC helper ───────────────────────────────────────────────────────────

    def _cdc_log(self, op: str, idx: int, row: Optional[Dict] = None,
                 old_row: Optional[Dict] = None) -> None:
        if self._cdc is not None:
            self._cdc.append(op, idx, row, old_row)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def insert(self, row: Dict[str, Any]) -> int:
        with self._m_time("insert"):
            with self._lock:
                return self._insert_locked(row)

    def _insert_locked(self, row: Dict[str, Any]) -> int:
            self._check_row_shape(row)
            self._validate_unique(row)
            if self.is_columnar():
                idx = self._table.insert(row)
                self._cdc_log("insert", idx, row)
                self._index_insert(idx, row)
                self._unique_insert(idx, row)
                return idx
            # Slabs fill strictly in order and deleted slots are never reused
            # (Slab._count is a high-water mark), so only the LAST slab can
            # have space — no need to scan from slab 0 on every insert.
            if not self._slabs or self._slabs[-1].is_full:
                self._expand()
            local_idx = self._slabs[-1].insert(row)
            global_idx = (len(self._slabs) - 1) * self._rows_per_slab + local_idx
            self._total_rows += 1
            self._cdc_log("insert", global_idx, row)
            self._index_insert(global_idx, row)
            self._unique_insert(global_idx, row)
            if self._in_tx:
                self._tx_state.append(("insert", global_idx, dict(row)))
                if self._wal is not None and not self._recovering:
                    self._wal.append("insert", idx=global_idx, row=dict(row))
            return global_idx

    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        """Insert multiple rows at once — much faster than individual inserts.

        Returns the number of rows inserted (same for row and columnar storage).
        """
        with self._lock:
            return self._batch_insert_locked(rows)

    def _batch_insert_locked(self, rows: List[Dict[str, Any]]) -> int:
        for row in rows:
            self._check_row_shape(row)
        seen = {col: set() for col in self._unique_columns}
        for row in rows:
            self._validate_unique(row)
            for col in self._unique_columns:
                key = self._unique_key(row.get(col))
                if key in seen[col]:
                    raise ValueError(f"UNIQUE constraint failed: {col}={row.get(col)!r}")
                seen[col].add(key)
        if self.is_columnar():
            start_idx = self._table.batch_insert(rows)
            if getattr(self._table, "_indexes", None):
                for offset, row in enumerate(rows):
                    self._index_insert(start_idx + offset, row)
            for offset, row in enumerate(rows):
                self._unique_insert(start_idx + offset, row)
            if self._cdc is not None:
                for offset, row in enumerate(rows):
                    self._cdc_log("insert", start_idx + offset, row)
            return len(rows)
        n = len(rows)
        if n == 0:
            return 0
        has_hash_indexes = self._indexes is not None and len(self._indexes._indexes) > 0
        has_range_indexes = bool(self._range_indexes)
        maintain_per_row = (
            has_hash_indexes
            or has_range_indexes
            or bool(self._unique_columns)
            or self._in_tx
            or self._cdc is not None
        )

        # Pre-grow the file ONCE for the whole batch (was one truncate/remap per
        # slab — O(n/rows_per_slab) remaps). Slabs fill in order so only the last
        # slab is ever partial; growing the exact number of units keeps that
        # invariant (which persistence relies on).
        free = sum(s.capacity - s.count for s in self._slabs)
        if n > free:
            needed = n - free
            self._grow_to((needed + self._rows_per_slab - 1) // self._rows_per_slab)

        total_inserted = 0
        # Advance through slabs from the first non-full one (no rescans). A
        # cursor into `rows` avoids re-slicing the remaining tail once per slab
        # (which was O(n^2) list copying for large batches).
        pos = 0
        slab_idx = 0
        while slab_idx < len(self._slabs) and self._slabs[slab_idx].is_full:
            slab_idx += 1
        while pos < n:
            if slab_idx >= len(self._slabs):
                self._grow_to(1)  # safety net; pre-grow should make this unreachable
            slab = self._slabs[slab_idx]
            space = slab.capacity - slab.count
            if space <= 0:
                slab_idx += 1
                continue
            chunk = rows[pos:pos + space]
            local_start = slab.batch_insert(chunk)
            if maintain_per_row:
                base = slab_idx * self._rows_per_slab + local_start
                for offset, row in enumerate(chunk):
                    gidx = base + offset
                    self._cdc_log("insert", gidx, row)
                    if has_hash_indexes or has_range_indexes:
                        self._index_insert(gidx, row)
                    if self._unique_columns:
                        self._unique_insert(gidx, row)
                    # Record undo state so a batch inside transaction() rolls
                    # back like single inserts do (atomicity).
                    if self._in_tx:
                        self._tx_state.append(("insert", gidx, dict(row)))
                        if self._wal is not None and not self._recovering:
                            self._wal.append("insert", idx=gidx, row=dict(row))
            self._total_rows += len(chunk)
            total_inserted += len(chunk)
            pos += len(chunk)
            slab_idx += 1
        return total_inserted

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        with self._m_time("get"):
            if self.is_columnar():
                return self._table.get(idx)
            if idx < 0:
                # Floor division would give slab_idx=-1, which Python indexing
                # silently resolves to the LAST slab — an unrelated row.
                return None
            slab_idx = idx // self._rows_per_slab
            local_idx = idx % self._rows_per_slab
            if slab_idx >= len(self._slabs):
                return None
            return self._slabs[slab_idx].get(local_idx)

    def get_raw(self, idx: int) -> Optional[memoryview]:
        with self._m_time("get_raw"):
            if self.is_columnar():
                return None
            if idx < 0:
                return None
            slab_idx = idx // self._rows_per_slab
            local_idx = idx % self._rows_per_slab
            if slab_idx >= len(self._slabs):
                return None
            if self._crypto_key is not None:
                row = self._slabs[slab_idx].get(local_idx)
                if row is None:
                    return None
                return memoryview(self.schema.encode_row(row))
            return self._slabs[slab_idx].get_raw(local_idx)

    # ── Hash Index (in-memory, optional) ─────────────────────────────────────
    #
    # Row storage uses the O(1) MultiIndex (index.py). Columnar storage uses a
    # plain ``value -> row_idx`` dict held on the table. Both are kept in sync
    # by the _index_insert / _index_update / _index_delete helpers below so that
    # find()/lookup() never go stale after a write.

    def _note_access(self, column: str) -> None:
        """Track equality-query frequency and auto-build an index past the
        threshold (issue #6). No-op unless auto_index was enabled."""
        if not self._auto_index:
            return
        if self.is_columnar():
            existing = getattr(self._table, "_indexes", None)
            if existing and column in existing:
                return
        elif self._indexes is not None and column in self._indexes:
            return
        count = self._access_counts.get(column, 0) + 1
        self._access_counts[column] = count
        if count >= self._auto_index_threshold:
            try:
                self.create_index(column)
            except (KeyError, ValueError, RuntimeError):
                pass

    def lookup(self, column_name: str, value: Any) -> Optional[Dict[str, Any]]:
        """Fast lookup by indexed column — falls back to a scan if unindexed."""
        self._note_access(column_name)
        norm = _norm_value(value)
        if self.is_columnar():
            indexes = getattr(self._table, "_indexes", None)
            if indexes is not None and column_name in indexes:
                # Index keys are decoded values; query with the normalized value
                # and return the first (smallest-index) match to match the scan.
                rows = indexes[column_name].get(norm)
                return self._table.get(min(rows)) if rows else None
            col = self._table.columns[column_name]
            for i in range(self._table._row_count):
                if col._nullmask[i] == 0 and col[i] == norm:
                    return self._table.get(i)
            return None

        if self._indexes is not None and column_name in self._indexes:
            row_indices = self._indexes.lookup(**{column_name: value})
            if row_indices:
                return self.get(row_indices[0])
            return None
        for idx, row in self:
            if row.get(column_name) == norm:
                return row
        return None

    def _index_insert(self, idx: int, row: Dict[str, Any]) -> None:
        """Maintain indexes after an insert."""
        if self.is_columnar():
            indexes = getattr(self._table, "_indexes", None)
            if indexes:
                for col_name in indexes:
                    val = self._table.columns[col_name][idx]
                    if val is not None:
                        indexes[col_name].setdefault(val, set()).add(idx)
            return
        if self._indexes is not None:
            for hidx in self._indexes._indexes.values():
                hidx.insert(idx, row)
        for ridx in self._range_indexes.values():
            ridx.insert(idx, row)

    def _index_update(self, idx: int, old_row: Optional[Dict[str, Any]],
                      new_row: Dict[str, Any]) -> None:
        """Maintain indexes after an update (old_row may be None)."""
        if self.is_columnar():
            indexes = getattr(self._table, "_indexes", None)
            if indexes:
                for col_name, idx_map in indexes.items():
                    if old_row is not None:
                        old_val = old_row.get(col_name)
                        if old_val is not None and old_val in idx_map:
                            idx_map[old_val].discard(idx)
                            if not idx_map[old_val]:
                                del idx_map[old_val]
                    new_val = self._table.columns[col_name][idx]
                    if new_val is not None:
                        idx_map.setdefault(new_val, set()).add(idx)
            return
        if self._indexes is not None and old_row is not None:
            for hidx in self._indexes._indexes.values():
                hidx.update(idx, old_row, new_row)
        if old_row is not None:
            for ridx in self._range_indexes.values():
                ridx.update(idx, old_row, new_row)

    def _index_delete(self, idx: int, row: Optional[Dict[str, Any]]) -> None:
        """Maintain indexes after a delete."""
        if self.is_columnar():
            indexes = getattr(self._table, "_indexes", None)
            if indexes and row is not None:
                for col_name, idx_map in indexes.items():
                    val = row.get(col_name)
                    if val is not None and val in idx_map:
                        idx_map[val].discard(idx)
                        if not idx_map[val]:
                            del idx_map[val]
            return
        if self._indexes is not None and row is not None:
            for hidx in self._indexes._indexes.values():
                hidx.delete(idx, row)
        if row is not None:
            for ridx in self._range_indexes.values():
                ridx.delete(idx, row)

    def update(self, idx: int, row: Dict[str, Any]) -> None:
        with self._m_time("update"):
            with self._lock:
                self._update_locked(idx, row)

    def _update_locked(self, idx: int, row: Dict[str, Any]) -> None:
            if self.is_columnar():
                old = self._table.get(idx)
                if old is None:
                    raise KeyError(f"Row {idx} not found")
                merged = {**old, **row}
                self._check_row_shape(merged)
                self._unique_update(idx, old, merged)
                self._table.update(idx, row)
                self._cdc_log("update", idx, merged, old)
                self._index_update(idx, old, merged)
                return
            if idx < 0:
                raise KeyError(f"Row {idx} not found")
            slab_idx = idx // self._rows_per_slab
            local_idx = idx % self._rows_per_slab
            if slab_idx >= len(self._slabs):
                raise KeyError(f"Row {idx} not found")
            old_row = self._slabs[slab_idx].get(local_idx)
            if old_row is None:
                raise KeyError(f"Row {idx} not found")
            merged = {**old_row, **row}
            self._check_row_shape(merged)
            self._unique_update(idx, old_row, merged)
            self._slabs[slab_idx].update(local_idx, merged)
            self._cdc_log("update", idx, merged, old_row)
            self._index_update(idx, old_row, merged)
            if self._in_tx:
                self._tx_state.append(("update", idx, old_row))
                if self._wal is not None and not self._recovering:
                    self._wal.append("update", idx=idx, row=merged)

    def delete(self, idx: int) -> None:
        with self._m_time("delete"):
            with self._lock:
                self._delete_locked(idx)

    def _delete_locked(self, idx: int) -> None:
            if self.is_columnar():
                old = self._table.get(idx)
                self._table.delete(idx)
                self._cdc_log("delete", idx, None, old)
                self._index_delete(idx, old)
                self._unique_delete(idx, old)
                return
            if idx < 0:
                raise KeyError(f"Row {idx} not found")
            slab_idx = idx // self._rows_per_slab
            local_idx = idx % self._rows_per_slab
            if slab_idx >= len(self._slabs):
                raise KeyError(f"Row {idx} not found")
            old_row = self._slabs[slab_idx].get(local_idx)
            if old_row is None:
                raise KeyError(f"Row {idx} not found")
            self._slabs[slab_idx].delete(local_idx)
            self._total_rows -= 1
            self._cdc_log("delete", idx, None, old_row)
            self._index_delete(idx, old_row)
            self._unique_delete(idx, old_row)
            if self._in_tx:
                self._tx_state.append(("delete", idx, old_row))
                if self._wal is not None and not self._recovering:
                    self._wal.append("delete", idx=idx)

    def query(self, predicate: Callable[[Dict[str, Any]], bool]) -> Iterator[Tuple[int, Dict[str, Any]]]:
        with self._m_time("query"):
            for idx, row in self:
                if predicate(row):
                    yield idx, row

    def batch_update(self, predicate: Callable[[Dict[str, Any]], bool],
                     updates: Union[Dict[str, Any], Callable[[Dict[str, Any]], Dict[str, Any]]]) -> int:
        """Update all matching rows. Returns the number of rows updated."""
        with self._lock:
            if (self.is_columnar()
                    and isinstance(updates, dict)
                    and not self._unique_columns
                    and not getattr(self._table, "_indexes", None)
                    and self._cdc is None):
                return self._table.batch_update(predicate, updates)
            matches = [(idx, row) for idx, row in self if predicate(row)]
            for idx, row in matches:
                patch = updates(row) if callable(updates) else updates
                self._update_locked(idx, patch)
            return len(matches)

    def group_by(self, key_column: str, value_column: str, agg: str = "count") -> Dict[Any, Any]:
        """Group rows by one column and aggregate another column."""
        if self.is_columnar():
            return self._table.group_by(key_column, value_column, agg)
        if key_column not in self.schema._name_to_idx:
            raise KeyError(f"Unknown column: {key_column}")
        if value_column not in self.schema._name_to_idx:
            raise KeyError(f"Unknown column: {value_column}")
        groups: Dict[Any, List[Any]] = {}
        for _, row in self:
            val = row.get(value_column)
            if val is not None:
                groups.setdefault(row.get(key_column), []).append(val)

        out: Dict[Any, Any] = {}
        for key, vals in groups.items():
            if agg == "count":
                out[key] = len(vals)
            elif agg == "sum":
                out[key] = sum(vals)
            elif agg == "avg":
                out[key] = sum(vals) / len(vals) if vals else 0
            elif agg == "min":
                out[key] = min(vals)
            elif agg == "max":
                out[key] = max(vals)
            else:
                raise ValueError(f"Unsupported aggregate: {agg}")
        return out

    def join(self, other: "SnapDB", left_column: str, right_column: str,
             how: str = "inner") -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """In-memory equi-join between two SnapDB instances."""
        if how != "inner":
            raise ValueError("Only inner joins are supported")
        if left_column not in self.schema._name_to_idx:
            raise KeyError(f"Unknown column: {left_column}")
        if right_column not in other.schema._name_to_idx:
            raise KeyError(f"Unknown column: {right_column}")
        right_map: Dict[Any, List[Dict[str, Any]]] = {}
        for _, row in other:
            right_map.setdefault(row.get(right_column), []).append(row)
        result = []
        for _, left in self:
            for right in right_map.get(left.get(left_column), ()):
                result.append((left, right))
        return result

    def __len__(self) -> int:
        if self.is_columnar():
            return len(self._table)
        return self._total_rows

    def __iter__(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        if self.is_columnar():
            for i in range(len(self._table)):
                row = self._table.get(i)
                if row is not None:
                    yield i, row
            return
        for slab_idx, slab in enumerate(self._slabs):
            base = slab_idx * self._rows_per_slab
            for local_idx, row in slab.iter_rows():
                yield base + local_idx, row

    def iter_raw(self) -> Iterator[Tuple[int, memoryview]]:
        if self.is_columnar():
            return
        for slab_idx, slab in enumerate(self._slabs):
            base = slab_idx * self._rows_per_slab
            for local_idx, raw in slab.iter_raw():
                yield base + local_idx, raw

    def _persist_bitmaps(self) -> None:
        """Write each slab's liveness bitmap into the mmap so row liveness
        survives close()/reopen. Row payloads are already written to the mmap on
        insert; the in-memory ``_live`` bitmaps are what needs flushing.

        Also records the last slab's high-water row count in the header `flags`
        field so the insert position can be restored exactly on reopen.
        """
        if self.is_columnar() or self._mm is None or not self._slabs:
            return
        base = self._schema_offset
        unit = self._unit
        bs = self._bitmap_size
        for i, slab in enumerate(self._slabs):
            off = base + i * unit
            self._mm[off : off + bs] = bytes(slab._live)
        last_count = self._slabs[-1]._count if self._slabs else 0
        self._mm[:_HEADER_SIZE] = struct.pack(
            _HEADER_FMT, _MAGIC, _VERSION, self.page_size,
            last_count, self._schema_offset)

    def flush(self) -> None:
        """Persist liveness bitmaps and flush the mmap to disk (row storage)."""
        if self.is_columnar() or self._mm is None:
            return
        self._persist_bitmaps()
        self._mm.flush()

    def backup(self, destination: Union[str, Path]) -> Path:
        """Create a consistent file-copy backup and return its path."""
        dest = Path(destination)
        with self._lock:
            if self.is_columnar():
                self._persist_columnar()
            else:
                self.flush()
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path, dest)
        return dest

    # ── Named snapshots ────────────────────────────────────────────────────
    #
    # A pragmatic v1: each snapshot is a full materialized copy (built on the
    # same flush()+copy machinery as backup()), tracked by name in a small
    # JSON manifest next to the live file. This is NOT copy-on-write / MVCC —
    # there's no continuous history and no O(1) snapshot — it's a named,
    # restorable point-in-time backup you can open and query independently.
    # True copy-on-write time travel (redirect writes to new slabs, version
    # the liveness bitmaps) is real future work; this ships the useful part
    # (restorable named history) now without committing to that on-disk
    # format decision.

    def _snapshot_dir(self) -> Path:
        return self.path.parent / f"{self.path.stem}.snapshots"

    @staticmethod
    def _read_snapshot_manifest(snap_dir: Path) -> Dict[str, Dict[str, Any]]:
        mpath = snap_dir / "manifest.json"
        if mpath.exists():
            return json.loads(mpath.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _write_snapshot_manifest(snap_dir: Path, manifest: Dict[str, Dict[str, Any]]) -> None:
        # Atomic write: a crash mid-write must never leave a half-written
        # (unparseable) manifest.json, which would make every existing
        # snapshot unlistable. Write to a temp file, then os.replace().
        mpath = snap_dir / "manifest.json"
        tmp = snap_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, mpath)

    def snapshot(self, name: Optional[str] = None) -> str:
        """Save a named, point-in-time copy of the database. Returns the
        snapshot name used (auto-generated as ``snap_<n>_<timestamp>`` if
        omitted). Calling this again with the same name overwrites that slot.

        Read it back with :meth:`open_snapshot`; list existing snapshots with
        :meth:`list_snapshots`.
        """
        with self._lock:
            if name is None:
                self._snapshot_seq = getattr(self, "_snapshot_seq", 0) + 1
                name = f"snap_{self._snapshot_seq}_{int(time.time())}"
            snap_dir = self._snapshot_dir()
            dest = snap_dir / f"{name}.snap"
            self.backup(dest)  # flush()/persist + copy — same durability as backup()
            manifest = self._read_snapshot_manifest(snap_dir)
            manifest[name] = {
                "file": dest.name,
                "created_at": time.time(),
                "rows": len(self),
                "storage_type": self._storage_type,
            }
            self._write_snapshot_manifest(snap_dir, manifest)
        return name

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Return metadata for every snapshot taken from this database,
        newest first: ``{"name", "file", "created_at", "rows", "storage_type"}``."""
        manifest = self._read_snapshot_manifest(self._snapshot_dir())
        out = [{"name": name, **meta} for name, meta in manifest.items()]
        out.sort(key=lambda e: e["created_at"], reverse=True)
        return out

    def open_snapshot(self, name: str,
                      encryption_key: Union[str, bytes, None] = None) -> "SnapDB":
        """Open a previously taken snapshot as an independent SnapDB instance
        — a separate materialized copy. Writes to it never affect the live
        database or the manifest entry; the caller must close() it. Pass
        ``encryption_key`` if the live database is encrypted (the key itself
        is never persisted, so it can't be recovered automatically)."""
        snap_dir = self._snapshot_dir()
        manifest = self._read_snapshot_manifest(snap_dir)
        if name not in manifest:
            raise KeyError(f"No snapshot named {name!r}")
        snap_path = snap_dir / manifest[name]["file"]
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot file missing: {snap_path}")
        return SnapDB(snap_path, self.schema, storage_type=manifest[name]["storage_type"],
                      encryption_key=encryption_key)

    def drop_snapshot(self, name: str) -> None:
        """Delete a named snapshot (file + manifest entry)."""
        snap_dir = self._snapshot_dir()
        manifest = self._read_snapshot_manifest(snap_dir)
        if name not in manifest:
            raise KeyError(f"No snapshot named {name!r}")
        snap_path = snap_dir / manifest[name]["file"]
        if snap_path.exists():
            snap_path.unlink()
        del manifest[name]
        self._write_snapshot_manifest(snap_dir, manifest)

    def fsck(self) -> Dict[str, Any]:
        """Run lightweight consistency checks and return a report."""
        issues: List[str] = []
        iter_count = 0
        try:
            for idx, row in self:
                iter_count += 1
                if row is None:
                    issues.append(f"iterator returned None at row {idx}")
        except Exception as exc:
            issues.append(f"iteration failed: {exc}")

        if not self.is_columnar():
            live_count = 0
            for slab in self._slabs:
                live_count += slab._live[:slab._count].count(1)
            if live_count != self._total_rows:
                issues.append(f"metadata row count {self._total_rows} != live bitmap count {live_count}")
            if iter_count != self._total_rows:
                issues.append(f"metadata row count {self._total_rows} != iterated rows {iter_count}")
        elif iter_count > len(self._table):
            issues.append(f"iterated rows {iter_count} exceed table row count {len(self._table)}")

        try:
            self._rebuild_unique_constraints()
        except Exception as exc:
            issues.append(f"constraint check failed: {exc}")

        return {"ok": not issues, "issues": issues, "rows": iter_count}

    def repair(self) -> Dict[str, Any]:
        """Best-effort repair: compact storage and rebuild in-memory metadata."""
        reclaimed = self.compact()
        report = self.fsck()
        report["reclaimed"] = reclaimed
        return report

    def _close_mapping_only(self) -> None:
        if self._slabs:
            for slab in self._slabs:
                slab._mm = None
            self._slabs = []
        if self._mm is not None:
            self._mm.flush()
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def compact(self) -> int:
        """Rewrite row storage with only live rows. Returns bytes reclaimed."""
        with self._lock:
            if self.is_columnar():
                before = self.path.stat().st_size if self.path.exists() else 0
                self._persist_columnar()
                after = self.path.stat().st_size if self.path.exists() else 0
                return max(0, before - after)

            rows = [row for _, row in self]
            index_columns = list(self._indexes._indexes.keys()) if self._indexes else []
            range_index_columns = list(self._range_indexes.keys())
            before = self.path.stat().st_size if self.path.exists() else 0
            schema_json = json.dumps(self.schema.to_json()).encode("utf-8")
            schema_offset = _HEADER_SIZE + len(schema_json)
            bitmap_size = self._rows_per_slab
            slab_count = max(1, (len(rows) + self._rows_per_slab - 1) // self._rows_per_slab)
            last_count = len(rows) - (slab_count - 1) * self._rows_per_slab
            if len(rows) == 0:
                last_count = 0
            elif last_count == 0:
                last_count = self._rows_per_slab

            tmp = self.path.with_suffix(self.path.suffix + ".compact")
            with open(tmp, "wb") as f:
                f.write(struct.pack(_HEADER_FMT, _MAGIC, _VERSION, self.page_size,
                                    last_count, schema_offset))
                f.write(schema_json)
                pos = 0
                for _ in range(slab_count):
                    chunk = rows[pos:pos + self._rows_per_slab]
                    live = bytearray(bitmap_size)
                    page = bytearray(self.page_size)
                    for i, row in enumerate(chunk):
                        live[i] = 1
                        raw = self.schema.encode_row(row)
                        start = i * self.schema.row_width
                        if self._crypto_key is not None:
                            absolute = schema_offset + (pos // self._rows_per_slab) * (
                                bitmap_size + self.page_size) + bitmap_size + start
                            raw = self._crypt_row(raw, absolute)
                        page[start:start + len(raw)] = raw
                    f.write(live)
                    f.write(page)
                    pos += len(chunk)
                f.flush()
                os.fsync(f.fileno())

            self._close_mapping_only()
            os.replace(tmp, self.path)
            self._slabs = []
            self._total_rows = 0
            self._load()
            if self._indexes is not None:
                self._indexes = MultiIndex()
                for column in index_columns:
                    self.create_index(column)
            self._range_indexes = {}
            for column in range_index_columns:
                self.create_range_index(column)
            self._rebuild_unique_constraints()
            if self._wal is not None:
                self._wal.clear()
            after = self.path.stat().st_size if self.path.exists() else 0
            return max(0, before - after)

    def close(self) -> None:
        import gc
        with self._lock:
            if self._closed:
                return
            # Closing an open transaction is a rollback, not an implicit commit.
            if self._in_tx:
                if self._wal is not None:
                    self._wal.rollback()
                self._tx_rollback()
            if self.is_columnar():
                self._persist_columnar()
            # Persist liveness bitmaps and flush BEFORE releasing slabs/mmap so
            # the database can be reopened with all rows intact.
            self._persist_bitmaps()
            # Slabs each hold a reference to the mmap, which would keep the
            # mapping alive (and the file locked on Windows) even after we drop
            # our own reference. Release those first so mmap.close() can unmap.
            if self._slabs:
                for slab in self._slabs:
                    slab._mm = None
                self._slabs = []
            if self._mm is not None:
                self._mm.flush()
                try:
                    self._mm.close()
                except BufferError:
                    # A caller still holds a memoryview (e.g. from get_raw); fall
                    # back to dropping our reference and letting GC unmap it.
                    gc.collect()
                    try:
                        self._mm.close()
                    except BufferError:
                        pass
                self._mm = None
            if self._file is not None:
                self._file.close()
                self._file = None
            if self._wal is not None:
                # Everything was just persisted (bitmaps + mmap flush), so any
                # remaining WAL records are stale; replaying them on the next
                # open would re-apply committed ops (e.g. resurrect a row that
                # was inserted and then deleted in a committed transaction).
                self._wal.clear()
                self._wal.close()
            if self._cdc is not None:
                self._cdc.close()
            self._closed = True
            self._release_file_lock()
            gc.collect()

    # ── Columnar-specific methods ─────────────────────────────────────────────

    def select(self, where=None, columns=None, limit=None, offset=0):
        """Select with filter/projection/limit/offset. Columnar only."""
        if not self.is_columnar():
            raise RuntimeError("Select requires columnar storage")
        return self._table.select(where=where, columns=columns, limit=limit, offset=offset)

    def aggregate(self, column_name: str, agg: str = "sum", where=None, use_numpy=None):
        """Aggregate on a column. Columnar only.

        When NumPy is installed it is used automatically for plain numeric
        columns (issue #14); pass use_numpy=False to force the pure-Python path.
        """
        if not self.is_columnar():
            raise RuntimeError("Aggregate requires columnar storage")
        return self._table.aggregate(column_name, agg, where, use_numpy=use_numpy)

    def select_column(self, column_name: str) -> List[Any]:
        """Fast extraction of a single column. Columnar only."""
        if not self.is_columnar():
            raise RuntimeError("select_column requires columnar storage")
        return self._table.select_column(column_name)

    def select_where(self, conditions, columns=None, limit=None, offset=0,
                     combine="and", use_numpy=None):
        """Vectorized multi-condition filter (columnar only). See
        :meth:`ColumnarTable.select_where`. NumPy-accelerated when available."""
        if not self.is_columnar():
            raise RuntimeError("select_where requires columnar storage")
        # Auto-indexing watches single-column equality filters too.
        if self._auto_index:
            for col, op, _ in self._table._normalize_conditions(conditions):
                if op in ("eq", "=="):
                    self._note_access(col)
        return self._table.select_where(conditions, columns=columns, limit=limit,
                                        use_numpy=use_numpy,
                                        offset=offset, combine=combine)

    def count_where(self, conditions, combine="and", use_numpy=None):
        """Count rows matching conditions without materializing them (columnar
        only). NumPy-accelerated when available. See
        :meth:`ColumnarTable.count_where`."""
        if not self.is_columnar():
            raise RuntimeError("count_where requires columnar storage")
        return self._table.count_where(conditions, combine=combine, use_numpy=use_numpy)

    def to_numpy(self, column_name: str, zero_copy: bool = False):
        """Export a column as a NumPy array (columnar only; requires numpy)."""
        if not self.is_columnar():
            raise RuntimeError("to_numpy requires columnar storage")
        return self._table.to_numpy(column_name, zero_copy=zero_copy)

    def column_buffer(self, column_name: str) -> memoryview:
        """Zero-copy buffer over a plain numeric column (columnar only)."""
        if not self.is_columnar():
            raise RuntimeError("column_buffer requires columnar storage")
        return self._table.column_buffer(column_name)

    def memory_usage(self) -> int:
        """Memory usage in bytes (columnar only)."""
        if not self.is_columnar():
            raise RuntimeError("memory_usage requires columnar storage")
        return self._table.memory_usage()

    # ── Row-only methods (indexing, transactions) ────────────────────────────

    def create_index(self, column: str) -> None:
        """Build an in-memory hash index on a column for O(1) lookups.

        Kept in sync automatically on insert/update/delete. Works for both row
        and columnar storage.
        """
        if self.is_columnar():
            if column not in self._table.columns:
                raise KeyError(f"Unknown column: {column}")
            if not hasattr(self._table, "_indexes"):
                self._table._indexes = {}
            col = self._table.columns[column]
            # value -> set of row indices (handles duplicates; lookup returns the
            # smallest index so it agrees with a first-match scan).
            index: Dict[Any, set] = {}
            for i in range(self._table._row_count):
                if col._nullmask[i] == 0:
                    index.setdefault(col[i], set()).add(i)
            self._table._indexes[column] = index
            return
        if self._indexes is None:
            raise RuntimeError("Indexing not available (index.py missing)")
        if column in self._indexes:
            return
        self._indexes.create(column)
        hidx = self._indexes._indexes[column]
        for idx, row in self:
            hidx.insert(idx, row)

    def drop_index(self, column: str) -> None:
        if self.is_columnar():
            raise RuntimeError("Indexing not supported in columnar storage")
        if self._indexes is not None and column in self._indexes:
            self._indexes.drop(column)

    def create_range_index(self, column: str) -> None:
        """Build an in-memory sorted index for range lookups on a row-store column."""
        if self.is_columnar():
            raise RuntimeError("Range indexes are row-storage only")
        if RangeIndex is None:
            raise RuntimeError("Range indexing not available (index.py missing)")
        if column not in self.schema._name_to_idx:
            raise KeyError(f"Unknown column: {column}")
        if column in self._range_indexes:
            return
        idx = RangeIndex(column)
        idx.build(list(self))
        self._range_indexes[column] = idx

    def drop_range_index(self, column: str) -> None:
        self._range_indexes.pop(column, None)

    def range_find(self, column: str, low: Any = None, high: Any = None,
                   include_low: bool = True, include_high: bool = True) -> List[Dict[str, Any]]:
        """Return rows where ``low <= column <= high`` using a range index when present."""
        if self.is_columnar():
            raise RuntimeError("range_find is row-storage only")
        if column not in self.schema._name_to_idx:
            raise KeyError(f"Unknown column: {column}")
        idx = self._range_indexes.get(column)
        if idx is not None:
            return [row for row_idx in idx.range_lookup(low, high, include_low, include_high)
                    if (row := self.get(row_idx)) is not None]

        out = []
        for slab in self._slabs:
            slab.ensure_zone_map()
            if not slab.zone_overlaps(column, low, high, include_low, include_high):
                continue  # entire slab pruned — never decoded
            for _, row in slab.iter_rows():
                value = row.get(column)
                if value is None:
                    continue
                if low is not None and (value < low if include_low else value <= low):
                    continue
                if high is not None and (value > high if include_high else value >= high):
                    continue
                out.append(row)
        return out

    def find(self, **kwargs) -> List[Dict[str, Any]]:
        """Return all rows matching every column=value pair.

        Uses hash indexes when all queried columns are indexed; otherwise falls
        back to a scan (so it works without create_index()). With auto_index
        enabled, frequently-queried columns get indexed automatically.
        """
        if self.is_columnar():
            raise RuntimeError("Find by index not supported in columnar storage")
        if self._indexes is None:
            raise RuntimeError("Indexing not available (index.py missing)")
        if not kwargs:
            return []
        for column in kwargs:
            if column not in self.schema._name_to_idx:
                raise KeyError(f"Unknown column: {column}")
            self._note_access(column)
        if all(column in self._indexes for column in kwargs):
            row_indices = self._indexes.lookup(**kwargs)
            if row_indices is None:
                return []
            return [self.get(i) for i in row_indices]
        # Scan fallback when one or more columns are not indexed.
        wanted = {k: _norm_value(v) for k, v in kwargs.items()}
        return [row for _, row in self
                if all(row.get(k) == v for k, v in wanted.items())]

    @contextmanager
    def transaction(self):
        if self.is_columnar():
            raise RuntimeError("Transactions not supported in columnar storage")
        if self._wal is None:
            raise RuntimeError("WAL not available (wal.py missing)")
        self._lock.acquire()
        if self._in_tx:
            self._lock.release()
            raise RuntimeError("Nested transactions are not supported")
        self._wal.begin()
        self._in_tx = True
        self._tx_state = []
        try:
            yield self
            self._wal.commit()
            self._in_tx = False
            self._tx_state = []
        except Exception:
            self._wal.rollback()
            self._tx_rollback()
            raise
        finally:
            self._lock.release()

    def _tx_rollback(self) -> None:
        if not self._in_tx:
            return
        # Roll back in reverse order. We must clear _in_tx first so the
        # index-maintenance helpers below don't try to re-record undo entries.
        self._in_tx = False
        for op, idx, data in reversed(self._tx_state):
            if op == "insert":
                slab_idx = idx // self._rows_per_slab
                local_idx = idx % self._rows_per_slab
                if slab_idx < len(self._slabs):
                    self._slabs[slab_idx].delete(local_idx)
                    self._total_rows -= 1
                    self._index_delete(idx, data)
                    self._unique_delete(idx, data)
            elif op == "update" and data is not None:
                slab_idx = idx // self._rows_per_slab
                local_idx = idx % self._rows_per_slab
                if slab_idx < len(self._slabs):
                    current = self._slabs[slab_idx].get(local_idx)
                    self._slabs[slab_idx].update(local_idx, data)
                    self._index_update(idx, current, data)
                    if current is not None:
                        self._unique_update(idx, current, data)
            elif op == "delete" and data is not None:
                slab_idx = idx // self._rows_per_slab
                local_idx = idx % self._rows_per_slab
                if slab_idx < len(self._slabs) and local_idx < self._slabs[slab_idx]._capacity:
                    raw = self.schema.encode_row(data)
                    start = self._slabs[slab_idx]._row_offset(local_idx)
                    if self._crypto_key is not None:
                        raw = self._crypt_row(raw, start)
                    mv = memoryview(self._mm)[start : start + len(raw)]
                    mv[:] = raw
                    self._slabs[slab_idx]._live[local_idx] = 1
                    self._total_rows += 1
                    self._index_insert(idx, data)
                    self._unique_insert(idx, data)
        self._in_tx = False
        self._tx_state = []
        if self._wal is not None:
            self._wal._buffer.clear()

    def checkpoint(self) -> None:
        if self.is_columnar():
            raise RuntimeError("Checkpoint not supported in columnar storage")
        # Persist liveness + data so a crash after checkpoint reopens cleanly.
        self.flush()
        if self._wal is not None:
            self._wal.clear()

    def __repr__(self) -> str:
        if self.is_columnar():
            return f"SnapDB({self.path!r}, storage=columnar, rows={len(self._table)})"
        indexes = list(self._indexes._indexes.keys()) if self._indexes else []
        return f"SnapDB({self.path!r}, rows={self._total_rows}, slabs={len(self._slabs)}, indexes={indexes})"
