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
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union
from contextlib import contextmanager

# v0.2.0 imports (optional)
try:
    from index import HashIndex, MultiIndex
except ImportError:
    HashIndex = None  # type: ignore
    MultiIndex = None  # type: ignore
try:
    from wal import WAL
except ImportError:
    WAL = None  # type: ignore

# v0.3.0 imports (optional)
try:
    from columnar import ColumnarTable
except ImportError:
    ColumnarTable = None  # type: ignore
try:
    from metrics import Metrics
except ImportError:
    Metrics = None  # type: ignore


# ── Type Mapping ───────────────────────────────────────────────────────────────

_TYPE_CODES: Dict[str, str] = {
    "i8": "b", "i16": "h", "i32": "i", "i64": "q",
    "u8": "B", "u16": "H", "u32": "I", "u64": "Q",
    "f32": "f", "f64": "d",
    "bool": "?",
}

_TYPE_SIZES: Dict[str, int] = {
    "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "u8": 1, "u16": 2, "u32": 4, "u64": 8,
    "f32": 4, "f64": 8,
    "bool": 1,
}


def _type_size(dtype: str) -> int:
    if dtype.startswith("bytes"):
        return int(dtype.split(":")[1])
    return _TYPE_SIZES[dtype]


def _struct_code(dtype: str) -> str:
    if dtype.startswith("bytes"):
        return f"{_type_size(dtype)}s"
    return _TYPE_CODES[dtype]


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
    width: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "width", _type_size(self.dtype))


class Schema:
    """Table schema definition."""

    def __init__(self, columns: List[ColumnDef]) -> None:
        self.columns = tuple(columns)
        self._name_to_idx = {c.name: i for i, c in enumerate(columns)}
        self._row_width = sum(c.width for c in columns)
        self._offsets = [0]
        for c in columns[:-1]:
            self._offsets.append(self._offsets[-1] + c.width)

    @property
    def row_width(self) -> int:
        return self._row_width

    def offset(self, name: str) -> int:
        return self._offsets[self._name_to_idx[name]]

    def index(self, name: str) -> int:
        return self._name_to_idx[name]

    def to_json(self) -> List[Dict[str, str]]:
        return [{"name": c.name, "dtype": c.dtype} for c in self.columns]

    @classmethod
    def from_json(cls, data: List[Dict[str, str]]) -> "Schema":
        return cls([ColumnDef(c["name"], c["dtype"]) for c in data])

    def to_columnar_schema(self) -> List[Tuple[str, str]]:
        """Convert to list of (name, dtype) tuples for ColumnarTable."""
        return [(c.name, c.dtype) for c in self.columns]

    def encode_row(self, row: Dict[str, Any]) -> bytes:
        parts = []
        for col in self.columns:
            val = row.get(col.name, 0)
            if col.dtype == "bool":
                parts.append(struct.pack("?", bool(val)))
            elif col.dtype.startswith("bytes"):
                raw = val if isinstance(val, bytes) else str(val).encode("utf-8")
                parts.append(struct.pack(f"{col.width}s", raw[:col.width].ljust(col.width, b"\x00")))
            else:
                parts.append(struct.pack(f"<{_struct_code(col.dtype)}", val))
        return b"".join(parts)

    def decode_row(self, buf: Union[bytes, memoryview]) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for col in self.columns:
            off = self.offset(col.name)
            raw = bytes(buf[off : off + col.width])
            if col.dtype == "bool":
                row[col.name] = struct.unpack("?", raw)[0]
            elif col.dtype.startswith("bytes"):
                row[col.name] = raw.rstrip(b"\x00").decode("utf-8", errors="replace")
            else:
                row[col.name] = struct.unpack(f"<{_struct_code(col.dtype)}", raw)[0]
        return row


# ── Slab (Segment) ───────────────────────────────────────────────────────────

class Slab:
    """Memory-mapped slab holding rows in a contiguous buffer."""

    def __init__(self, schema: Schema, mm: mmap.mmap, offset: int, capacity: int) -> None:
        self.schema = schema
        self._mm = mm
        self._offset = offset
        self._capacity = capacity
        self._count = 0
        self._live = bytearray(capacity)

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
        return self.schema.decode_row(memoryview(self._mm)[start:end])

    def get_raw(self, idx: int) -> Optional[memoryview]:
        if idx >= self._capacity or not self._live[idx]:
            return None
        start = self._row_offset(idx)
        end = start + self.schema.row_width
        return memoryview(self._mm)[start:end]

    def insert(self, row: Dict[str, Any]) -> int:
        if self.is_full:
            raise RuntimeError("Slab is full")
        idx = self._count
        self._count += 1
        self._live[idx] = 1
        raw = self.schema.encode_row(row)
        start = self._row_offset(idx)
        mv = memoryview(self._mm)[start : start + len(raw)]
        mv[:] = raw
        return idx

    def update(self, idx: int, row: Dict[str, Any]) -> None:
        if idx >= self._capacity or not self._live[idx]:
            raise KeyError(f"Row {idx} not found")
        raw = self.schema.encode_row(row)
        start = self._row_offset(idx)
        mv = memoryview(self._mm)[start : start + len(raw)]
        mv[:] = raw

    def delete(self, idx: int) -> None:
        if idx < self._capacity:
            self._live[idx] = 0

    def iter_rows(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        for i in range(self._count):
            if self._live[i]:
                yield i, self.get(i)

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

    def __init__(self, path: Union[str, Path], schema: Schema,
                 page_size: int = _DEFAULT_PAGE_SIZE,
                 storage_type: str = "row",
                 metrics: Optional[Metrics] = None,
                 cdc: Optional[CDCLog] = None) -> None:
        self.path = Path(path)
        self.schema = schema
        self.page_size = page_size
        self._storage_type = storage_type
        self._metrics = metrics
        self._cdc = cdc

        if storage_type == "columnar":
            if ColumnarTable is None:
                raise RuntimeError("Columnar storage not available (columnar.py missing)")
            self._table = ColumnarTable("columnar_store", schema.to_columnar_schema())
            self._indexes = None
            self._wal = None
            self._slabs = None
            self._mm = None
            self._file = None
            self._total_rows = 0
            self._in_tx = False
            self._tx_state: List[Tuple[str, int, Optional[Dict]]] = []
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
        self._wal: Optional["WAL"] = None
        if MultiIndex is not None:
            self._indexes = MultiIndex()
        if WAL is not None:
            wal_path = str(self.path).replace(".snap", ".wal")
            self._wal = WAL(wal_path)

        if self._rows_per_slab < 1:
            raise ValueError(f"Row size ({schema.row_width}) exceeds page size ({page_size})")

        if self.path.exists() and self.path.stat().st_size >= _HEADER_SIZE:
            self._load()
        else:
            self._create()

    def _create(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        schema_json = json.dumps(self.schema.to_json()).encode("utf-8")
        schema_offset = _HEADER_SIZE + len(schema_json)
        bitmap_size = self._rows_per_slab
        data_offset = schema_offset + bitmap_size
        total_size = data_offset + self.page_size

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

        bitmap_size = self._rows_per_slab
        data_offset = schema_offset + bitmap_size
        file_size = len(self._mm)
        slab_count = (file_size - data_offset) // self.page_size

        for i in range(slab_count):
            slab_off = data_offset + i * self.page_size
            slab = Slab(self.schema, self._mm, slab_off, self._rows_per_slab)
            bitmap_start = schema_offset + i * bitmap_size
            slab._live = bytearray(self._mm[bitmap_start : bitmap_start + bitmap_size])
            slab._count = sum(1 for b in slab._live if b)
            self._slabs.append(slab)
            self._total_rows += slab._count

    def _expand(self) -> None:
        self._mm.flush()
        old_size = len(self._mm)
        new_bitmap = self._rows_per_slab
        new_size = old_size + self.page_size + new_bitmap

        self._file.close()
        with open(self.path, "r+b") as f:
            f.truncate(new_size)

        self._file = open(self.path, "r+b")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)

        old_end = old_size
        new_bitmap_end = old_end + new_bitmap
        for i in range(old_end, new_bitmap_end):
            self._mm[i] = 0

        slab = Slab(self.schema, self._mm, new_bitmap_end, self._rows_per_slab)
        self._slabs.append(slab)

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
        """Context manager for timing operations."""
        class _Timer:
            def __init__(self, outer, name):
                self._outer = outer
                self._name = name
                self._start = 0.0
            def __enter__(self):
                self._start = time.time()
            def __exit__(self, *args):
                elapsed = time.time() - self._start
                self._outer._m_inc(f"db_{self._name}_total")
                self._outer._m_lat(f"db_{self._name}_latency", elapsed)
        return _Timer(self, metric_name)

    # ── CDC helper ───────────────────────────────────────────────────────────

    def _cdc_log(self, op: str, idx: int, row: Optional[Dict] = None,
                 old_row: Optional[Dict] = None) -> None:
        if self._cdc is not None:
            self._cdc.append(op, idx, row, old_row)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def insert(self, row: Dict[str, Any]) -> int:
        with self._m_time("insert"):
            if self.is_columnar():
                idx = self._table.insert(row)
                self._cdc_log("insert", idx, row)
                return idx
            for slab_idx, slab in enumerate(self._slabs):
                if not slab.is_full:
                    local_idx = slab.insert(row)
                    global_idx = slab_idx * self._rows_per_slab + local_idx
                    self._total_rows += 1
                    self._cdc_log("insert", global_idx, row)
                    return global_idx
            self._expand()
            idx = self._slabs[-1].insert(row)
            global_idx = (len(self._slabs) - 1) * self._rows_per_slab + idx
            self._total_rows += 1
            self._cdc_log("insert", global_idx, row)
            return global_idx

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        with self._m_time("get"):
            if self.is_columnar():
                return self._table.get(idx)
            slab_idx = idx // self._rows_per_slab
            local_idx = idx % self._rows_per_slab
            if slab_idx >= len(self._slabs):
                return None
            return self._slabs[slab_idx].get(local_idx)

    def get_raw(self, idx: int) -> Optional[memoryview]:
        with self._m_time("get_raw"):
            if self.is_columnar():
                return None
            slab_idx = idx // self._rows_per_slab
            local_idx = idx % self._rows_per_slab
            if slab_idx >= len(self._slabs):
                return None
            return self._slabs[slab_idx].get_raw(local_idx)

    def update(self, idx: int, row: Dict[str, Any]) -> None:
        with self._m_time("update"):
            if self.is_columnar():
                old = self._table.get(idx)
                self._table.update(idx, row)
                self._cdc_log("update", idx, row, old)
                return
            slab_idx = idx // self._rows_per_slab
            local_idx = idx % self._rows_per_slab
            if slab_idx >= len(self._slabs):
                raise KeyError(f"Row {idx} not found")
            old_row = self._slabs[slab_idx].get(local_idx)
            self._slabs[slab_idx].update(local_idx, row)
            self._cdc_log("update", idx, row, old_row)

    def delete(self, idx: int) -> None:
        with self._m_time("delete"):
            if self.is_columnar():
                old = self._table.get(idx)
                self._table.delete(idx)
                self._cdc_log("delete", idx, None, old)
                return
            slab_idx = idx // self._rows_per_slab
            local_idx = idx % self._rows_per_slab
            if slab_idx >= len(self._slabs):
                raise KeyError(f"Row {idx} not found")
            old_row = self._slabs[slab_idx].get(local_idx)
            self._slabs[slab_idx].delete(local_idx)
            self._total_rows -= 1
            self._cdc_log("delete", idx, None, old_row)

    def query(self, predicate: Callable[[Dict[str, Any]], bool]) -> Iterator[Tuple[int, Dict[str, Any]]]:
        with self._m_time("query"):
            for idx, row in self:
                if predicate(row):
                    yield idx, row

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

    def close(self) -> None:
        import gc
        gc.collect()
        if self._mm:
            self._mm.flush()
            try:
                self._mm.close()
            except BufferError:
                pass
            self._mm = None
        if self._file:
            self._file.close()
            self._file = None
        if self._wal is not None:
            self._wal.close()
        if self._cdc is not None:
            self._cdc.close()

    # ── Columnar-specific methods ─────────────────────────────────────────────

    def select(self, where=None, columns=None, limit=None, offset=0):
        """Select with filter/projection/limit/offset. Columnar only."""
        if not self.is_columnar():
            raise RuntimeError("Select requires columnar storage")
        return self._table.select(where=where, columns=columns, limit=limit, offset=offset)

    def aggregate(self, column_name: str, agg: str = "sum", where=None):
        """Aggregate on a column. Columnar only."""
        if not self.is_columnar():
            raise RuntimeError("Aggregate requires columnar storage")
        return self._table.aggregate(column_name, agg, where)

    def select_column(self, column_name: str) -> List[Any]:
        """Fast extraction of a single column. Columnar only."""
        if not self.is_columnar():
            raise RuntimeError("select_column requires columnar storage")
        return self._table.select_column(column_name)

    def memory_usage(self) -> int:
        """Memory usage in bytes (columnar only)."""
        if not self.is_columnar():
            raise RuntimeError("memory_usage requires columnar storage")
        return self._table.memory_usage()

    # ── Row-only methods (indexing, transactions) ────────────────────────────

    def create_index(self, column: str) -> None:
        if self.is_columnar():
            raise RuntimeError("Indexing not supported in columnar storage")
        if self._indexes is None:
            raise RuntimeError("Indexing not available (index.py missing)")
        if column in self._indexes:
            return
        self._indexes.create(column)
        for idx, row in self:
            self._indexes._indexes[column].insert(idx, row)

    def drop_index(self, column: str) -> None:
        if self.is_columnar():
            raise RuntimeError("Indexing not supported in columnar storage")
        if self._indexes is not None and column in self._indexes:
            self._indexes.drop(column)

    def find(self, **kwargs) -> List[Dict[str, Any]]:
        if self.is_columnar():
            raise RuntimeError("Find by index not supported in columnar storage")
        if self._indexes is None:
            raise RuntimeError("Indexing not available (index.py missing)")
        row_indices = self._indexes.lookup(**kwargs)
        if row_indices is None:
            return []
        return [self.get(i) for i in row_indices]

    @contextmanager
    def transaction(self):
        if self.is_columnar():
            raise RuntimeError("Transactions not supported in columnar storage")
        if self._wal is None:
            raise RuntimeError("WAL not available (wal.py missing)")
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

    def _tx_rollback(self) -> None:
        if not self._in_tx:
            return
        for op, idx, data in reversed(self._tx_state):
            if op == "insert":
                slab_idx = idx // self._rows_per_slab
                local_idx = idx % self._rows_per_slab
                if slab_idx < len(self._slabs):
                    self._slabs[slab_idx].delete(local_idx)
                    self._total_rows -= 1
            elif op == "update" and data is not None:
                slab_idx = idx // self._rows_per_slab
                local_idx = idx % self._rows_per_slab
                if slab_idx < len(self._slabs):
                    self._slabs[slab_idx].update(local_idx, data)
            elif op == "delete" and data is not None:
                slab_idx = idx // self._rows_per_slab
                local_idx = idx % self._rows_per_slab
                if slab_idx < len(self._slabs) and local_idx < self._slabs[slab_idx]._capacity:
                    raw = self.schema.encode_row(data)
                    start = self._slabs[slab_idx]._row_offset(local_idx)
                    mv = memoryview(self._mm)[start : start + len(raw)]
                    mv[:] = raw
                    self._slabs[slab_idx]._live[local_idx] = 1
                    self._total_rows += 1
        self._in_tx = False
        self._tx_state = []
        if self._wal is not None:
            self._wal._buffer.clear()

    def checkpoint(self) -> None:
        if self.is_columnar():
            raise RuntimeError("Checkpoint not supported in columnar storage")
        if self._wal is not None:
            self._wal.clear()

    def __repr__(self) -> str:
        if self.is_columnar():
            return f"SnapDB({self.path!r}, storage=columnar, rows={len(self._table)})"
        indexes = list(self._indexes._indexes.keys()) if self._indexes else []
        return f"SnapDB({self.path!r}, rows={self._total_rows}, slabs={len(self._slabs)}, indexes={indexes})"
