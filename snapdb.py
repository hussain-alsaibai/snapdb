"""
SnapDB — Extremely Lightweight Lightning-Fast In-Memory Database

A single-file, zero-dependency, pure-Python in-memory database using
memory-mapped files, memoryview zero-copy reads, and slab-oriented storage.

Key Innovations:
- Slab-oriented: Each segment (slab) holds all columns for N rows contiguously
- Zero-copy reads: memoryview slices into mmap — no deserialization
- Single-file: Schema, bitmap, and data all in one .snap file
- Fixed-width types only: int8/16/32/64, float32/64, bool, fixed bytes
- Pure Python, no dependencies (stdlib only: mmap, struct, os, json)

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
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union
from contextlib import contextmanager

# v0.2.0 imports (optional — gracefully degrade if not present)
try:
    from index import HashIndex, MultiIndex
except ImportError:
    HashIndex = None  # type: ignore
    MultiIndex = None  # type: ignore
try:
    from wal import WAL
except ImportError:
    WAL = None  # type: ignore


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
    """Return byte size for a SnapDB type."""
    if dtype.startswith("bytes"):
        return int(dtype.split(":")[1])
    return _TYPE_SIZES[dtype]


def _struct_code(dtype: str) -> str:
    """Return struct format code for a SnapDB type."""
    if dtype.startswith("bytes"):
        return f"{_type_size(dtype)}s"
    return _TYPE_CODES[dtype]


# ── Header ─────────────────────────────────────────────────────────────────────

_HEADER_FMT = "<4sHHIQ"  # magic(4) version(2) page_size(2) flags(4) schema_offset(8)
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
        """Total bytes per row."""
        return self._row_width

    def offset(self, name: str) -> int:
        """Byte offset of column within a row."""
        return self._offsets[self._name_to_idx[name]]

    def index(self, name: str) -> int:
        """Column index by name."""
        return self._name_to_idx[name]

    def to_json(self) -> List[Dict[str, str]]:
        return [{"name": c.name, "dtype": c.dtype} for c in self.columns]

    @classmethod
    def from_json(cls, data: List[Dict[str, str]]) -> "Schema":
        return cls([ColumnDef(c["name"], c["dtype"]) for c in data])

    def encode_row(self, row: Dict[str, Any]) -> bytes:
        """Pack a dict into fixed-width binary row."""
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
        """Unpack binary row into dict."""
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
    """
    A memory-mapped slab holding rows in a contiguous buffer.
    
    Zero-copy reads: memoryview slices into the mmap buffer.
    In-place writes: struct.pack_into overwrites existing bytes.
    """

    def __init__(self, schema: Schema, mm: mmap.mmap, offset: int, capacity: int) -> None:
        self.schema = schema
        self._mm = mm
        self._offset = offset
        self._capacity = capacity
        self._count = 0
        # Bitmap of live rows within this slab
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
        """Byte offset of row idx within the mmap."""
        return self._offset + idx * self.schema.row_width

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        """Zero-copy read of row idx (returns decoded dict)."""
        if idx >= self._capacity or not self._live[idx]:
            return None
        start = self._row_offset(idx)
        end = start + self.schema.row_width
        # memoryview slice — zero copy
        view = memoryview(self._mm)[start:end]
        return self.schema.decode_row(view)

    def get_raw(self, idx: int) -> Optional[memoryview]:
        """Zero-copy read returning memoryview (fastest)."""
        if idx >= self._capacity or not self._live[idx]:
            return None
        start = self._row_offset(idx)
        end = start + self.schema.row_width
        return memoryview(self._mm)[start:end]

    def insert(self, row: Dict[str, Any]) -> int:
        """Insert a row, return its index. Raises if full."""
        if self.is_full:
            raise RuntimeError("Slab is full")
        idx = self._count
        self._count += 1
        self._live[idx] = 1
        raw = self.schema.encode_row(row)
        start = self._row_offset(idx)
        # In-place write via memoryview
        mv = memoryview(self._mm)[start : start + len(raw)]
        mv[:] = raw
        return idx

    def update(self, idx: int, row: Dict[str, Any]) -> None:
        """In-place update of row idx."""
        if idx >= self._capacity or not self._live[idx]:
            raise KeyError(f"Row {idx} not found")
        raw = self.schema.encode_row(row)
        start = self._row_offset(idx)
        mv = memoryview(self._mm)[start : start + len(raw)]
        mv[:] = raw

    def delete(self, idx: int) -> None:
        """Mark row as deleted (lazy)."""
        if idx < self._capacity:
            self._live[idx] = 0

    def iter_rows(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Iterate live rows: (idx, row_dict)."""
        for i in range(self._count):
            if self._live[i]:
                yield i, self.get(i)

    def iter_raw(self) -> Iterator[Tuple[int, memoryview]]:
        """Iterate live rows: (idx, memoryview) — fastest."""
        for i in range(self._count):
            if self._live[i]:
                yield i, self.get_raw(i)


# ── SnapDB Engine ──────────────────────────────────────────────────────────────

class SnapDB:
    """
    Single-file in-memory database with zero-copy reads.
    
    Usage:
        db = SnapDB("data.snap", schema)
        db.insert({"id": 1, "name": b"alice", "score": 95.5})
        row = db.get(0)           # dict (decoded)
        raw = db.get_raw(0)       # memoryview (zero-copy)
    """

    def __init__(self, path: Union[str, Path], schema: Schema, page_size: int = _DEFAULT_PAGE_SIZE) -> None:
        self.path = Path(path)
        self.schema = schema
        self.page_size = page_size
        self._slabs: List[Slab] = []
        self._rows_per_slab = page_size // schema.row_width
        self._total_rows = 0
        self._mm: Optional[mmap.mmap] = None
        self._file: Optional[BinaryIO] = None

        # v0.2.0: transaction state tracking
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
        """Create a new database file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        schema_json = json.dumps(self.schema.to_json()).encode("utf-8")
        schema_offset = _HEADER_SIZE + len(schema_json)
        bitmap_offset = schema_offset
        # Reserve space for bitmap (one byte per row in first slab)
        bitmap_size = self._rows_per_slab
        data_offset = bitmap_offset + bitmap_size

        # Calculate total size: header + schema + bitmap + one slab
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
        """Load existing database file."""
        file_size = os.path.getsize(self.path)
        if file_size < _HEADER_SIZE:
            raise ValueError(f"File too small ({file_size} bytes) — not a valid SnapDB")

        self._file = open(self.path, "r+b")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)

        version, page_size, flags, schema_offset = _unpack_header(self._mm[:_HEADER_SIZE])
        self.page_size = page_size

        # Read schema
        schema_json = json.loads(self._mm[_HEADER_SIZE:schema_offset].decode("utf-8"))
        self.schema = Schema.from_json(schema_json)
        self._rows_per_slab = page_size // self.schema.row_width

        # Calculate slab layout
        bitmap_size = self._rows_per_slab
        data_offset = schema_offset + bitmap_size
        file_size = len(self._mm)
        slab_count = (file_size - data_offset) // self.page_size

        for i in range(slab_count):
            slab_off = data_offset + i * self.page_size
            slab = Slab(self.schema, self._mm, slab_off, self._rows_per_slab)
            # Count live rows
            bitmap_start = schema_offset + i * bitmap_size
            slab._live = bytearray(self._mm[bitmap_start : bitmap_start + bitmap_size])
            slab._count = sum(1 for b in slab._live if b)
            self._slabs.append(slab)
            self._total_rows += slab._count

    def _expand(self) -> None:
        """Add a new slab to the file."""
        # Flush current mmap
        self._mm.flush()
        
        old_size = len(self._mm)
        # New slab data + bitmap
        new_data = self.page_size
        new_bitmap = self._rows_per_slab
        new_size = old_size + new_data + new_bitmap
        
        # Use ftruncate to resize the file, then re-mmap
        self._file.close()
        with open(self.path, "r+b") as f:
            f.truncate(new_size)
        
        self._file = open(self.path, "r+b")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)
        
        # Clear new bitmap area
        old_end = old_size
        new_bitmap_end = old_end + new_bitmap
        for i in range(old_end, new_bitmap_end):
            self._mm[i] = 0
        
        # Create slab object pointing to data area (after bitmap)
        slab_idx = len(self._slabs)
        slab_offset = new_bitmap_end
        slab = Slab(self.schema, self._mm, slab_offset, self._rows_per_slab)
        self._slabs.append(slab)

    def insert(self, row: Dict[str, Any]) -> int:
        """Insert a row. Returns global row index."""
        # Find slab with space
        for slab_idx, slab in enumerate(self._slabs):
            if not slab.is_full:
                local_idx = slab.insert(row)
                global_idx = slab_idx * self._rows_per_slab + local_idx
                self._total_rows += 1
                self._on_insert(row, global_idx)
                return global_idx
        # All slabs full — expand
        self._expand()
        idx = self._slabs[-1].insert(row)
        global_idx = (len(self._slabs) - 1) * self._rows_per_slab + idx
        self._total_rows += 1
        self._on_insert(row, global_idx)
        return global_idx

    def _on_insert(self, row: Dict[str, Any], global_idx: int) -> None:
        """v0.2.0: update indexes, WAL, and transaction state."""
        if self._indexes is not None:
            for idx_obj in self._indexes._indexes.values():
                idx_obj.insert(global_idx, row)
        if self._wal is not None:
            self._wal.append("insert", row=row)
        if self._in_tx:
            self._tx_state.append(("insert", global_idx, None))

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get row as decoded dict."""
        slab_idx = idx // self._rows_per_slab
        local_idx = idx % self._rows_per_slab
        if slab_idx >= len(self._slabs):
            return None
        return self._slabs[slab_idx].get(local_idx)

    def get_raw(self, idx: int) -> Optional[memoryview]:
        """Get row as zero-copy memoryview (fastest)."""
        slab_idx = idx // self._rows_per_slab
        local_idx = idx % self._rows_per_slab
        if slab_idx >= len(self._slabs):
            return None
        return self._slabs[slab_idx].get_raw(local_idx)

    def update(self, idx: int, row: Dict[str, Any]) -> None:
        """Update row in-place."""
        slab_idx = idx // self._rows_per_slab
        local_idx = idx % self._rows_per_slab
        if slab_idx >= len(self._slabs):
            raise KeyError(f"Row {idx} not found")

        # v0.2.0: update indexes + WAL + tx tracking
        old_row = self._slabs[slab_idx].get(local_idx)
        self._slabs[slab_idx].update(local_idx, row)
        if self._indexes is not None and old_row is not None:
            for idx_obj in self._indexes._indexes.values():
                idx_obj.update(idx, old_row, row)
        if self._wal is not None:
            self._wal.append("update", idx=idx, row=row)
        if self._in_tx and old_row is not None:
            self._tx_state.append(("update", idx, old_row))
        if self._in_tx:
            self._tx_rows.add(idx)

    def delete(self, idx: int) -> None:
        """Delete row (lazy)."""
        slab_idx = idx // self._rows_per_slab
        local_idx = idx % self._rows_per_slab
        if slab_idx >= len(self._slabs):
            raise KeyError(f"Row {idx} not found")

        # v0.2.0: update indexes + WAL + tx tracking
        old_row = self._slabs[slab_idx].get(local_idx)
        self._slabs[slab_idx].delete(local_idx)
        self._total_rows -= 1
        if self._indexes is not None and old_row is not None:
            for idx_obj in self._indexes._indexes.values():
                idx_obj.delete(idx, old_row)
        if self._wal is not None:
            self._wal.append("delete", idx=idx)
        if self._in_tx and old_row is not None:
            self._tx_state.append(("delete", idx, old_row))
        if self._in_tx:
            self._tx_rows.add(idx)

    def __iter__(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Iterate all live rows: (global_idx, row_dict)."""
        for slab_idx, slab in enumerate(self._slabs):
            base = slab_idx * self._rows_per_slab
            for local_idx, row in slab.iter_rows():
                yield base + local_idx, row

    def iter_raw(self) -> Iterator[Tuple[int, memoryview]]:
        """Iterate all live rows: (global_idx, memoryview)."""
        for slab_idx, slab in enumerate(self._slabs):
            base = slab_idx * self._rows_per_slab
            for local_idx, raw in slab.iter_raw():
                yield base + local_idx, raw

    def query(self, predicate: Callable[[Dict[str, Any]], bool]) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Filter rows by predicate."""
        for idx, row in self:
            if predicate(row):
                yield idx, row

    def query_raw(self, predicate: Callable[[memoryview], bool]) -> Iterator[Tuple[int, memoryview]]:
        """Filter rows by raw memoryview predicate (fastest)."""
        for idx, raw in self.iter_raw():
            if predicate(raw):
                yield idx, raw

    def __len__(self) -> int:
        return self._total_rows

    def close(self) -> None:
        """Close the database."""
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

    def _tx_begin(self) -> None:
        """Start tracking transaction changes."""
        self._in_tx = True
        self._tx_state = []

    def _tx_end(self) -> None:
        """End transaction tracking."""
        self._in_tx = False
        self._tx_state = []

    def _tx_rollback(self) -> None:
        """Rollback current transaction by reversing all tracked operations."""
        if not self._in_tx:
            return
        # Reverse operations in reverse order (LIFO)
        for op, idx, data in reversed(self._tx_state):
            if op == "insert":
                # Undo insert: delete the row
                self._raw_delete(idx)
            elif op == "update":
                # Undo update: restore old data
                self._raw_update(idx, data)
            elif op == "delete":
                # Undo delete: re-insert old data
                self._raw_insert(idx, data)
        self._tx_end()
        if self._wal is not None:
            self._wal._buffer.clear()

    def _raw_delete(self, idx: int) -> None:
        """Internal: delete without tracking."""
        slab_idx = idx // self._rows_per_slab
        local_idx = idx % self._rows_per_slab
        if slab_idx < len(self._slabs):
            self._slabs[slab_idx].delete(local_idx)
            self._total_rows -= 1

    def _raw_update(self, idx: int, row: Dict[str, Any]) -> None:
        """Internal: update without tracking."""
        slab_idx = idx // self._rows_per_slab
        local_idx = idx % self._rows_per_slab
        if slab_idx < len(self._slabs):
            self._slabs[slab_idx].update(local_idx, row)

    def _raw_insert(self, idx: int, row: Dict[str, Any]) -> None:
        """Internal: re-insert without tracking (at specific idx, not append)."""
        slab_idx = idx // self._rows_per_slab
        local_idx = idx % self._rows_per_slab
        if slab_idx < len(self._slabs) and local_idx < self._slabs[slab_idx]._capacity:
            raw = self.schema.encode_row(row)
            start = self._slabs[slab_idx]._row_offset(local_idx)
            mv = memoryview(self._mm)[start : start + len(raw)]
            mv[:] = raw
            self._slabs[slab_idx]._live[local_idx] = 1
            self._total_rows += 1

    def create_index(self, column: str) -> None:
        """Create a hash index on a column."""
        if self._indexes is None:
            raise RuntimeError("Indexing not available (index.py missing)")
        if column in self._indexes:
            return  # Already exists
        self._indexes.create(column)
        # Index existing rows
        for idx, row in self:
            self._indexes._indexes[column].insert(idx, row)

    def drop_index(self, column: str) -> None:
        """Drop a hash index."""
        if self._indexes is not None and column in self._indexes:
            self._indexes.drop(column)

    def find(self, **kwargs) -> List[Dict[str, Any]]:
        """Find rows by indexed column(s). Returns list of matching rows."""
        if self._indexes is None:
            raise RuntimeError("Indexing not available")
        row_indices = self._indexes.lookup(**kwargs)
        if row_indices is None:
            return []
        return [self.get(i) for i in row_indices]

    # ── v0.2.0: Transactions ──────────────────────────────────────────────────

    @contextmanager
    def transaction(self):
        """Transaction context manager.

        Usage:
            with db.transaction():
                db.insert({...})
                db.update(0, {...})
                # auto-commit on success, rollback on exception
        """
        if self._wal is None:
            raise RuntimeError("WAL not available (wal.py missing)")
        self._wal.begin()
        self._tx_begin()
        try:
            yield self
            self._wal.commit()
            self._tx_end()
        except Exception:
            self._wal.rollback()
            self._tx_rollback()
            raise

    def checkpoint(self) -> None:
        """Checkpoint WAL — clear after successful operations."""
        if self._wal is not None:
            self._wal.clear()

    # ── v0.2.0: Query Engine ──────────────────────────────────────────────────

    def select(self, **conditions) -> "Query":
        """Start a query (requires query.py)."""
        try:
            from query import query as _query
            return _query(self).filter(**conditions)
        except ImportError:
            raise RuntimeError("Query engine not available (query.py missing)")

    def __repr__(self) -> str:
        indexes = list(self._indexes._indexes.keys()) if self._indexes else []
        return f"SnapDB({self.path!r}, rows={self._total_rows}, slabs={len(self._slabs)}, indexes={indexes})"
