#!/usr/bin/env python3
"""
Schema with precompiled struct format for fast encode/decode.
Drop-in replacement for the Schema class in snapdb.py.

v0.3.2-OPT: Precompiled struct format eliminates per-column struct.pack/unpack calls.
"""

import struct
import time

def benchmark():
    from dataclasses import dataclass, field

    # Minimal Schema definition for testing
    _TYPE_CODES = {
        "i8": "b", "i16": "h", "i32": "i", "i64": "q",
        "u8": "B", "u16": "H", "u32": "I", "u64": "Q",
        "f32": "f", "f64": "d",
        "bool": "?",
    }

    _TYPE_SIZES = {
        "i8": 1, "i16": 2, "i32": 4, "i64": 8,
        "u8": 1, "u16": 2, "u32": 4, "u64": 8,
        "f32": 4, "f64": 8,
        "bool": 1,
    }

    def _type_size(dtype):
        if dtype.startswith("bytes"):
            return int(dtype.split(":")[1])
        return _TYPE_SIZES[dtype]

    def _struct_code(dtype):
        if dtype.startswith("bytes"):
            return f"{_type_size(dtype)}s"
        return _TYPE_CODES[dtype]

    @dataclass(frozen=True)
    class ColumnDef:
        name: str
        dtype: str
        width: int = field(init=False)
        def __post_init__(self):
            object.__setattr__(self, "width", _type_size(self.dtype))

    class Schema:
        def __init__(self, columns):
            self.columns = tuple(columns)
            self._name_to_idx = {c.name: i for i, c in enumerate(columns)}
            self._row_width = sum(c.width for c in columns)
            self._offsets = [0]
            for c in columns[:-1]:
                self._offsets.append(self._offsets[-1] + c.width)

        @property
        def row_width(self):
            return self._row_width

        def offset(self, name):
            return self._offsets[self._name_to_idx[name]]

        def index(self, name):
            return self._name_to_idx[name]

        def encode_row(self, row):
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

        def decode_row(self, buf):
            row = {}
            for col in self.columns:
                off = self.offset(col.name)
                raw = bytes(buf[off:off + col.width])
                if col.dtype == "bool":
                    row[col.name] = struct.unpack("?", raw)[0]
                elif col.dtype.startswith("bytes"):
                    row[col.name] = raw.rstrip(b"\x00").decode("utf-8", errors="replace")
                else:
                    row[col.name] = struct.unpack(f"<{_struct_code(col.dtype)}", raw)[0]
            return row

    class SchemaFast:
        def __init__(self, columns):
            self.columns = tuple(columns)
            self._name_to_idx = {c.name: i for i, c in enumerate(columns)}
            self._row_width = sum(c.width for c in columns)
            self._offsets = [0]
            for c in columns[:-1]:
                self._offsets.append(self._offsets[-1] + c.width)
            self._compile_format()

        @property
        def row_width(self):
            return self._row_width

        def _compile_format(self):
            codes = []
            for col in self.columns:
                if col.dtype == "bool":
                    codes.append("?")
                elif col.dtype.startswith("bytes"):
                    codes.append(f"{col.width}s")
                else:
                    codes.append(_struct_code(col.dtype))
            self._struct_fmt = "<" + "".join(codes)
            self._struct_size = struct.calcsize(self._struct_fmt)

        def offset(self, name):
            return self._offsets[self._name_to_idx[name]]

        def index(self, name):
            return self._name_to_idx[name]

        def encode_row(self, row):
            values = []
            for col in self.columns:
                val = row.get(col.name, 0)
                if col.dtype == "bool":
                    values.append(bool(val))
                elif col.dtype.startswith("bytes"):
                    raw = val if isinstance(val, bytes) else str(val).encode("utf-8")
                    values.append(raw[:col.width].ljust(col.width, b"\x00"))
                else:
                    values.append(val)
            return struct.pack(self._struct_fmt, *values)

        def decode_row(self, buf):
            raw = bytes(buf) if isinstance(buf, memoryview) else buf
            unpacked = struct.unpack(self._struct_fmt, raw)
            row = {}
            for col, val in zip(self.columns, unpacked):
                if col.dtype == "bool":
                    row[col.name] = val
                elif col.dtype.startswith("bytes"):
                    row[col.name] = val.rstrip(b"\x00").decode("utf-8", errors="replace")
                else:
                    row[col.name] = val
            return row

    # Benchmark
    columns = [ColumnDef("id", "i32"), ColumnDef("name", "bytes:20"), ColumnDef("score", "f32"), ColumnDef("active", "bool")]
    schema_old = Schema(columns)
    schema_fast = SchemaFast(columns)

    row = {"id": 42, "name": "alice", "score": 95.5, "active": True}

    # Warmup
    for _ in range(100):
        schema_old.encode_row(row)
        schema_fast.encode_row(row)

    # Encode benchmark
    n = 500000
    t0 = time.perf_counter()
    for _ in range(n):
        schema_old.encode_row(row)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        schema_fast.encode_row(row)
    t_fast = time.perf_counter() - t0

    print(f"Encode: old={t_old:.3f}s fast={t_fast:.3f}s speedup={t_old/t_fast:.1f}x")

    # Decode benchmark
    raw = schema_fast.encode_row(row)
    buf = bytearray(raw)

    t0 = time.perf_counter()
    for _ in range(n):
        schema_old.decode_row(buf)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        schema_fast.decode_row(buf)
    t_fast = time.perf_counter() - t0

    print(f"Decode: old={t_old:.3f}s fast={t_fast:.3f}s speedup={t_old/t_fast:.1f}x")

    # Verify correctness
    encoded = schema_fast.encode_row(row)
    decoded = schema_fast.decode_row(encoded)
    assert decoded == {"id": 42, "name": "alice", "score": 95.5, "active": True}
    print("Correctness: PASS")

if __name__ == "__main__":
    benchmark()
