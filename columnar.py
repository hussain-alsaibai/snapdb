"""
Columnar storage engine for SnapDB.
Inspired by ClickHouse columnar layout for analytical workloads.
Zero-dependency, pure Python.
"""

from __future__ import annotations

import array
from typing import Any, Dict, List, Tuple, Callable, Iterator, Optional

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


def _type_size(dtype: str) -> int:
    if dtype.startswith("bytes"):
        return int(dtype.split(":")[1])
    return _TYPE_SIZES[dtype]


def _array_typecode(dtype: str) -> str:
    if dtype.startswith("bytes"):
        return "list"
    mapping = {
        "i8": "b", "i16": "h", "i32": "i", "i64": "q",
        "u8": "B", "u16": "H", "u32": "I", "u64": "Q",
        "f32": "f", "f64": "d",
        "bool": "B",
    }
    return mapping[dtype]


class Column:
    """A single column storing values of one type."""

    def __init__(self, name: str, dtype: str) -> None:
        self.name = name
        self.dtype = dtype
        self.width = _type_size(dtype)
        self._data: Any = None
        self._nullmask: Optional[array.array] = None
        self._init_storage()

    def _init_storage(self) -> None:
        if self.dtype.startswith("bytes"):
            self._data: List[bytes] = []
            self._nullmask = array.array("b")
        else:
            typecode = _array_typecode(self.dtype)
            self._data = array.array(typecode)
            self._nullmask = array.array("b")

    def append(self, value: Any) -> None:
        if value is None:
            self._nullmask.append(1)
            if self.dtype.startswith("bytes"):
                self._data.append(b"")
            else:
                self._data.append(0 if self._data.typecode not in ("f", "d") else 0.0)
        else:
            self._nullmask.append(0)
            if self.dtype.startswith("bytes"):
                if isinstance(value, str):
                    value = value.encode("utf-8")
                self._data.append(bytes(value))
            elif self.dtype == "bool":
                self._data.append(1 if value else 0)
            else:
                self._data.append(value)

    def __getitem__(self, idx: int) -> Any:
        if self._nullmask[idx]:
            return None
        if self.dtype.startswith("bytes"):
            return self._data[idx].decode("utf-8", errors="replace")
        if self.dtype == "bool":
            return bool(self._data[idx])
        return self._data[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        if idx < 0 or idx >= len(self._nullmask):
            raise IndexError(f"Index {idx} out of range")
        if value is None:
            self._nullmask[idx] = 1
            if self.dtype.startswith("bytes"):
                self._data[idx] = b""
            else:
                self._data[idx] = 0.0 if self._data.typecode in ("f", "d") else 0
        else:
            self._nullmask[idx] = 0
            if self.dtype.startswith("bytes"):
                if isinstance(value, str):
                    value = value.encode("utf-8")
                self._data[idx] = bytes(value)
            elif self.dtype == "bool":
                self._data[idx] = 1 if value else 0
            else:
                self._data[idx] = value

    def __len__(self) -> int:
        return len(self._nullmask)

    def iter_valid(self) -> Iterator[Tuple[int, Any]]:
        """Iterate (index, value) for non-null entries."""
        for i in range(len(self._nullmask)):
            if self._nullmask[i] == 0:
                yield i, self[i]

    def count_valid(self) -> int:
        return self._nullmask.count(0)

    def memory_usage(self) -> int:
        if self.dtype.startswith("bytes"):
            total = sum(len(d) for d in self._data)
            return total + len(self._nullmask) + len(self._data) * 8
        else:
            return len(self._data) * self._data.itemsize + len(self._nullmask)


class ColumnarTable:
    """
    Simple columnar in-memory table for analytical workloads.
    Supports insert, get, update, delete, select, aggregates.
    """

    def __init__(self, name: str, schema: List[Tuple[str, str]]) -> None:
        self.name = name
        self.columns: Dict[str, Column] = {}
        for col_name, col_type in schema:
            self.columns[col_name] = Column(col_name, col_type)
        self._row_count = 0

    def insert(self, row: Dict[str, Any]) -> int:
        for col_name, col in self.columns.items():
            col.append(row.get(col_name))
        idx = self._row_count
        self._row_count += 1
        return idx

    def __len__(self) -> int:
        return self._row_count

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx < 0 or idx >= self._row_count:
            return None
        row = {name: col[idx] for name, col in self.columns.items()}
        # If all values are None, treat as deleted
        if all(v is None for v in row.values()):
            return None
        return row

    def update(self, idx: int, row: Dict[str, Any]) -> None:
        if idx < 0 or idx >= self._row_count:
            raise IndexError(f"Row index {idx} out of range")
        current = self.get(idx)
        if current is None:
            raise KeyError(f"Row {idx} not found")
        merged = {**current, **row}
        for col_name, col in self.columns.items():
            col[idx] = merged.get(col_name)

    def delete(self, idx: int) -> None:
        if idx < 0 or idx >= self._row_count:
            raise IndexError(f"Row index {idx} out of range")
        for col in self.columns.values():
            col._nullmask[idx] = 1
            if col.dtype.startswith("bytes"):
                col._data[idx] = b""
            elif col._data.typecode in ("f", "d"):
                col._data[idx] = 0.0
            else:
                col._data[idx] = 0

    def select(self,
               where: Optional[Callable[[Dict[str, Any]], bool]] = None,
               columns: Optional[List[str]] = None,
               limit: Optional[int] = None,
               offset: int = 0) -> List[Dict[str, Any]]:
        if columns is None:
            columns = list(self.columns.keys())
        else:
            for c in columns:
                if c not in self.columns:
                    raise ValueError(f"Unknown column: {c}")
        result: List[Dict[str, Any]] = []
        matched = 0
        for idx in range(self._row_count):
            row = {name: self.columns[name][idx] for name in self.columns}
            if where is not None and not where(row):
                continue
            if matched < offset:
                matched += 1
                continue
            projected = {name: self.columns[name][idx] for name in columns}
            result.append(projected)
            matched += 1
            if limit is not None and len(result) >= limit:
                break
        return result

    def select_column(self, column_name: str) -> List[Any]:
        if column_name not in self.columns:
            raise ValueError(f"Unknown column: {column_name}")
        col = self.columns[column_name]
        return [col[i] for i in range(len(col))]

    def aggregate(self,
                  column_name: str,
                  agg: str = "sum",
                  where: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Any:
        if column_name not in self.columns:
            raise ValueError(f"Unknown column: {column_name}")
        col = self.columns[column_name]
        if agg == "count":
            return col.count_valid()
        if agg == "sum":
            return sum(v for _, v in col.iter_valid())
        if agg == "avg":
            total = 0
            count = 0
            for _, v in col.iter_valid():
                total += v
                count += 1
            return total / count if count else 0
        if agg == "min":
            min_val = None
            for _, v in col.iter_valid():
                if min_val is None or v < min_val:
                    min_val = v
            return min_val
        if agg == "max":
            max_val = None
            for _, v in col.iter_valid():
                if max_val is None or v > max_val:
                    max_val = v
            return max_val
        raise ValueError(f"Unsupported aggregation: {agg}")

    def memory_usage(self) -> int:
        return sum(col.memory_usage() for col in self.columns.values())

    def __repr__(self) -> str:
        return f"ColumnarTable({self.name!r}, rows={self._row_count}, cols={list(self.columns.keys())})"
