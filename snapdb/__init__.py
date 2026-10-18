"""
SnapDB — Extremely Lightweight Lightning-Fast In-Memory Database for Python

A single-file, zero-dependency, pure-Python in-memory database using
memory-mapped files, memoryview zero-copy reads, and slab-oriented storage.

Quick start:
    >>> from snapdb import SnapDB, Schema, ColumnDef
    >>> schema = Schema([ColumnDef("id", "i32"), ColumnDef("name", "bytes:20")])
    >>> db = SnapDB("data.snap", schema)
    >>> db.insert({"id": 1, "name": "alice"})
    0
    >>> db.get(0)
    {"id": 1, "name": "alice"}

Index-aware queries (v0.18.0+):
    >>> from snapdb import query
    >>> db.create_range_index("age")
    >>> [r for _, r in query(db).between("age", 25, 65).execute()]
    [{"id": 25, "age": 25, ...}, ...]
    >>> query(db).between("age", 25, 65).explain()
    "RangeIndex[age [25, 65)]"

Vector search (v0.17.0+):
    >>> from snapdb import VectorIndex
    >>> idx = VectorIndex(metric="cosine")
    >>> idx.add("alice", [0.1, 0.2, 0.3])
    >>> idx.add("bob", [0.4, 0.5, 0.6])
    >>> idx.search([0.1, 0.21, 0.29], k=1)
    [('alice', 0.9998...)]
"""

__version__ = "0.18.0"
__author__ = "OpenClaw (hussain-alsaibai)"
__license__ = "MIT"

# Core re-exports — these are the public API
from .core import SnapDB, Schema, ColumnDef, CDCLog
from .columnar import ColumnarTable, Column
from .metrics import Metrics
from .profiler import QueryProfiler
from .vector import (
    VectorIndex,
    cosine_similarity,
    euclidean_distance,
    dot_product,
)

__all__ = [
    "SnapDB",
    "Schema",
    "ColumnDef",
    "ColumnarTable",
    "Column",
    "Metrics",
    "CDCLog",
    "QueryProfiler",
    "VectorIndex",
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
    "__version__",
]
