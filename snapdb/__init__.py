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
"""

__version__ = "0.4.0"
__author__ = "OpenClaw (hussain-alsaibai)"
__license__ = "MIT"

# Core re-exports — these are the public API
from .core import SnapDB, Schema, ColumnDef, CDCLog, _DEFAULT_PAGE_SIZE
from .columnar import ColumnarTable, Column
from .metrics import Metrics

__all__ = [
    "SnapDB",
    "Schema",
    "ColumnDef",
    "ColumnarTable",
    "Column",
    "Metrics",
    "CDCLog",
    "__version__",
]
