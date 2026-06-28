"""
SnapDB WAL — Write-Ahead Log for durability and transactions

Format: one JSON line per record (append-only)

Types:
    {"op": "insert", "row": {...}}
    {"op": "update", "idx": 0, "row": {...}}
    {"op": "delete", "idx": 0}
    {"op": "begin", "txid": 1}
    {"op": "commit", "txid": 1}
    {"op": "rollback", "txid": 1}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class WAL:
    """Write-ahead log for SnapDB.

    Usage:
        wal = WAL("data.wal")
        wal.append("insert", row={"id": 1, ...})
        wal.append("update", idx=0, row={"id": 1, ...})
        wal.checkpoint()  # replay and clear
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._file: Optional[Any] = None
        self._buffer: List[Dict] = []
        self._buffer_size = 100
        self._txid = 0
        self._in_tx = False

    def _open(self) -> None:
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a")

    def append(self, op: str, **kwargs) -> None:
        """Append a log record."""
        def _encode(val):
            if isinstance(val, bytes):
                return {"__bytes__": val.hex()}
            if isinstance(val, dict):
                return {k: _encode(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_encode(v) for v in val]
            return val

        record = _encode({"op": op, **kwargs})
        self._buffer.append(record)
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._open()
        for record in self._buffer:
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._buffer.clear()

    def begin(self) -> int:
        """Start a transaction. Returns txid."""
        self._txid += 1
        self._in_tx = True
        self.append("begin", txid=self._txid)
        return self._txid

    def commit(self) -> None:
        """Commit current transaction."""
        if self._in_tx:
            self.append("commit", txid=self._txid)
            self._flush()
            self._in_tx = False

    def rollback(self) -> None:
        """Rollback current transaction."""
        if self._in_tx:
            self.append("rollback", txid=self._txid)
            self._flush()
            self._in_tx = False

    def replay(self) -> Iterator[Dict]:
        """Iterate all log records (for recovery)."""
        self._flush()
        if not self.path.exists():
            return
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def clear(self) -> None:
        """Clear the WAL after successful checkpoint."""
        self._flush()
        if self._file:
            self._file.close()
            self._file = None
        if self.path.exists():
            self.path.unlink()

    def close(self) -> None:
        self._flush()
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
