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
from typing import Any, Dict, Iterator, List, Optional, Union

from ._util import _derive_key, _xor_stream


class WAL:
    """Write-ahead log for SnapDB.

    Usage:
        wal = WAL("data.wal")
        wal.append("insert", row={"id": 1, ...})
        wal.append("update", idx=0, row={"id": 1, ...})
        wal.checkpoint()  # replay and clear
    """

    def __init__(self, path: str, encryption_key: Union[str, bytes, None] = None) -> None:
        self.path = Path(path)
        self._file: Optional[Any] = None
        self._buffer: List[Dict] = []
        self._buffer_size = 100
        self._txid = 0
        self._in_tx = False
        self._key = _derive_key(encryption_key)

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
            payload = json.dumps(record, separators=(",", ":")).encode("utf-8")
            if self._key is not None:
                nonce = os.urandom(16)
                payload = json.dumps({
                    "enc": True,
                    "nonce": nonce.hex(),
                    "data": _xor_stream(self._key, nonce, payload).hex(),
                }, separators=(",", ":")).encode("utf-8")
            self._file.write(payload.decode("utf-8") + "\n")
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

        def _decode(val):
            if isinstance(val, dict):
                if set(val) == {"__bytes__"}:
                    return bytes.fromhex(val["__bytes__"])
                return {k: _decode(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_decode(v) for v in val]
            return val

        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Torn/partial trailing record from a crash mid-append.
                    # Everything before it is intact and already yielded;
                    # stop here instead of making the database unopenable.
                    break
                if isinstance(record, dict) and record.get("enc"):
                    if self._key is None:
                        raise ValueError("WAL is encrypted; pass encryption_key to replay it")
                    try:
                        nonce = bytes.fromhex(record["nonce"])
                        data = bytes.fromhex(record["data"])
                        record = json.loads(_xor_stream(self._key, nonce, data).decode("utf-8"))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Torn encrypted record (or truncated hex) — same as above.
                        break
                yield _decode(record)

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
