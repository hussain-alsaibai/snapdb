"""
Shared internal helpers for SnapDB modules.

Single source of truth for the dtype tables, query-value normalization, and
the XOR-stream cipher. core.py, columnar.py, index.py, and wal.py all import
from here so the copies can never drift apart.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Union

# ── Type mapping ───────────────────────────────────────────────────────────────

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


def _norm_value(value: Any) -> Any:
    """Normalize a query value to match how rows decode (bytes -> str), so
    scans, hash-index lookups, and columnar filters all compare identically."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


# ── XOR-stream cipher (SHA-256 keystream) ──────────────────────────────────────

def _derive_key(key: Union[str, bytes, None]) -> Optional[bytes]:
    if key is None:
        return None
    raw = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    return hashlib.sha256(raw).digest()


def _xor_stream(key: bytes, nonce: bytes, data: bytes) -> bytes:
    n = len(data)
    if n == 0:
        return data
    parts: List[bytes] = []
    pos = 0
    counter = 0
    while pos < n:
        block = hashlib.sha256(key + nonce + counter.to_bytes(8, "little")).digest()
        end = min(pos + 32, n)
        chunk = end - pos
        if chunk == 32:
            # Fast path: single 256-bit integer XOR avoids a 32-iteration Python loop.
            d_int = int.from_bytes(data[pos:end], "little")
            k_int = int.from_bytes(block, "little")
            parts.append((d_int ^ k_int).to_bytes(32, "little"))
        else:
            # Tail block (< 32 bytes).
            arr = bytearray(data[pos:end])
            for i in range(chunk):
                arr[i] ^= block[i]
            parts.append(bytes(arr))
        pos = end
        counter += 1
    return b"".join(parts)
