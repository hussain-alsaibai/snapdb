"""
Columnar storage engine for SnapDB.
A column-oriented storage layout for analytical workloads.
Zero-dependency, pure Python.

v0.3.1: Optimized batch operations, precomputed column lists, faster iteration.
v0.4.0: Dictionary encoding for low-cardinality string columns.
v0.5.0: Delta encoding for monotonic numeric columns.
v0.6.0: Vectorized filtering, auto-indexing, NumPy export.
v0.7.0: Frame-of-Reference + bit packing for bounded numeric ranges.
v0.8.0: Optional NumPy-accelerated aggregates over the zero-copy column buffer.
v0.9.0: NumPy-accelerated select_where masks + count_where.
"""

from __future__ import annotations

import array
import importlib.util
import operator
from typing import Any, Dict, List, Tuple, Callable, Iterator, Optional

# NumPy is an OPTIONAL accelerator (issue #14). We only check availability at
# import time (no actual import / no hard dependency); the array module path is
# always the zero-dependency default.
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None

# Comparison operators for vectorized predicates (issue #4)
_FILTER_OPS = {
    "eq": operator.eq, "==": operator.eq,
    "ne": operator.ne, "!=": operator.ne,
    "gt": operator.gt, ">": operator.gt,
    "gte": operator.ge, ">=": operator.ge,
    "lt": operator.lt, "<": operator.lt,
    "lte": operator.le, "<=": operator.le,
}

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

# Numeric types eligible for delta encoding
_DELTA_ELIGIBLE = {"i32", "i64", "u32", "u64"}

# Integer types eligible for Frame-of-Reference (FOR) encoding. Restricted to
# the wider types where 16-bit packing yields a real (>=50%) space win.
_FOR_ELIGIBLE = {"i32", "i64", "u32", "u64"}

# array.array typecode -> NumPy dtype string, for zero-copy export (PEP 688 / #7)
_NUMPY_DTYPE = {
    "b": "int8", "B": "uint8", "h": "int16", "H": "uint16",
    "i": "int32", "I": "uint32", "q": "int64", "Q": "uint64",
    "f": "float32", "d": "float64",
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


def _smallest_delta_typecode(max_delta: int, signed: bool = False) -> str:
    """Return the smallest array typecode that can hold max_delta."""
    if signed:
        if max_delta <= 127:
            return "b"   # i8
        elif max_delta <= 32767:
            return "h"   # i16
        elif max_delta <= 2147483647:
            return "i"   # i32
        else:
            return "q"   # i64
    else:
        if max_delta <= 255:
            return "B"   # u8
        elif max_delta <= 65535:
            return "H"   # u16
        elif max_delta <= 4294967295:
            return "I"   # u32
        else:
            return "Q"   # u64


class Column:
    """A single column storing values of one type.

    v0.4.0: Dictionary encoding for low-cardinality string columns.
    v0.5.0: Delta encoding for monotonic numeric columns.
    v0.6.0: O(1) delta reads via a lazily-built reconstruction cache;
            __slots__ for lower per-column memory overhead.
    """

    __slots__ = (
        "name", "dtype", "width",
        "_dict_encode", "_dict_threshold", "_dict_mode", "_dict_fallback",
        "_dict", "_dict_values", "_dict_codes",
        "_delta_encode", "_delta_samples", "_delta_mode", "_delta_fallback",
        "_delta_base", "_delta_prev", "_deltas", "_delta_typecode", "_delta_cache",
        "_for_encode", "_for_threshold", "_for_mode", "_for_fallback",
        "_for_stats", "_for_min", "_for_bits", "_for_mask", "_for_packed", "_for_count",
        "_data", "_nullmask",
    )

    def __init__(self, name: str, dtype: str,
                 dict_encode: bool = False, dict_threshold: int = 256,
                 delta_encode: bool = False, delta_samples: int = 50,
                 for_encode: bool = False, for_threshold: int = 50) -> None:
        self.name = name
        self.dtype = dtype
        self.width = _type_size(dtype)
        # Dictionary encoding config
        self._dict_encode = dict_encode
        self._dict_threshold = dict_threshold
        self._dict_mode: bool = False          # True when actively using dict
        self._dict_fallback: bool = False      # True when dict overflowed, using raw
        self._dict: Dict[bytes, int] = {}      # value -> code
        self._dict_values: List[bytes] = []     # code -> value (reverse)
        self._dict_codes: Optional[array.array] = None  # code per row
        # Delta encoding config
        self._delta_encode = delta_encode and dtype in _DELTA_ELIGIBLE
        self._delta_samples = delta_samples
        self._delta_mode: bool = False         # True when using delta encoding
        self._delta_fallback: bool = False   # True when delta disabled (non-monotonic)
        self._delta_base: int = 0              # base value
        self._delta_prev: int = 0              # previous value for delta computation
        self._deltas: Optional[array.array] = None  # delta per row
        self._delta_typecode: Optional[str] = None
        # Lazily-built full reconstruction of a delta column. Built on first
        # read after a write so random access / scans are O(1) / O(n) instead
        # of O(n) / O(n^2). Invalidated (set to None) whenever deltas change.
        self._delta_cache: Optional[array.array] = None
        # Frame-of-Reference encoding config
        self._for_encode = for_encode and dtype in _FOR_ELIGIBLE
        self._for_threshold = for_threshold
        self._for_mode: bool = False           # True when using FOR encoding
        self._for_fallback: bool = False       # True when FOR disabled (range too large)
        self._for_stats: Optional[Dict[str, Any]] = None  # min/max tracking during sampling
        self._for_min: int = 0
        self._for_bits: int = 0
        self._for_mask: int = 0
        self._for_packed: int = 0
        self._for_count: int = 0
        self._data: Any = None
        self._nullmask: Optional[array.array] = None
        self._init_storage()

    def _init_storage(self) -> None:
        if self.dtype == "bool":
            # Bit-packed boolean storage: Python int bitmask
            # _data is an int where bit i = value of row i (1=True, 0=False)
            self._data = 0
            self._nullmask = array.array("b")
        elif self.dtype.startswith("bytes"):
            if self._dict_encode:
                self._dict_mode = True
                self._dict: Dict[bytes, int] = {}
                self._dict_values: List[bytes] = []
                # Auto-select code type based on threshold
                if self._dict_threshold <= 256:
                    self._dict_codes = array.array("B")  # u8
                elif self._dict_threshold <= 65536:
                    self._dict_codes = array.array("H")  # u16
                else:
                    self._dict_codes = array.array("I")  # u32
                self._nullmask = array.array("b")
            else:
                self._data: List[bytes] = []
                self._nullmask = array.array("b")
        else:
            typecode = _array_typecode(self.dtype)
            self._data = array.array(typecode)
            self._nullmask = array.array("b")

    def _convert_to_raw(self) -> None:
        """Convert from dict mode back to raw storage (dict overflow)."""
        if not self._dict_mode or self._dict_fallback:
            return
        self._dict_fallback = True
        # Build raw list from codes + dict_values
        raw: List[bytes] = []
        for code in self._dict_codes:
            raw.append(self._dict_values[code])
        self._data = raw
        self._dict_mode = False
        self._dict = {}
        self._dict_values = []
        self._dict_codes = None

    def _reconstruct_delta(self) -> array.array:
        """Rebuild the full value array from base + deltas (O(n), one pass)."""
        out = array.array(_array_typecode(self.dtype))
        val = self._delta_base
        out.append(val)  # row 0 == base
        for d in self._deltas:
            val += d
            out.append(val)
        return out

    def _ensure_delta_cache(self) -> array.array:
        """Return a cached full reconstruction, building it once if needed."""
        cache = self._delta_cache
        if cache is None:
            cache = self._reconstruct_delta()
            self._delta_cache = cache
        return cache

    def _convert_delta_to_raw(self) -> None:
        """Convert from delta mode back to raw storage (non-monotonic detected)."""
        if not self._delta_mode or self._delta_fallback:
            return
        self._delta_fallback = True
        # Reconstruct full values from base + deltas
        self._data = self._reconstruct_delta()
        self._delta_mode = False
        self._deltas = None
        self._delta_cache = None
        self._delta_base = 0
        self._delta_prev = 0
        self._delta_typecode = None

    def _add_to_dict(self, value: bytes) -> int:
        """Add value to dict and return code."""
        code = self._dict.get(value)
        if code is not None:
            return code
        # New value
        code = len(self._dict_values)
        self._dict[value] = code
        self._dict_values.append(value)
        # Check threshold
        if code >= self._dict_threshold:
            self._convert_to_raw()
            # After fallback, return None to signal caller
            return None  # type: ignore
        return code

    def _start_delta_mode(self, value: int) -> None:
        """Switch to delta encoding mode."""
        self._delta_mode = True
        self._delta_base = value
        self._delta_prev = value
        # Determine smallest typecode for sample deltas
        old_data = list(self._data) if hasattr(self._data, '__iter__') else []
        max_delta = 0
        prev = value
        for v in old_data[1:]:
            delta = abs(v - prev)
            if delta > max_delta:
                max_delta = delta
            prev = v
        # Also consider future deltas - start with a safe default
        # For i64 timestamps, use i32; for u32, use u32; etc.
        if self.dtype in ("i64", "u64"):
            self._deltas = array.array("i")  # i32 for timestamps
            self._delta_typecode = "i"
        elif self.dtype in ("i32", "u32"):
            self._deltas = array.array("i")  # i32
            self._delta_typecode = "i"
        else:
            self._deltas = array.array("h")  # i16
            self._delta_typecode = "h"
        # Remove first value from raw data (it becomes base)
        self._data = array.array(_array_typecode(self.dtype))

    def _upgrade_delta_storage(self, needed_max: int) -> None:
        """Upgrade delta array to larger typecode if needed."""
        signed = self.dtype in ("i32", "i64")
        new_tc = _smallest_delta_typecode(needed_max, signed=signed)
        if new_tc == self._delta_typecode:
            return
        # Convert existing deltas
        new_arr = array.array(new_tc)
        for d in self._deltas:
            new_arr.append(d)
        self._deltas = new_arr
        self._delta_typecode = new_tc

    def _start_for_mode(self) -> None:
        """Switch to Frame-of-Reference bit-packing mode.

        Derive the base (min) and bit-width from the actual data being packed
        rather than from the sampling stats — the two can differ (e.g. a column
        that is both delta- and FOR-encoded falls back from delta and then only
        samples the *post-fallback* values, while ``_data`` still holds the
        earlier rows). Packing with a too-narrow width would silently corrupt
        those rows, so if the real range doesn't fit in 16 bits we stay raw.
        """
        old_data = list(self._data)
        if old_data:
            self._for_min = min(old_data)
            range_val = max(old_data) - self._for_min
        else:
            self._for_min = self._for_stats["min"] if self._for_stats else 0
            range_val = 0
        if range_val.bit_length() > 16:
            # No real space win — keep raw storage.
            self._for_fallback = True
            self._for_stats = None
            return
        self._for_mode = True
        if range_val <= 0:
            # All values identical - use 1 bit per value (0)
            self._for_bits = 1
            self._for_mask = 1
        else:
            # Number of bits needed per value
            bits = max(1, range_val.bit_length())
            self._for_bits = bits
            self._for_mask = (1 << bits) - 1
        self._for_packed = 0  # Python int bitmask for packed values
        self._for_count = 0
        # Convert existing raw data to FOR packed format
        self._data = array.array(_array_typecode(self.dtype))
        for v in old_data:
            self._append_for(v)

    def _append_for(self, value: int) -> None:
        """Append value using FOR bit-packing."""
        delta = value - self._for_min
        # Pack delta into the bitmask at the correct position
        bit_pos = self._for_count * self._for_bits
        self._for_packed |= (delta & self._for_mask) << bit_pos
        self._for_count += 1

    def _get_for_value(self, idx: int) -> int:
        """Reconstruct value from FOR encoding at index."""
        bit_pos = idx * self._for_bits
        delta = (self._for_packed >> bit_pos) & self._for_mask
        return self._for_min + delta

    def _widen_for(self, needed_delta: int) -> bool:
        """Repack the FOR column with enough bits to hold ``needed_delta``.

        Returns True if widening kept FOR worthwhile (<=16 bits, so still a
        space win for i32/i64); otherwise converts to raw and returns False.
        """
        new_bits = max(1, needed_delta.bit_length())
        if new_bits > 16:
            self._convert_for_to_raw()
            return False
        old_vals = [self._get_for_value(i) for i in range(self._for_count)]
        self._for_bits = new_bits
        self._for_mask = (1 << new_bits) - 1
        self._for_packed = 0
        self._for_count = 0
        for v in old_vals:
            self._append_for(v)
        return True

    def _convert_for_to_raw(self) -> None:
        """Convert from FOR mode back to raw storage."""
        if not self._for_mode or self._for_fallback:
            return
        self._for_fallback = True
        # Reconstruct full values from FOR packed data
        raw = array.array(_array_typecode(self.dtype))
        for i in range(self._for_count):
            raw.append(self._get_for_value(i))
        self._data = raw
        self._for_mode = False
        self._for_packed = 0
        self._for_count = 0

    def _append_delta(self, value: int) -> None:
        """Append value using delta encoding."""
        self._delta_cache = None  # invalidate reconstruction cache
        delta = value - self._delta_prev
        self._delta_prev = value

        # Check if delta fits in current typecode
        if self._delta_typecode == "B":
            if not (0 <= delta <= 255):
                self._upgrade_delta_storage(abs(delta))
        elif self._delta_typecode == "H":
            if not (0 <= delta <= 65535):
                self._upgrade_delta_storage(abs(delta))
        elif self._delta_typecode == "I":
            if not (0 <= delta <= 4294967295):
                self._upgrade_delta_storage(abs(delta))
        elif self._delta_typecode == "b":
            if not (-128 <= delta <= 127):
                self._upgrade_delta_storage(abs(delta))
        elif self._delta_typecode == "h":
            if not (-32768 <= delta <= 32767):
                self._upgrade_delta_storage(abs(delta))
        elif self._delta_typecode == "i":
            if not (-2147483648 <= delta <= 2147483647):
                self._upgrade_delta_storage(abs(delta))

        self._deltas.append(delta)

    def append(self, value: Any) -> None:
        if value is None:
            self._nullmask.append(1)
            if self.dtype.startswith("bytes"):
                if self._dict_mode and not self._dict_fallback:
                    self._dict_codes.append(0)  # code 0 for null placeholder
                else:
                    self._data.append(b"")
            elif self.dtype == "bool":
                # For bool bitmask, nulls just need nullmask entry
                pass
            elif self._delta_mode and not self._delta_fallback:
                self._deltas.append(0)
                self._delta_cache = None  # invalidate reconstruction cache
            elif self._for_mode and not self._for_fallback:
                # Pack a placeholder so _for_count stays aligned with the row
                # index; the value is hidden by the nullmask anyway.
                self._append_for(self._for_min)
            else:
                self._data.append(0 if self._data.typecode not in ("f", "d") else 0.0)
        else:
            self._nullmask.append(0)
            if self.dtype.startswith("bytes"):
                if isinstance(value, str):
                    value = value.encode("utf-8")
                bval = bytes(value)
                if self._dict_mode and not self._dict_fallback:
                    code = self._add_to_dict(bval)
                    if code is not None:
                        self._dict_codes.append(code)
                    else:
                        # Fallback occurred, append to raw
                        self._data.append(bval)
                else:
                    self._data.append(bval)
            elif self.dtype == "bool":
                bit_pos = len(self._nullmask) - 1
                if value:
                    self._data |= (1 << bit_pos)
            elif self._delta_encode and not self._delta_fallback:
                # Delta encoding path
                if not self._delta_mode:
                    # Still in sampling phase
                    self._data.append(value)
                    # Check if we should enable delta mode
                    if len(self._data) >= self._delta_samples:
                        # Analyze samples for monotonicity
                        monotonic = True
                        for i in range(1, len(self._data)):
                            if self._data[i] < self._data[i - 1]:
                                monotonic = False
                                break
                        if monotonic:
                            # Enable delta mode
                            base = self._data[0]
                            old_data = list(self._data)  # Save before reset
                            self._start_delta_mode(base)
                            # Convert existing samples to deltas
                            prev = base
                            for i in range(1, len(old_data)):
                                delta = old_data[i] - prev
                                self._deltas.append(delta)
                                prev = old_data[i]
                            self._delta_prev = prev
                            self._delta_cache = None  # invalidate reconstruction cache
                        else:
                            self._delta_fallback = True
                else:
                    # Already in delta mode
                    self._append_delta(value)
            elif self._for_encode and not self._for_fallback:
                # Frame-of-Reference encoding path
                if not self._for_mode:
                    # Still in sampling phase
                    self._data.append(value)
                    # Track min/max during sampling
                    if self._for_stats is None:
                        self._for_stats = {"min": value, "max": value, "samples": []}
                    self._for_stats["samples"].append(value)
                    if value < self._for_stats["min"]:
                        self._for_stats["min"] = value
                    if value > self._for_stats["max"]:
                        self._for_stats["max"] = value
                    # Check if we should enable FOR mode
                    if len(self._for_stats["samples"]) >= self._for_threshold:
                        range_val = self._for_stats["max"] - self._for_stats["min"]
                        # Only enable if we save at least 50% space
                        # i32 = 32 bits, so FOR must use <= 16 bits per value
                        if range_val.bit_length() <= 16:
                            self._start_for_mode()
                        else:
                            self._for_fallback = True
                            self._for_stats = None
                else:
                    # Already in FOR mode. A value outside the packable range
                    # [min, min + mask] would be silently truncated by the mask.
                    # Above the range: widen the bit-width (still compressed) and
                    # fall back to raw only past 16 bits. Below min: rebasing is
                    # expensive, so fall back to raw.
                    delta = value - self._for_min
                    if delta < 0:
                        self._convert_for_to_raw()
                        self._data.append(value)
                    elif delta > self._for_mask:
                        if self._widen_for(delta):
                            self._append_for(value)
                        else:
                            self._data.append(value)
                    else:
                        self._append_for(value)
            else:
                self._data.append(value)

    def _get_delta_value(self, idx: int) -> int:
        """Reconstruct value from delta encoding at index (O(1) amortized).

        Uses a cached full reconstruction so repeated point/scan access does
        not re-sum the delta prefix on every call (previously O(n) per read).
        """
        return self._ensure_delta_cache()[idx]

    def __getitem__(self, idx: int) -> Any:
        if self._nullmask[idx]:
            return None
        if self.dtype.startswith("bytes"):
            if self._dict_mode and not self._dict_fallback:
                code = self._dict_codes[idx]
                return self._dict_values[code].decode("utf-8", errors="replace")
            return self._data[idx].decode("utf-8", errors="replace")
        if self.dtype == "bool":
            return bool((self._data >> idx) & 1)
        if self._delta_mode and not self._delta_fallback:
            return self._get_delta_value(idx)
        if self._for_mode and not self._for_fallback:
            return self._get_for_value(idx)
        return self._data[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        if idx < 0 or idx >= len(self._nullmask):
            raise IndexError(f"Index {idx} out of range")
        if value is None:
            self._nullmask[idx] = 1
            if self.dtype.startswith("bytes"):
                if self._dict_mode and not self._dict_fallback:
                    self._dict_codes[idx] = 0  # null placeholder
                else:
                    self._data[idx] = b""
            elif self.dtype == "bool":
                # Clear the bit
                self._data &= ~(1 << idx)
            elif self._delta_mode and not self._delta_fallback:
                # Delta values are cumulative — zeroing a delta would shift every
                # later row. Leave the chain intact; the nullmask hides this row.
                pass
            elif self._for_mode and not self._for_fallback:
                # FOR values are bit-packed and _data is empty; leave the packed
                # value intact (the nullmask hides this row).
                pass
            else:
                self._data[idx] = 0.0 if self._data.typecode in ("f", "d") else 0
        else:
            self._nullmask[idx] = 0
            if self.dtype.startswith("bytes"):
                if isinstance(value, str):
                    value = value.encode("utf-8")
                bval = bytes(value)
                if self._dict_mode and not self._dict_fallback:
                    code = self._add_to_dict(bval)
                    if code is not None:
                        self._dict_codes[idx] = code
                    else:
                        self._data[idx] = bval
                else:
                    self._data[idx] = bval
            elif self.dtype == "bool":
                if value:
                    self._data |= (1 << idx)
                else:
                    self._data &= ~(1 << idx)
            elif self._delta_mode and not self._delta_fallback:
                # Delta mode: update is expensive (must recalc from idx)
                # For now, convert to raw on update
                self._convert_delta_to_raw()
                self._data[idx] = value
            elif self._for_mode and not self._for_fallback:
                # FOR mode: update is expensive (must repack)
                # For now, convert to raw on update
                self._convert_for_to_raw()
                self._data[idx] = value
            else:
                self._data[idx] = value

    def __len__(self) -> int:
        return len(self._nullmask)

    def iter_valid(self) -> Iterator[Tuple[int, Any]]:
        """Iterate (index, value) for non-null entries — fast path."""
        data = self._data
        nullmask = self._nullmask
        is_bytes = self.dtype.startswith("bytes")
        is_bool = self.dtype == "bool"
        is_dict = self._dict_mode and not self._dict_fallback
        is_delta = self._delta_mode and not self._delta_fallback
        is_for = self._for_mode and not self._for_fallback
        if is_delta:
            # Reconstruct once (O(n)); previously this re-summed per element (O(n^2)).
            data = self._ensure_delta_cache()
        for i in range(len(nullmask)):
            if nullmask[i] == 0:
                if is_dict:
                    yield i, self._dict_values[self._dict_codes[i]].decode("utf-8", errors="replace")
                elif is_bytes:
                    yield i, data[i].decode("utf-8", errors="replace")
                elif is_bool:
                    yield i, bool((data >> i) & 1)
                elif is_for:
                    yield i, self._get_for_value(i)
                else:
                    yield i, data[i]

    def tolist(self) -> List[Any]:
        """Materialize the whole column to a Python list (None for nulls).

        Single O(n) pass that avoids per-element __getitem__ dispatch — used by
        column scans (select / select_column) for a large constant-factor win.
        """
        nullmask = self._nullmask
        n = len(nullmask)
        if self.dtype == "bool":
            data = self._data
            return [None if nullmask[i] else bool((data >> i) & 1) for i in range(n)]
        if self.dtype.startswith("bytes"):
            if self._dict_mode and not self._dict_fallback:
                values, codes = self._dict_values, self._dict_codes
                return [None if nullmask[i]
                        else values[codes[i]].decode("utf-8", errors="replace")
                        for i in range(n)]
            data = self._data
            return [None if nullmask[i]
                    else data[i].decode("utf-8", errors="replace")
                    for i in range(n)]
        if self._delta_mode and not self._delta_fallback:
            data = self._ensure_delta_cache()
        elif self._for_mode and not self._for_fallback:
            data = [self._get_for_value(i) for i in range(n)]
        else:
            data = self._data
        return [None if nullmask[i] else data[i] for i in range(n)]

    def count_valid(self) -> int:
        return self._nullmask.count(0)

    # ── Zero-copy / NumPy interop (issue #7) ─────────────────────────────────

    def _is_plain_numeric(self) -> bool:
        """True when values live in a contiguous array.array (no encoding)."""
        return (
            not self.dtype.startswith("bytes")
            and self.dtype != "bool"
            and not (self._dict_mode and not self._dict_fallback)
            and not (self._delta_mode and not self._delta_fallback)
            and not (self._for_mode and not self._for_fallback)
        )

    def buffer(self) -> memoryview:
        """Zero-copy ``memoryview`` over the raw numeric buffer.

        Only available for plain (un-encoded) numeric columns. While the
        returned view is alive the column cannot grow — the underlying
        ``array.array`` is locked against resizing by the buffer export — so
        release the view before further inserts. Null entries are not masked;
        pair with :meth:`null_mask`.
        """
        if not self._is_plain_numeric():
            raise TypeError(
                f"zero-copy buffer unavailable for column {self.name!r} "
                f"(dtype={self.dtype}; encoded or non-numeric) — use to_numpy()"
            )
        return memoryview(self._data)

    def __buffer__(self, flags):  # PEP 688 (Python 3.12+)
        return self.buffer()

    def to_numpy(self, zero_copy: bool = False):
        """Export the column as a NumPy array (requires ``numpy``).

        ``zero_copy=True`` returns a view that shares memory with the column
        (plain numeric columns only; see :meth:`buffer` for the lifetime
        caveat). The default returns a safe copy that also works for encoded
        columns; null entries come back as the column's zero value unless the
        column contains nulls, in which case an ``object`` array with ``None``
        is returned. Use :meth:`null_mask` for validity.
        """
        import numpy as np  # optional dependency, imported lazily

        if zero_copy and self._is_plain_numeric():
            return np.frombuffer(self.buffer(), dtype=_NUMPY_DTYPE[self._data.typecode])
        if self._is_plain_numeric() and self._nullmask.count(1) == 0:
            # np.array (not np.asarray) forces a real copy so the result does
            # not alias the column or lock its array against further inserts.
            return np.array(self._data, dtype=_NUMPY_DTYPE[self._data.typecode])
        return np.array(self.tolist(), dtype=object)

    def null_mask(self) -> List[bool]:
        """Return the validity bitmap as a list of bools (True == null)."""
        return [bool(x) for x in self._nullmask]

    def memory_usage(self) -> int:
        if self.dtype.startswith("bytes"):
            if self._dict_mode and not self._dict_fallback:
                # Dict mode: dict size + codes + values
                codes_size = len(self._dict_codes) * self._dict_codes.itemsize  # type: ignore
                values_size = sum(len(v) for v in self._dict_values)
                dict_overhead = len(self._dict_values) * 8  # pointer-ish overhead
                return codes_size + values_size + dict_overhead + len(self._nullmask)
            else:
                total = sum(len(d) for d in self._data)
                return total + len(self._nullmask) + len(self._data) * 8
        elif self.dtype == "bool":
            # Python int bitmask: ~1 bit per value
            return (self._data.bit_length() + 7) // 8 + len(self._nullmask)
        elif self._delta_mode and not self._delta_fallback:
            # Delta mode: base + deltas array
            return 8 + len(self._deltas) * self._deltas.itemsize + len(self._nullmask)  # type: ignore
        elif self._for_mode and not self._for_fallback:
            # FOR mode: packed int bitmask + metadata
            return (self._for_packed.bit_length() + 7) // 8 + 32 + len(self._nullmask)
        else:
            return len(self._data) * self._data.itemsize + len(self._nullmask)

    def unique_count(self) -> int:
        """Return number of unique non-null values in this column."""
        if self._dict_mode and not self._dict_fallback:
            return len(self._dict_values)
        # Read through the value accessor so encoded columns (delta / FOR, whose
        # raw _data array is empty) reconstruct correctly instead of IndexError-ing.
        nullmask = self._nullmask
        seen = set()
        for i in range(len(nullmask)):
            if not nullmask[i]:
                seen.add(self[i])
        return len(seen)


class ColumnarTable:
    """
    Simple columnar in-memory table for analytical workloads.
    Supports insert, batch insert, get, update, delete, select, aggregates.

    v0.4.0: Dictionary encoding for low-cardinality string columns.
    v0.5.0: Delta encoding for monotonic numeric columns.
    v0.6.0: Vectorized filtering, auto-indexing, NumPy export.
    v0.7.0: Frame-of-Reference + bit packing for bounded numeric ranges.
    """

    def __init__(self, name: str, schema: List[Tuple[str, str]],
                 dict_columns: Optional[List[str]] = None,
                 dict_threshold: int = 256,
                 delta_columns: Optional[List[str]] = None,
                 for_columns: Optional[List[str]] = None,
                 for_threshold: int = 50) -> None:
        self.name = name
        self.columns: Dict[str, Column] = {}
        self._col_list: List[Column] = []
        self._col_names: List[str] = []
        dict_cols = set(dict_columns or [])
        delta_cols = set(delta_columns or [])
        for_cols = set(for_columns or [])
        for col_name, col_type in schema:
            use_dict = col_name in dict_cols
            use_delta = col_name in delta_cols
            use_for = col_name in for_cols
            col = Column(col_name, col_type,
                         dict_encode=use_dict, dict_threshold=dict_threshold,
                         delta_encode=use_delta,
                         for_encode=use_for, for_threshold=for_threshold)
            self.columns[col_name] = col
            self._col_list.append(col)
            self._col_names.append(col_name)
        self._row_count = 0

    def insert(self, row: Dict[str, Any]) -> int:
        for col in self._col_list:
            col.append(row.get(col.name))
        idx = self._row_count
        self._row_count += 1
        return idx

    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        """Insert multiple rows at once — much faster than individual inserts."""
        start_idx = self._row_count
        for row in rows:
            for col in self._col_list:
                col.append(row.get(col.name))
            self._row_count += 1
        return start_idx

    def __len__(self) -> int:
        return self._row_count

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx < 0 or idx >= self._row_count:
            return None
        # Check first column's nullmask as quick check for deleted rows
        first_col = self._col_list[0]
        if first_col._nullmask[idx]:
            # Check if ALL columns are null (deleted row)
            all_null = True
            for col in self._col_list:
                if not col._nullmask[idx]:
                    all_null = False
                    break
            if all_null:
                return None
        return {col.name: col[idx] for col in self._col_list}

    def update(self, idx: int, row: Dict[str, Any]) -> None:
        if idx < 0 or idx >= self._row_count:
            raise IndexError(f"Row index {idx} out of range")
        current = self.get(idx)
        if current is None:
            raise KeyError(f"Row {idx} not found")
        merged = {**current, **row}
        for col in self._col_list:
            col[idx] = merged.get(col.name)

    def delete(self, idx: int) -> None:
        if idx < 0 or idx >= self._row_count:
            raise IndexError(f"Row index {idx} out of range")
        for col in self._col_list:
            col._nullmask[idx] = 1
            if col.dtype == "bool":
                # Clear the bit at idx in the bitmask
                col._data &= ~(1 << idx)
            elif col.dtype.startswith("bytes"):
                if col._dict_mode and not col._dict_fallback:
                    col._dict_codes[idx] = 0  # null placeholder
                else:
                    col._data[idx] = b""
            elif col._delta_mode and not col._delta_fallback:
                # Delta values are cumulative — mutating a delta would corrupt
                # every later row. The nullmask alone marks the row deleted.
                pass
            elif col._for_mode and not col._for_fallback:
                # FOR values are bit-packed — mutating would corrupt packing.
                # The nullmask alone marks the row deleted.
                pass
            elif hasattr(col._data, 'typecode') and col._data.typecode in ("f", "d"):
                col._data[idx] = 0.0
            else:
                col._data[idx] = 0

    def select(self,
               where: Optional[Callable[[Dict[str, Any]], bool]] = None,
               columns: Optional[List[str]] = None,
               limit: Optional[int] = None,
               offset: int = 0) -> List[Dict[str, Any]]:
        if columns is None:
            col_names = self._col_names
        else:
            for c in columns:
                if c not in self.columns:
                    raise ValueError(f"Unknown column: {c}")
            col_names = columns

        result: List[Dict[str, Any]] = []
        matched = 0

        all_cols = self._col_list
        first_nullmask = all_cols[0]._nullmask

        # Materialize the needed columns up front. This avoids per-cell
        # __getitem__ dispatch and, crucially, keeps delta columns O(n) instead
        # of O(n^2). When a predicate is present it sees every column, so all
        # columns are materialized; otherwise only the projected ones are.
        if where is not None:
            mat = {col.name: col.tolist() for col in all_cols}
        else:
            mat = {name: self.columns[name].tolist() for name in col_names}

        for idx in range(self._row_count):
            # Quick null check: skip if first col is null (likely deleted)
            if first_nullmask[idx]:
                continue

            if where is not None:
                row = {name: mat[name][idx] for name in self._col_names}
                if not where(row):
                    continue

            if matched < offset:
                matched += 1
                continue

            result.append({name: mat[name][idx] for name in col_names})
            matched += 1
            if limit is not None and len(result) >= limit:
                break
        return result

    def select_column(self, column_name: str) -> List[Any]:
        if column_name not in self.columns:
            raise ValueError(f"Unknown column: {column_name}")
        # tolist() is a single O(n) pass (and O(n) — not O(n^2) — for delta cols).
        return self.columns[column_name].tolist()

    def to_numpy(self, column_name: str, zero_copy: bool = False):
        """Export a column as a NumPy array (requires ``numpy``). See
        :meth:`Column.to_numpy`."""
        if column_name not in self.columns:
            raise ValueError(f"Unknown column: {column_name}")
        return self.columns[column_name].to_numpy(zero_copy=zero_copy)

    def column_buffer(self, column_name: str) -> memoryview:
        """Zero-copy ``memoryview`` over a plain numeric column's raw buffer."""
        if column_name not in self.columns:
            raise ValueError(f"Unknown column: {column_name}")
        return self.columns[column_name].buffer()

    # ── Vectorized multi-condition filter (issue #4) ─────────────────────────

    def _normalize_conditions(self, conditions) -> List[Tuple[str, str, Any]]:
        """Accept either a list of ``(column, op, value)`` triples or a dict
        ``{column: value}`` / ``{column: {op: value}}`` and return triples."""
        norm: List[Tuple[str, str, Any]] = []
        if isinstance(conditions, dict):
            for col, spec in conditions.items():
                if isinstance(spec, dict):
                    for op, val in spec.items():
                        norm.append((col, op, val))
                else:
                    norm.append((col, "eq", spec))
        else:
            for cond in conditions:
                if len(cond) != 3:
                    raise ValueError(f"condition must be (column, op, value): {cond!r}")
                norm.append((cond[0], cond[1], cond[2]))
        for col, op, _ in norm:
            if col not in self.columns:
                raise ValueError(f"Unknown column: {col}")
            if op not in _FILTER_OPS and op not in ("in", "between"):
                raise ValueError(f"Unsupported operator: {op!r}")
        return norm

    def _condition_mask(self, col_name: str, op: str, value: Any,
                        materialized: Dict[str, List[Any]]) -> bytearray:
        """Build a 1-byte-per-row match mask for a single condition."""
        col = self.columns[col_name]
        vals = materialized.get(col_name)
        if vals is None:
            vals = col.tolist()
            materialized[col_name] = vals

        # bytes columns materialize to decoded str — normalize comparison
        # values (given as bytes or str) the same way so they compare equal.
        if col.dtype.startswith("bytes"):
            def _conv(x):
                return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
        else:
            def _conv(x):
                return x

        mask = bytearray(len(vals))
        if op == "in":
            members = {_conv(m) for m in value}
            for i, v in enumerate(vals):
                if v is not None and v in members:
                    mask[i] = 1
        elif op == "between":
            lo, hi = _conv(value[0]), _conv(value[1])
            for i, v in enumerate(vals):
                if v is not None and lo <= v <= hi:
                    mask[i] = 1
        else:
            target = _conv(value)
            fn = _FILTER_OPS[op]
            for i, v in enumerate(vals):
                if v is not None and fn(v, target):
                    mask[i] = 1
        return mask

    def count_where(self, conditions, combine: str = "and",
                    use_numpy: Optional[bool] = None) -> int:
        """Count rows matching the conditions, without materializing any rows.

        A fast analytical primitive (``SELECT COUNT(*) WHERE ...``): builds the
        condition masks and counts the matches. NumPy-accelerated when available.
        """
        if combine not in ("and", "or"):
            raise ValueError(f"combine must be 'and' or 'or', got {combine!r}")
        norm = self._normalize_conditions(conditions)
        n = self._row_count
        if n == 0:
            return 0
        materialized: Dict[str, List[Any]] = {}
        if _HAS_NUMPY and (use_numpy is None or use_numpy):
            import numpy as np
            live = np.frombuffer(self._col_list[0]._nullmask, dtype=np.int8) == 0
            if not norm:
                return int(live.sum())
            masks = [self._condition_mask_numpy(self.columns[c], op, v, materialized, np)
                     for (c, op, v) in norm]
            comb = masks[0]
            if combine == "or":
                for m in masks[1:]:
                    comb = comb | m
            else:
                for m in masks[1:]:
                    comb = comb & m
            return int((comb & live).sum())
        # Pure-Python: combine byte masks as a big integer and popcount.
        live = bytearray(n)
        first_nullmask = self._col_list[0]._nullmask
        for i in range(n):
            if not first_nullmask[i]:
                live[i] = 1
        live_int = int.from_bytes(live, "little")
        if not norm:
            return bin(live_int).count("1")
        masks = [int.from_bytes(self._condition_mask(c, op, v, materialized), "little")
                 for (c, op, v) in norm]
        comb = masks[0]
        if combine == "or":
            for m in masks[1:]:
                comb |= m
        else:
            for m in masks[1:]:
                comb &= m
        return bin(comb & live_int).count("1")

    def _condition_mask_numpy(self, col: "Column", op: str, value: Any,
                              materialized: Dict[str, List[Any]], np):
        """Build a boolean NumPy mask for one condition. Plain numeric columns
        compare vectorially over the raw buffer; other columns reuse the Python
        byte mask (converted)."""
        if col._is_plain_numeric():
            arr = np.frombuffer(col.buffer(), dtype=_NUMPY_DTYPE[col._data.typecode])
            valid = np.frombuffer(col._nullmask, dtype=np.int8) == 0
            if op in ("eq", "=="):
                m = arr == value
            elif op in ("ne", "!="):
                m = arr != value
            elif op in ("gt", ">"):
                m = arr > value
            elif op in ("gte", ">="):
                m = arr >= value
            elif op in ("lt", "<"):
                m = arr < value
            elif op in ("lte", "<="):
                m = arr <= value
            elif op == "in":
                m = np.isin(arr, np.asarray(list(value)))
            elif op == "between":
                lo, hi = value
                m = (arr >= lo) & (arr <= hi)
            else:  # pragma: no cover - ops are validated upstream
                m = np.zeros(arr.shape[0], dtype=bool)
            return m & valid  # nulls never match
        bm = self._condition_mask(col.name, op, value, materialized)
        return np.frombuffer(bytes(bm), dtype=np.int8) != 0

    def _select_where_numpy(self, norm, col_names, n, limit, offset, combine, materialized):
        import numpy as np
        live = np.frombuffer(self._col_list[0]._nullmask, dtype=np.int8) == 0
        if not norm:
            final = live
        else:
            masks = [self._condition_mask_numpy(self.columns[c], op, v, materialized, np)
                     for (c, op, v) in norm]
            comb = masks[0]
            if combine == "or":
                for m in masks[1:]:
                    comb = comb | m
            else:
                for m in masks[1:]:
                    comb = comb & m
            final = comb & live
        idx = np.nonzero(final)[0]
        if offset:
            idx = idx[offset:]
        if limit is not None:
            idx = idx[:limit]
        if idx.size == 0:
            return []
        for c in col_names:
            if c not in materialized:
                materialized[c] = self.columns[c].tolist()
        proj = [(name, materialized[name]) for name in col_names]
        return [{name: vals[i] for name, vals in proj} for i in idx.tolist()]

    def select_where(self, conditions, columns: Optional[List[str]] = None,
                     limit: Optional[int] = None, offset: int = 0,
                     combine: str = "and",
                     use_numpy: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Filter rows with one or more conditions, combined vectorially.

        ``conditions`` is a list of ``(column, op, value)`` triples (op in
        eq/ne/gt/gte/lt/lte/in/between) or the dict shorthand. Each condition is
        evaluated column-at-a-time into a mask; masks are combined with
        ``AND``/``OR`` (``combine``). When NumPy is installed the masks are built
        vectorially over the column buffers (issue #14); ``use_numpy=False``
        forces the pure-Python big-integer path. Returns projected rows.
        """
        if combine not in ("and", "or"):
            raise ValueError(f"combine must be 'and' or 'or', got {combine!r}")
        norm = self._normalize_conditions(conditions)
        n = self._row_count
        if n == 0:
            return []

        if columns is None:
            col_names = self._col_names
        else:
            for c in columns:
                if c not in self.columns:
                    raise ValueError(f"Unknown column: {c}")
            col_names = columns

        materialized: Dict[str, List[Any]] = {}

        if _HAS_NUMPY and (use_numpy is None or use_numpy):
            return self._select_where_numpy(norm, col_names, n, limit, offset,
                                            combine, materialized)

        # Live rows: exclude deleted rows via the first-column-null heuristic
        # (matching select()).
        first_nullmask = self._col_list[0]._nullmask
        live = bytearray(n)
        for i in range(n):
            if not first_nullmask[i]:
                live[i] = 1
        live_int = int.from_bytes(live, "little")

        if not norm:
            final_int = live_int
        else:
            masks = [int.from_bytes(self._condition_mask(c, op, v, materialized), "little")
                     for (c, op, v) in norm]
            if combine == "or":
                comb = 0
                for m in masks:
                    comb |= m
            else:
                comb = masks[0]
                for m in masks[1:]:
                    comb &= m
            final_int = comb & live_int

        if final_int == 0:
            return []
        result = final_int.to_bytes(n, "little")

        for c in col_names:
            if c not in materialized:
                materialized[c] = self.columns[c].tolist()
        proj = [(name, materialized[name]) for name in col_names]

        out: List[Dict[str, Any]] = []
        matched = 0
        for i in range(n):
            if result[i]:
                if matched < offset:
                    matched += 1
                    continue
                out.append({name: vals[i] for name, vals in proj})
                matched += 1
                if limit is not None and len(out) >= limit:
                    break
        return out

    def _numpy_aggregate(self, col: "Column", agg: str):
        """NumPy-accelerated aggregate for a plain numeric column, no WHERE.

        Returns ``(handled, value)``. ``handled`` is False when NumPy can't be
        used while preserving exact parity with the pure-Python path (empty
        result, or a 64-bit integer sum where a fixed-width accumulator could
        disagree with Python's arbitrary precision) — the caller then falls
        back. Reads the column's raw buffer with no copy.
        """
        import numpy as np
        tc = col._data.typecode
        arr = np.frombuffer(col.buffer(), dtype=_NUMPY_DTYPE[tc])
        # Detect/filter nulls with NumPy over a zero-copy view of the nullmask —
        # array.array.count() is ~10x slower and would dominate the runtime.
        nm_view = np.frombuffer(col._nullmask, dtype=np.int8)
        if nm_view.any():
            arr = arr[nm_view == 0]
        if arr.size == 0:
            return False, None
        if tc in ("f", "d") and np.isnan(arr).any():
            # NumPy min/max propagate NaN while the pure-Python comparison path
            # skips it — defer to Python so both paths return the same result.
            return False, None
        if agg == "min":
            return True, arr.min().item()
        if agg == "max":
            return True, arr.max().item()
        if agg == "avg":
            return True, float(arr.mean(dtype=np.float64))
        if agg == "sum":
            if tc in ("f", "d"):
                return True, float(arr.sum(dtype=np.float64))
            if col.width <= 4:
                # i8..i32 / u8..u32: a 64-bit accumulator can't overflow for any
                # realistic row count, so this matches Python's exact sum.
                acc = np.uint64 if tc in ("B", "H", "I", "Q") else np.int64
                return True, int(arr.sum(dtype=acc))
            # i64 / u64: defer to the exact Python sum to avoid overflow drift.
            return False, None
        return False, None

    def aggregate(self,
                  column_name: str,
                  agg: str = "sum",
                  where: Optional[Callable[[Dict[str, Any]], bool]] = None,
                  use_numpy: Optional[bool] = None) -> Any:
        if column_name not in self.columns:
            raise ValueError(f"Unknown column: {column_name}")
        col = self.columns[column_name]

        if agg == "count":
            return col.count_valid()

        # NumPy-accelerated path (issue #14): plain numeric column, no WHERE.
        # Auto-enabled when NumPy is installed; pass use_numpy=False to force the
        # pure-Python path. Falls through when NumPy can't preserve exact parity.
        want_numpy = _HAS_NUMPY and (use_numpy is None or use_numpy)
        if (where is None and want_numpy and col._is_plain_numeric()
                and len(col._data) > 0):
            handled, value = self._numpy_aggregate(col, agg)
            if handled:
                return value

        # Vectorized path: plain (array-backed) numeric column with no nulls and
        # no where clause. sum()/min()/max() over an array.array run in C, far
        # faster than a Python per-element loop.
        if where is None:
            is_plain = (
                not col.dtype.startswith("bytes")
                and col.dtype != "bool"
                and not (col._delta_mode and not col._delta_fallback)
            )
            if is_plain and len(col._data) > 0 and col._nullmask.count(1) == 0:
                data = col._data
                if agg == "sum":
                    return sum(data)
                if agg == "avg":
                    return sum(data) / len(data)
                if agg == "min":
                    return min(data)
                if agg == "max":
                    return max(data)

        # Fast path: no where clause — iterate column directly
        if where is None:
            if agg == "sum":
                total = 0
                for _, v in col.iter_valid():
                    total += v
                return total
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

        # Slow path: with where clause. Materialize columns once (keeps delta
        # columns O(n), avoids per-cell dispatch) and read the aggregate value
        # via the materialized list — col._data[idx] is wrong for encoded
        # columns (e.g. delta mode stores deltas, not values).
        all_cols = self._col_list
        col_names = self._col_names
        mat = {c.name: c.tolist() for c in all_cols}
        target = mat[column_name]
        first_nullmask = all_cols[0]._nullmask
        col_nullmask = col._nullmask

        def _matching_values():
            for idx in range(self._row_count):
                if first_nullmask[idx] or col_nullmask[idx]:
                    continue
                row = {name: mat[name][idx] for name in col_names}
                if where(row):
                    yield target[idx]

        if agg == "sum":
            return sum(_matching_values())
        if agg == "avg":
            total = 0
            count = 0
            for v in _matching_values():
                total += v
                count += 1
            return total / count if count else 0
        if agg == "min":
            min_val = None
            for v in _matching_values():
                if min_val is None or v < min_val:
                    min_val = v
            return min_val
        if agg == "max":
            max_val = None
            for v in _matching_values():
                if max_val is None or v > max_val:
                    max_val = v
            return max_val

        raise ValueError(f"Unsupported aggregation: {agg}")

    def memory_usage(self) -> int:
        return sum(col.memory_usage() for col in self._col_list)

    def __repr__(self) -> str:
        return f"ColumnarTable({self.name!r}, rows={self._row_count}, cols={self._col_names})"
