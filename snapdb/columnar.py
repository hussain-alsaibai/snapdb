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
v0.10.0: Bitmask query engine, PEP 688 __buffer__ on tables, to_dataframe,
         compact() optimization.
"""

from __future__ import annotations

import array
import importlib.util
import operator
from typing import Any, Dict, List, Tuple, Callable, Iterator, Optional

from ._util import _TYPE_CODES, _type_size, _norm_value

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


# array.array has no "?" typecode, so bool columns use "B" instead.
_ARRAY_TYPECODES = {**_TYPE_CODES, "bool": "B"}


def _array_typecode(dtype: str) -> str:
    if dtype.startswith("bytes"):
        return "list"
    return _ARRAY_TYPECODES[dtype]


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


def _rebuild_column_alive(col: "Column", live: bytearray) -> int:
    """Rebuild a column's value/null-mask arrays with dead rows dropped.

    Returns the estimated bytes freed (col's old memory footprint minus the
    rebuilt one), or ``0`` if the column is non-rebuildable for the current
    encoding (e.g. dict overflow), in which case the column is left intact.
    """
    before = col.memory_usage()
    alive_n = sum(1 for b in live if b)
    if col.dtype == "bool":
        # Bit-packed: rebuild the bytearray from the live bits.
        old = col._data
        new = bytearray((alive_n + 7) // 8)
        out_idx = 0
        for i, b in enumerate(live):
            if b:
                if (i >> 3) < len(old) and (old[i >> 3] >> (i & 7)) & 1:
                    new[out_idx >> 3] |= 1 << (out_idx & 7)
                out_idx += 1
        col._data = new
        col._nullmask = array.array("b", [0] * alive_n)
        return max(0, before - col.memory_usage())
    if col.dtype.startswith("bytes"):
        if col._dict_mode and not col._dict_fallback:
            # Drop dead rows from the dict-coded column; rebuild the dictionary
            # so the codes stay dense.
            new_values: List[bytes] = []
            new_codes = array.array(col._dict_codes.typecode)
            old_codes = col._dict_codes
            mapping: Dict[int, int] = {}
            for i, b in enumerate(live):
                if not b:
                    continue
                old_code = old_codes[i]
                if old_code not in mapping:
                    mapping[old_code] = len(new_values)
                    new_values.append(col._dict_values[old_code])
                new_codes.append(mapping[old_code])
            col._dict_values = new_values
            col._dict = {v: i for i, v in enumerate(new_values)}
            col._dict_codes = new_codes
            col._nullmask = array.array("b", [0] * alive_n)
            return max(0, before - col.memory_usage())
        # Plain list-of-bytes fallback.
        new_list = [col._data[i] for i, b in enumerate(live) if b]
        col._data = new_list
        col._nullmask = array.array("b", [0] * alive_n)
        return max(0, before - col.memory_usage())
    if col._delta_mode and not col._delta_fallback:
        cache = col._ensure_delta_cache()
        new_deltas = array.array(col._delta_typecode or "q")
        prev = col._delta_base
        for i, b in enumerate(live):
            if not b:
                continue
            new_deltas.append(int(cache[i]) - prev)
            prev = int(cache[i])
        col._deltas = new_deltas
        col._delta_cache = None
        col._nullmask = array.array("b", [0] * alive_n)
        return max(0, before - col.memory_usage())
    if col._for_mode and not col._for_fallback:
        # Rebuild the bit-packed FOR storage. Decode the live values, then
        # re-pack against the new min/max so the encoding stays tight.
        values = [col._get_for_value(i) for i in range(len(col._nullmask))
                  if live[i]]
        if not values:
            col._nullmask = array.array("b", [])
            return max(0, before - col.memory_usage())
        new_min = min(values)
        new_max = max(values)
        new_range = new_max - new_min
        new_bits = max(1, new_range.bit_length())
        new_mask = (1 << new_bits) - 1
        # Drop the legacy _for_packed integer (which is a 64-bit OR of the
        # bit-windows) and let the column re-pack lazily on the next write.
        col._for_min = new_min
        col._for_bits = new_bits
        col._for_mask = new_mask
        col._for_packed = 0
        col._for_count = len(values)
        col._for_cache = None
        col._nullmask = array.array("b", [0] * alive_n)
        return max(0, before - col.memory_usage())
    # Plain numeric column.
    new_arr = array.array(col._data.typecode)
    new_arr.extend(int(col._data[i]) for i, b in enumerate(live) if b)
    col._data = new_arr
    col._nullmask = array.array("b", [0] * alive_n)
    return max(0, before - col.memory_usage())


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
        "_for_cache",
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
        # Lazily-built full reconstruction of a FOR-packed column, mirroring
        # _delta_cache — an aggregate over a bit-packed column previously
        # decoded each value one shift-and-mask at a time through a Python
        # generator (iter_valid); building the array once and reducing it
        # with the C-level sum()/min()/max() is faster for repeated reads.
        self._for_cache: Optional[array.array] = None
        self._data: Any = None
        self._nullmask: Optional[array.array] = None
        self._init_storage()

    def _init_storage(self) -> None:
        if self.dtype == "bool":
            # Bit-packed boolean storage: mutable bytearray bitset where bit i
            # = value of row i (1=True, 0=False). A bytearray sets bits in
            # place — a Python int bitmask is immutable, so every append would
            # copy the whole mask (O(n) per append, O(n^2) per column build).
            self._data = bytearray()
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

    def _bool_get(self, idx: int) -> bool:
        """Read bit ``idx`` of the bool bitset (missing bytes read as False)."""
        byte_idx = idx >> 3
        if byte_idx >= len(self._data):
            return False
        return bool((self._data[byte_idx] >> (idx & 7)) & 1)

    def _bool_set(self, idx: int, value: bool) -> None:
        """Set/clear bit ``idx`` of the bool bitset in place (O(1))."""
        byte_idx = idx >> 3
        if value:
            while len(self._data) <= byte_idx:
                self._data.append(0)
            self._data[byte_idx] |= 1 << (idx & 7)
        elif byte_idx < len(self._data):
            self._data[byte_idx] &= 0xFF ^ (1 << (idx & 7))

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
        """Switch to delta encoding mode.

        Deltas are differences, so storage is always SIGNED regardless of the
        column dtype (a u64 column can still step downward between rows);
        _append_delta widens the typecode on demand.
        """
        self._delta_mode = True
        self._delta_base = value
        self._delta_prev = value
        self._deltas = array.array("i")
        self._delta_typecode = "i"
        # Remove first value from raw data (it becomes base)
        self._data = array.array(_array_typecode(self.dtype))

    def _upgrade_delta_storage(self, needed_max: int) -> None:
        """Upgrade delta array to larger typecode if needed. Always signed:
        deltas can be negative even for unsigned column dtypes, and an
        unsigned target typecode would OverflowError copying existing
        negative deltas."""
        new_tc = _smallest_delta_typecode(needed_max, signed=True)
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
        self._for_cache = None  # invalidate reconstruction cache

    def _get_for_value(self, idx: int) -> int:
        """Reconstruct value from FOR encoding at index."""
        bit_pos = idx * self._for_bits
        delta = (self._for_packed >> bit_pos) & self._for_mask
        return self._for_min + delta

    def _ensure_for_cache(self) -> array.array:
        """Return a cached full reconstruction, building it once if needed
        (mirrors _ensure_delta_cache)."""
        cache = self._for_cache
        if cache is None:
            cache = array.array(_array_typecode(self.dtype))
            for i in range(self._for_count):
                cache.append(self._get_for_value(i))
            self._for_cache = cache
        return cache

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
        self._for_cache = None

    def _append_delta(self, value: int) -> None:
        """Append value using delta encoding."""
        self._delta_cache = None  # invalidate reconstruction cache
        delta = value - self._delta_prev
        self._delta_prev = value

        # Check if delta fits in current typecode
        if self._delta_typecode == "b":
            if not (-128 <= delta <= 127):
                self._upgrade_delta_storage(abs(delta))
        elif self._delta_typecode == "h":
            if not (-32768 <= delta <= 32767):
                self._upgrade_delta_storage(abs(delta))
        elif self._delta_typecode == "i":
            if not (-2147483648 <= delta <= 2147483647):
                self._upgrade_delta_storage(abs(delta))

        try:
            self._deltas.append(delta)
        except OverflowError:
            # Delta exceeds even i64 range (possible for u64 columns whose
            # values span more than 2^63) — delta encoding can't represent it;
            # fall back to raw storage for the whole column.
            self._convert_delta_to_raw()
            self._data.append(value)

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
                if value:
                    self._bool_set(len(self._nullmask) - 1, True)
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
                            # Convert existing samples through _append_delta so
                            # typecode upgrades / raw fallback apply to sample
                            # deltas too (a bare append could OverflowError on
                            # a large step between samples).
                            for v in old_data[1:]:
                                if self._delta_fallback:
                                    self._data.append(v)
                                else:
                                    self._append_delta(v)
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
                        self._for_stats = {"min": value, "max": value, "count": 0}
                    self._for_stats["count"] += 1
                    if value < self._for_stats["min"]:
                        self._for_stats["min"] = value
                    if value > self._for_stats["max"]:
                        self._for_stats["max"] = value
                    # Check if we should enable FOR mode
                    if self._for_stats["count"] >= self._for_threshold:
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
            return self._bool_get(idx)
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
                self._bool_set(idx, False)
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
                self._bool_set(idx, bool(value))
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
                    yield i, self._bool_get(i)
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
            nd = len(data)
            return [None if nullmask[i]
                    else bool(((data[i >> 3] if (i >> 3) < nd else 0) >> (i & 7)) & 1)
                    for i in range(n)]
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
            # Bytearray bitset: ~1 bit per value
            return len(self._data) + len(self._nullmask)
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

    def _live_mask(self) -> bytearray:
        """Per-row liveness bitmap (1 = live). A row is deleted only when EVERY
        column is null — that is how delete() marks rows — so a legitimately
        null first column must not hide a live row from scans/aggregates (it
        previously did: get() saw the row while select/count/group_by skipped
        it). The first column short-circuits the common case."""
        cols = self._col_list
        first = cols[0]._nullmask
        rest = cols[1:]
        n = self._row_count
        live = bytearray(n)
        for i in range(n):
            if not first[i]:
                live[i] = 1
            else:
                for col in rest:
                    if not col._nullmask[i]:
                        live[i] = 1
                        break
        return live

    def _live_mask_np(self, np):
        """NumPy boolean liveness mask (True = live), for the vectorized
        filter/count paths. Same semantics as :meth:`_live_mask` (a row is
        dead only when EVERY column is null) but computed with C-level array
        ops instead of a per-row Python loop — otherwise the correctness fix
        would make liveness the slowest part of an otherwise-vectorized scan.

        ``all_null`` is intersected column-by-column; the intersection only
        shrinks, so once no row is still all-null we stop early (the common
        no-deletes / NOT-NULL-first-column case exits after one column)."""
        cols = self._col_list
        all_null = np.frombuffer(cols[0]._nullmask, dtype=np.int8) != 0
        for col in cols[1:]:
            if not all_null.any():
                break
            all_null &= np.frombuffer(col._nullmask, dtype=np.int8) != 0
        return ~all_null

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
                col._bool_set(idx, False)
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

    def _matching_indices(self, predicate: Optional[Callable[[Dict[str, Any]], bool]]) -> List[int]:
        live = self._live_mask()
        if predicate is None:
            return [i for i in range(self._row_count) if live[i]]

        mat = {col.name: col.tolist() for col in self._col_list}
        names = self._col_names
        out = []
        for i in range(self._row_count):
            if not live[i]:
                continue
            row = {name: mat[name][i] for name in names}
            if predicate(row):
                out.append(i)
        return out

    def batch_update(self, predicate: Callable[[Dict[str, Any]], bool],
                     updates: Dict[str, Any]) -> int:
        """Column-wise batch update without materializing each full row twice."""
        for name in updates:
            if name not in self.columns:
                raise ValueError(f"Unknown column: {name}")
        matches = self._matching_indices(predicate)
        for name, value in updates.items():
            col = self.columns[name]
            for idx in matches:
                col[idx] = value
        return len(matches)

    def group_by(self, key_column: str, value_column: str, agg: str = "count") -> Dict[Any, Any]:
        """Columnar grouped aggregate using only the two participating columns."""
        if key_column not in self.columns:
            raise ValueError(f"Unknown column: {key_column}")
        if value_column not in self.columns:
            raise ValueError(f"Unknown column: {value_column}")

        keys = self.columns[key_column].tolist()
        vals = self.columns[value_column].tolist()
        live = self._live_mask()
        groups: Dict[Any, Any] = {}
        counts: Dict[Any, int] = {}

        for i in range(self._row_count):
            if not live[i]:
                continue
            key = keys[i]
            val = vals[i]
            if val is None:
                continue
            if agg == "count":
                groups[key] = groups.get(key, 0) + 1
            elif agg == "sum":
                groups[key] = groups.get(key, 0) + val
            elif agg == "avg":
                groups[key] = groups.get(key, 0) + val
                counts[key] = counts.get(key, 0) + 1
            elif agg == "min":
                groups[key] = val if key not in groups or val < groups[key] else groups[key]
            elif agg == "max":
                groups[key] = val if key not in groups or val > groups[key] else groups[key]
            else:
                raise ValueError(f"Unsupported aggregate: {agg}")

        if agg == "avg":
            return {key: total / counts[key] for key, total in groups.items()}
        return groups

    def select(self,
               where: Optional[Callable[[Dict[str, Any]], bool]] = None,
               columns: Optional[List[str]] = None,
               limit: Optional[int] = None,
               offset: int = 0) -> List[Dict[str, Any]]:
        if limit is not None and limit <= 0:
            return []

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
        live = self._live_mask()

        # Materialize the needed columns up front. This avoids per-cell
        # __getitem__ dispatch and, crucially, keeps delta columns O(n) instead
        # of O(n^2). When a predicate is present it sees every column, so all
        # columns are materialized; otherwise only the projected ones are.
        if where is not None:
            mat = {col.name: col.tolist() for col in all_cols}
        else:
            mat = {name: self.columns[name].tolist() for name in col_names}

        for idx in range(self._row_count):
            if not live[idx]:
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

    def __buffer__(self, flags: int) -> memoryview:
        """PEP 688: expose a column buffer for zero-copy NumPy access.

        Returns the buffer of the first column. Useful for the common case
        where a :class:`ColumnarTable` holds a single primary-key-style column
        and the caller wants to hand it straight to NumPy without copying.
        For multi-column tables, prefer :meth:`column_buffer` to pick the
        column explicitly. An empty table returns a 0-byte memoryview, which
        matches :meth:`Column.buffer` semantics.
        """
        if not self._col_list:
            # No schema columns at all — fall through to a 0-byte view.
            return memoryview(b"")
        return self._col_list[0].buffer()

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
            _conv = _norm_value
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
            live = self._live_mask_np(np)
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
        live_int = int.from_bytes(self._live_mask(), "little")
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
            # f32 columns: compare at float64 like the pure-Python path (which
            # widens stored float32 to float), so a literal such as 0.1 matches
            # identically instead of at float32 precision.
            if col._data.typecode == "f":
                arr = arr.astype(np.float64)
            valid = np.frombuffer(col._nullmask, dtype=np.int8) == 0
            if op == "in":
                m = np.isin(arr, np.asarray(list(value)))
            elif op == "between":
                lo, hi = value
                m = (arr >= lo) & (arr <= hi)
            else:
                # operator.eq/gt/... apply element-wise to NumPy arrays, so the
                # shared table serves here too (ops are validated upstream).
                m = _FILTER_OPS[op](arr, value)
            return m & valid  # nulls never match

        # Dict-encoded string column: equality/membership can compare the
        # integer codes (NumPy-fast) instead of per-row string comparison.
        # (Ordering ops can't — codes aren't lexically ordered — so they fall
        # through to the Python mask.)
        if (col._dict_mode and not col._dict_fallback
                and op in ("eq", "==", "ne", "!=", "in")):
            codes = np.frombuffer(col._dict_codes, dtype=_NUMPY_DTYPE[col._dict_codes.typecode])
            valid = np.frombuffer(col._nullmask, dtype=np.int8) == 0

            def _code(v):
                vb = v.encode("utf-8") if isinstance(v, str) else bytes(v)
                return col._dict.get(vb)  # None if the value isn't in the dict

            if op == "in":
                wanted = [c for c in (_code(m) for m in value) if c is not None]
                m = np.isin(codes, np.asarray(wanted)) if wanted else np.zeros(codes.shape[0], dtype=bool)
            else:
                c = _code(value)
                if op in ("eq", "=="):
                    m = (codes == c) if c is not None else np.zeros(codes.shape[0], dtype=bool)
                else:  # ne: a value not in the dict differs from every stored value
                    m = (codes != c) if c is not None else np.ones(codes.shape[0], dtype=bool)
            return m & valid

        bm = self._condition_mask(col.name, op, value, materialized)
        return np.frombuffer(bytes(bm), dtype=np.int8) != 0

    def _select_where_numpy(self, norm, col_names, n, limit, offset, combine, materialized):
        import numpy as np
        live = self._live_mask_np(np)
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
        if limit is not None and limit <= 0:
            return []
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

        # Live rows: exclude deleted rows (matching select()).
        live_int = int.from_bytes(self._live_mask(), "little")

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

    # ── Bitmask query engine (v0.10.0) ───────────────────────────────────────
    #
    # Bit-level predicates on integer columns: ``(column_value >> position) & 1``
    # is compared against an expected bit value. Rows are emitted when the
    # combination semantics (AND / OR / XOR across the supplied bits) match.
    # Skips null rows (the encoded value is ``None`` / not addressable).

    def select_bitmask(
        self,
        bitmask: Dict[str, Any],
        match_all: bool = True,
        operator: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query rows using bitmask operations on integer columns.

        Args:
            bitmask: ``{column: predicate}`` where ``predicate`` is one of:

                - ``(bit_position, expected_value)`` — single bit, expected
                  value 0 or 1.
                - ``[(pos_0, val_0), (pos_1, val_1), ...]`` — multiple bits
                  on the same column, combined internally with AND (every
                  listed bit must match for that column to contribute).

                For each row, the bit at ``bit_position`` (0-indexed from LSB)
                of the column's integer value is compared against
                ``expected_value`` (0 or 1). The engine supports three
                combination semantics across the supplied columns
                (``operator``), exposed as the standard bitwise ops:

                - ``"AND"`` (default when ``match_all=True``): every column's
                  bit predicate must match.
                - ``"OR"`` (default when ``match_all=False``): any column's
                  bit predicate matches.
                - ``"XOR"``: an odd number of columns' bit predicates must
                  match — useful for parity-style predicates.

            columns: optional projected column list (defaults to all columns).
            limit: cap on returned rows.
            offset: skip this many matching rows before emitting.

        Example:
            db.select_bitmask({'status': (0, 1), 'type': (5, 1)})
            # returns rows where bit 0 of 'status' == 1 AND bit 5 of 'type' == 1

            db.select_bitmask({'flags': [(0, 1), (3, 1)]}, operator='XOR')
            # returns rows where exactly one of bits 0/3 of 'flags' is set
        """
        if not bitmask:
            return []
        if operator is None:
            operator = "AND" if match_all else "OR"
        operator = operator.upper()
        if operator not in ("AND", "OR", "XOR"):
            raise ValueError(
                f"operator must be 'AND', 'OR', or 'XOR', got {operator!r}"
            )

        # Normalise the bitmask spec: each value becomes a list[(pos, val)].
        norm: Dict[str, List[Tuple[int, int]]] = {}
        for col, spec in bitmask.items():
            if col not in self.columns:
                raise ValueError(f"Unknown column: {col!r}")
            if isinstance(spec, tuple):
                norm[col] = [spec]
            elif isinstance(spec, list):
                norm[col] = list(spec)
            else:
                raise TypeError(
                    f"bitmask values for {col!r} must be (pos, val) or a list "
                    f"of (pos, val) tuples, got {type(spec).__name__}"
                )

        # Validate columns and materialise the integer values column-at-a-time.
        # tolist() handles delta / FOR / dict-encoded columns in a single O(n)
        # pass each, so the overall scan is O(n * k) for k bitmask columns.
        col_values: Dict[str, List[Any]] = {}
        for col in norm:
            if self.columns[col].dtype.startswith("bytes"):
                raise TypeError(
                    f"bitmask queries need an integer column, got {col!r} "
                    f"with dtype={self.columns[col].dtype!r}"
                )
            col_values[col] = self.columns[col].tolist()

        if columns is None:
            col_names = self._col_names
        else:
            for c in columns:
                if c not in self.columns:
                    raise ValueError(f"Unknown column: {c!r}")
            col_names = columns

        # Project the requested columns once. Avoids per-row dict construction
        # with the bits (the bitmask lookups live in col_values).
        mat = {name: self.columns[name].tolist() for name in col_names}
        live = self._live_mask()
        n = self._row_count

        def _column_iv_match(iv: int, bits: List[Tuple[int, int]]) -> bool:
            """AND-combine the bits within a single column."""
            for pos, val in bits:
                if (iv >> pos) & 1 != (1 if val else 0):
                    return False
            return True

        out: List[Dict[str, Any]] = []
        matched = 0
        for i in range(n):
            if not live[i]:
                continue
            col_matches: Dict[str, bool] = {}
            saw_value = False
            for col, bits in norm.items():
                v = col_values[col][i]
                if v is None:
                    col_matches[col] = False
                    continue
                saw_value = True
                col_matches[col] = _column_iv_match(int(v), bits)

            if operator == "AND":
                # Every column (that has a value) must match.
                ok = saw_value and all(col_matches.values())
            elif operator == "OR":
                ok = any(col_matches.values())
            else:  # XOR — odd number of columns must match
                ok = sum(1 for v in col_matches.values() if v) % 2 == 1

            if not ok:
                continue
            if matched < offset:
                matched += 1
                continue
            out.append({name: mat[name][i] for name in col_names})
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

        if agg == "count" and where is None:
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

            # Cached-reconstruction path: delta/FOR-encoded numeric columns,
            # no nulls. The full reconstruction is already cached (built once,
            # amortized across repeated reads via _ensure_delta_cache /
            # _ensure_for_cache) — reducing it with sum()/min()/max() in C
            # beats draining it through the iter_valid() generator one
            # Python-level yield at a time.
            cache = None
            if col._delta_mode and not col._delta_fallback:
                cache = col._ensure_delta_cache()
            elif col._for_mode and not col._for_fallback:
                cache = col._ensure_for_cache()
            if cache is not None and len(cache) > 0 and col._nullmask.count(1) == 0:
                if agg == "sum":
                    return sum(cache)
                if agg == "avg":
                    return sum(cache) / len(cache)
                if agg == "min":
                    return min(cache)
                if agg == "max":
                    return max(cache)

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
        live = self._live_mask()
        col_nullmask = col._nullmask

        def _matching_values():
            for idx in range(self._row_count):
                if not live[idx] or col_nullmask[idx]:
                    continue
                row = {name: mat[name][idx] for name in col_names}
                if where(row):
                    yield target[idx]

        if agg == "count":
            # Filtered count: previously the early return above ignored the
            # WHERE predicate entirely while sum/avg/min/max honored it.
            return sum(1 for _ in _matching_values())
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

    def to_dataframe(self):
        """Export the table as a ``pandas.DataFrame`` (returns ``None`` if
        pandas is not installed).

        The optional pandas dependency is imported lazily so the zero-dep
        default is unchanged. Nulls come back as ````NaN`` for numeric columns
        and ````None`` otherwise — which is what pandas expects for object
        columns. Encoded columns (delta / FOR / dict) are decoded on the way
        out via :meth:`Column.tolist`.
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            return None
        data = {col: self.columns[col].tolist() for col in self._col_names}
        return pd.DataFrame(data)

    def compact(self) -> Dict[str, int]:
        """Reclaim space after many deletes.

        The columnar engine uses a per-row liveness bitmap rather than a
        live-rows list, so compaction here is a count-and-report operation —
        the underlying column arrays stay allocated (the live bits are what
        filter them out). For callers that want a structurally-shrunk layout
        we additionally rebuild the columns' value arrays with the dead rows
        dropped, which visibly reduces :meth:`memory_usage` on tables where
        many rows have been deleted. After rebuild, ``self._row_count`` is
        lowered to the live row count so subsequent ``len()`` / scans work
        on the shrunken dataset.

        Returns a dict with ``rows_before``, ``rows_after``, ``rows_removed``,
        and ``bytes_freed`` (estimate of bytes released by the column rebuild;
        ``0`` if the engine skipped the rebuild because nothing was dead).
        """
        live = self._live_mask()
        rows_before = self._row_count
        rows_removed = sum(1 for b in live if not b)
        rows_after = rows_before - rows_removed

        bytes_freed = 0
        if rows_removed > 0 and rows_after >= 0:
            for col in self._col_list:
                rebuilt = _rebuild_column_alive(col, live)
                if rebuilt is not None:
                    bytes_freed += rebuilt
            self._row_count = rows_after

        return {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_removed,
            "bytes_freed": bytes_freed,
        }

    def __repr__(self) -> str:
        return f"ColumnarTable({self.name!r}, rows={self._row_count}, cols={self._col_names})"
