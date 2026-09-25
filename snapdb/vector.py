"""
SnapDB Vector Index — zero-dependency similarity search for agent memory.

Implements a flat (brute-force) vector index with three distance metrics:
cosine similarity (default), Euclidean (L2), and dot product. Designed for
small-to-medium corpora (≤ ~50k vectors per index) where the simplicity of
a flat scan beats the complexity of HNSW / IVF / ScaNN.

Why flat over HNSW:
    - No build step. Insert one vector, search immediately.
    - Deterministic recall (1.0) — no approximation surprises in tests.
    - Pure Python, zero deps, ~150 LOC.
    - For 1k vectors at dim 384, brute-force cosine is ~5ms on a laptop —
      plenty fast for agent retrieval loops that fire ≤1k times/day.

For larger corpora, swap the underlying storage for `tiny-eval`'s IVF-PQ
or any external FAISS/OpenSearch backend; the API stays identical.

Example:
    >>> idx = VectorIndex(metric="cosine")
    >>> idx.add("doc-1", [0.1, 0.2, 0.3, 0.4])
    >>> idx.add("doc-2", [0.5, 0.1, 0.0, 0.9])
    >>> idx.search([0.1, 0.25, 0.3, 0.4], k=2)
    [("doc-1", 0.9998...), ("doc-2", 0.4xxx...)]
"""

from __future__ import annotations

import math
import threading
from typing import Iterable, List, Sequence, Tuple, Union


__all__ = ["VectorIndex", "cosine_similarity", "euclidean_distance", "dot_product"]


Vector = Sequence[float]
Id = Union[str, int]


# ── Distance / similarity primitives (public, zero-dep, reusable) ────────────


def dot_product(a: Vector, b: Vector) -> float:
    """Inner product of two equal-length vectors.

    Raises:
        ValueError: if dimensions differ.
    """
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    return float(sum(x * y for x, y in zip(a, b)))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity in [-1.0, 1.0]. Zero vectors return 0.0 (not NaN)."""
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Euclidean (L2) distance. Always ≥ 0."""
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


_METRIC_FN = {
    "cosine": cosine_similarity,    # higher = better
    "dot": dot_product,             # higher = better
    "euclidean": euclidean_distance,  # lower = better
}


# ── Flat vector index ───────────────────────────────────────────────────────


class VectorIndex:
    """A flat (brute-force) vector index.

    Stores vectors by id and supports top-k similarity search. Insertion is
    O(1); search is O(n·d). Thread-safe for concurrent reads after bulk
    inserts (uses an internal lock around writes).

    Args:
        metric: One of "cosine", "dot", "euclidean". Default: "cosine".
        normalize_on_add: If True, L2-normalize vectors on ``add`` so dot
            product behaves identically to cosine similarity on unit
            vectors. Useful when callers pass raw model outputs and want
            to skip the per-query norm computation. Default: False.
        dim: Optional expected dimensionality. If set, ``add`` validates
            each vector's length. If None, the first ``add`` sets it.

    Attributes:
        size: Number of vectors currently stored.
        dim: Vector dimensionality (or None until first add).
    """

    def __init__(
        self,
        metric: str = "cosine",
        *,
        normalize_on_add: bool = False,
        dim: int | None = None,
    ) -> None:
        if metric not in _METRIC_FN:
            raise ValueError(f"unknown metric {metric!r}; pick one of {sorted(_METRIC_FN)}")
        self.metric = metric
        self._normalize = normalize_on_add
        self._dim = dim
        self._ids: List[Id] = []
        self._vectors: List[List[float]] = []
        self._lock = threading.RLock()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._ids)

    def __len__(self) -> int:
        return self.size

    @property
    def dim(self) -> int | None:
        return self._dim

    # ── Mutators ────────────────────────────────────────────────────────────

    def add(self, id: Id, vector: Vector) -> None:  # noqa: A002 — id is the public kwarg name
        """Add a vector under ``id``. Duplicate ids REPLACE the prior vector."""
        v = [float(x) for x in vector]
        if self._dim is None:
            self._dim = len(v)
        elif len(v) != self._dim:
            raise ValueError(f"dim mismatch: expected {self._dim}, got {len(v)}")
        if self._normalize:
            v = _l2_normalize(v)
        with self._lock:
            # Replace if id already exists
            for i, existing in enumerate(self._ids):
                if existing == id:
                    self._vectors[i] = v
                    return
            self._ids.append(id)
            self._vectors.append(v)

    def extend(self, items: Iterable[Tuple[Id, Vector]]) -> int:
        """Bulk-add. Returns the number of new vectors inserted."""
        n = 0
        with self._lock:
            for id, vector in items:  # noqa: A001
                self.add(id, vector)
                n += 1
        return n

    def remove(self, id: Id) -> bool:  # noqa: A002
        """Remove a vector by id. Returns True if removed, False if absent."""
        with self._lock:
            for i, existing in enumerate(self._ids):
                if existing == id:
                    del self._ids[i]
                    del self._vectors[i]
                    return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._ids.clear()
            self._vectors.clear()

    # ── Search ──────────────────────────────────────────────────────────────

    def search(self, query: Vector, k: int = 5) -> List[Tuple[Id, float]]:
        """Return top-k (id, score) tuples, sorted by score descending.

        For "cosine" and "dot" metrics: higher score = closer.
        For "euclidean": distances are negated so that higher = closer
        (so a single descending sort works for all three metrics).
        """
        if k <= 0:
            return []
        if self._dim is None:
            return []
        if len(query) != self._dim:
            raise ValueError(f"query dim {len(query)} != index dim {self._dim}")
        q = [float(x) for x in query]
        if self._normalize:
            q = _l2_normalize(q)
        score_fn = _METRIC_FN[self.metric]
        negate = (self.metric == "euclidean")
        with self._lock:
            scored: List[Tuple[Id, float]] = []
            for id, vec in zip(self._ids, self._vectors):  # noqa: A001
                raw = score_fn(q, vec)
                scored.append((id, -raw if negate else raw))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:k]

    # ── Persistence helpers ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (id → vector)."""
        with self._lock:
            return {
                "metric": self.metric,
                "normalize_on_add": self._normalize,
                "dim": self._dim,
                "vectors": [
                    {"id": id, "vector": list(vec)}  # noqa: A001
                    for id, vec in zip(self._ids, self._vectors)
                ],
            }

    @classmethod
    def from_dict(cls, data: dict) -> "VectorIndex":
        """Reconstruct from ``to_dict()`` output."""
        idx = cls(
            metric=data.get("metric", "cosine"),
            normalize_on_add=data.get("normalize_on_add", False),
            dim=data.get("dim"),
        )
        items = [(item["id"], item["vector"]) for item in data.get("vectors", [])]
        idx.extend(items)
        return idx

    def __repr__(self) -> str:
        return (
            f"VectorIndex(metric={self.metric!r}, size={self.size}, "
            f"dim={self._dim}, normalize={self._normalize})"
        )


# ── Internal helpers ─────────────────────────────────────────────────────────


def _l2_normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        return [0.0 for _ in v]
    return [x / n for x in v]
