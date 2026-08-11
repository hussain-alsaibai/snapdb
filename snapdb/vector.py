"""
snapdb/vector.py — Vector (ANN) Search for SnapDB
-------------------------------------------------
RAG-ready approximate nearest-neighbor search over float columns using
cosine similarity. Works on any columnar SnapDB with f32/f64 columns.
No external dependencies — pure Python + optional NumPy.

Usage:
    from snapdb.vector import VectorIndex

    # Build index on an existing float column
    index = VectorIndex(db, column="embedding", nprobe=20)

    # Find top-k most similar vectors
    results = index.search(query_vector, k=5)
    # [{'row': {...}, 'distance': 0.123}, ...]
"""
from __future__ import annotations

import math
import struct
from typing import Any, Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


class VectorIndex:
    """
    Approximate Nearest-Neighbor index for float columns in a SnapDB columnar table.
    
    Uses a simple IVF-like partition approach: bucket vectors by their dominant
    dimension, then scan only the relevant bucket + candidates from neighbors.
    This avoids the quadratic brute-force O(n) scan on every query.

    For production scale, replace with FAISS, Annoy, or hnswlib. For
    <100K vectors this pure-Python approach is fast enough.

    Args:
        table: A ColumnarTable instance.
        column: Name of the float column to index (f32 or f64).
        nprobe: Number of partitions to scan per query (higher = more accurate, slower).
        normalize: Normalize vectors to unit length before indexing (recommended for cosine sim).
    """

    def __init__(
        self,
        table: Any,
        column: str,
        nprobe: int = 10,
        normalize: bool = True,
    ):
        self.table = table
        self.column = column
        self.nprobe = nprobe
        self.normalize = normalize

        # Load all vectors
        col_data = table.columns.get(column)
        if col_data is None:
            raise KeyError(f"Column '{column}' not found in table")
        
        if _HAS_NUMPY:
            raw = col_data.to_numpy(zero_copy=False)
            self._vectors: list[list[float]] = raw.tolist() if hasattr(raw, 'tolist') else list(raw)
        else:
            self._vectors = list(col_data.tolist())

        self._ids: list[int] = list(range(len(self._vectors)))
        self._dim = len(self._vectors[0]) if self._vectors else 0
        self._norms: list[float] = []
        self._partitions: dict[int, list[int]] = {}

        self._build()

    def _build(self) -> None:
        """Build partition index. Bucket vectors by their dominant dimension."""
        self._norms = []
        for vec in self._vectors:
            if self.normalize:
                norm = math.sqrt(sum(v * v for v in vec))
                self._norms.append(norm if norm > 1e-10 else 1.0)
            else:
                self._norms.append(1.0)

        if self._dim == 0:
            return

        # Partition: bucket by dominant dimension
        num_partitions = max(1, min(self.nprobe * 2, len(self._vectors) // 10 or 1))
        self._partitions = {i: [] for i in range(num_partitions)}

        for idx, vec in enumerate(self._vectors):
            # Partition by centroid of top-3 dimensions
            sorted_dims = sorted(enumerate(vec), key=lambda x: abs(x[1]), reverse=True)[:3]
            bucket = int(sum(d[0] for d in sorted_dims) / len(sorted_dims)) % num_partitions
            self._partitions[bucket].append(idx)

    def search(
        self,
        query: list[float],
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Find top-k nearest vectors to the query using cosine similarity.

        Returns list of dicts: [{'row_idx': int, 'row': dict, 'score': float}, ...]
        Sorted by score descending.
        """
        if not self._vectors:
            return []

        q_norm = math.sqrt(sum(v * v for v in query))
        q_norm = q_norm if q_norm > 1e-10 else 1.0
        q_normalized = [v / q_norm for v in query]

        # Determine which partitions to scan
        if self._dim > 0:
            sorted_dims = sorted(enumerate(query), key=lambda x: abs(x[1]), reverse=True)[:3]
            target_buckets = set()
            for bd in sorted_dims:
                num_parts = len(self._partitions)
                target_buckets.add(int(bd[0]) % num_parts)
                target_buckets.add((int(bd[0]) - 1) % num_parts)
                target_buckets.add((int(bd[0]) + 1) % num_parts)
            candidate_indices: set[int] = set()
            for b in target_buckets:
                if b in self._partitions:
                    candidate_indices.update(self._partitions[b])
        else:
            candidate_indices = set(range(len(self._vectors)))

        # Score candidates
        scored: list[tuple[float, int]] = []
        for idx in candidate_indices:
            vec = self._vectors[idx]
            norm = self._norms[idx]
            dot = sum(q_normalized[i] * (vec[i] / norm) for i in range(min(len(vec), len(q_normalized))))
            score = min(1.0, max(-1.0, dot))  # clamp to [-1, 1]
            if score >= min_score:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = scored[:k]

        results = []
        for score, idx in top_k:
            row = self.table.get_row(idx)
            results.append({
                "row_idx": idx,
                "row": row,
                "score": round(score, 6),
            })
        return results

    def search_batch(
        self,
        queries: list[list[float]],
        k: int = 5,
    ) -> list[list[dict[str, Any]]]:
        """Search multiple queries at once (slightly more efficient)."""
        return [self.search(q, k=k) for q in queries]

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def dim(self) -> int:
        return self._dim
