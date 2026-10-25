"""Tests for snapdb VectorIndex (v0.17.0)."""

from __future__ import annotations

import math

import pytest

from snapdb import VectorIndex, cosine_similarity, dot_product, euclidean_distance


# ── Primitive distance functions ────────────────────────────────────────────


def test_cosine_identical_vectors():
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    # 0 / 0 — must return 0.0, not NaN.
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_dim_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_dot_product_basic():
    assert dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)


def test_dot_product_dim_mismatch_raises():
    with pytest.raises(ValueError):
        dot_product([1.0], [1.0, 2.0])


def test_euclidean_basic():
    assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_euclidean_zero_distance():
    assert euclidean_distance([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_euclidean_dim_mismatch_raises():
    with pytest.raises(ValueError):
        euclidean_distance([1.0, 2.0], [1.0])


# ── VectorIndex construction & validation ──────────────────────────────────


def test_construction_defaults():
    idx = VectorIndex()
    assert idx.metric == "cosine"
    assert idx.size == 0
    assert idx.dim is None
    assert len(idx) == 0


def test_construction_unknown_metric_raises():
    with pytest.raises(ValueError):
        VectorIndex(metric="manhattan")


def test_construction_with_explicit_dim():
    idx = VectorIndex(dim=4)
    assert idx.dim == 4


def test_add_sets_dim_from_first_vector():
    idx = VectorIndex()
    idx.add("x", [1.0, 2.0, 3.0])
    assert idx.dim == 3


def test_add_dim_mismatch_raises_after_first():
    idx = VectorIndex()
    idx.add("x", [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        idx.add("y", [1.0, 2.0])


def test_add_with_explicit_dim_validates():
    idx = VectorIndex(dim=3)
    with pytest.raises(ValueError):
        idx.add("x", [1.0, 2.0])


# ── Search semantics ────────────────────────────────────────────────────────


def test_search_cosine_returns_top_k_descending():
    idx = VectorIndex(metric="cosine")
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.9, 0.1, 0.0])
    idx.add("c", [0.0, 1.0, 0.0])
    results = idx.search([1.0, 0.0, 0.0], k=3)
    assert [r[0] for r in results] == ["a", "b", "c"]
    assert all(r[1] >= results[-1][1] for r in results)


def test_search_euclidean_returns_closest_first():
    idx = VectorIndex(metric="euclidean")
    idx.add("near", [1.0, 0.0])
    idx.add("mid", [2.0, 0.0])
    idx.add("far", [10.0, 0.0])
    results = idx.search([1.0, 0.0], k=3)
    assert [r[0] for r in results] == ["near", "mid", "far"]


def test_search_dot_higher_first():
    idx = VectorIndex(metric="dot")
    idx.add("a", [1.0, 1.0, 1.0])
    idx.add("b", [0.1, 0.1, 0.1])
    results = idx.search([1.0, 1.0, 1.0], k=2)
    assert results[0][0] == "a"
    assert results[0][1] > results[1][1]


def test_search_k_zero_returns_empty():
    idx = VectorIndex()
    idx.add("x", [1.0, 0.0])
    assert idx.search([1.0, 0.0], k=0) == []


def test_search_k_larger_than_index_returns_all():
    idx = VectorIndex()
    idx.add("x", [1.0, 0.0])
    idx.add("y", [0.0, 1.0])
    results = idx.search([1.0, 0.0], k=10)
    assert len(results) == 2


def test_search_empty_index_returns_empty():
    idx = VectorIndex()
    assert idx.search([1.0, 0.0]) == []


def test_search_query_dim_mismatch_raises():
    idx = VectorIndex()
    idx.add("x", [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        idx.search([1.0, 0.0], k=1)


def test_search_zero_vector_query():
    # cosine: zero query → all scores are 0.0
    idx = VectorIndex(metric="cosine")
    idx.add("x", [1.0, 0.0, 0.0])
    results = idx.search([0.0, 0.0, 0.0], k=1)
    assert results[0] == ("x", 0.0)


# ── Mutation semantics ──────────────────────────────────────────────────────


def test_add_replaces_existing_id():
    idx = VectorIndex()
    idx.add("x", [1.0, 0.0])
    idx.add("x", [0.0, 1.0])  # replace
    assert idx.size == 1
    results = idx.search([0.0, 1.0], k=1)
    assert results[0][0] == "x"


def test_extend_bulk_add():
    idx = VectorIndex()
    n = idx.extend([("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [1.0, 1.0])])
    assert n == 3
    assert idx.size == 3


def test_remove_existing_returns_true():
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0])
    assert idx.remove("a") is True
    assert idx.size == 0


def test_remove_missing_returns_false():
    idx = VectorIndex()
    assert idx.remove("nope") is False


def test_clear_resets_index():
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0])
    idx.add("b", [0.0, 1.0])
    idx.clear()
    assert idx.size == 0


def test_accepts_int_ids():
    idx = VectorIndex()
    idx.add(1, [1.0, 0.0])
    idx.add(2, [0.0, 1.0])
    results = idx.search([1.0, 0.0], k=2)
    assert 1 in [r[0] for r in results]


# ── normalize_on_add ────────────────────────────────────────────────────────


def test_normalize_on_add_does_not_change_cosine_search():
    idx_n = VectorIndex(metric="cosine", normalize_on_add=True)
    idx = VectorIndex(metric="cosine")
    vec = [1.0, 2.0, 3.0]
    idx_n.add("a", vec)
    idx.add("a", vec)
    # When queries are also unit-normalized, both should rank the same.
    q = [4.0, 5.0, 6.0]
    q_n = [4.0 / math.sqrt(77), 5.0 / math.sqrt(77), 6.0 / math.sqrt(77)]
    r1 = idx_n.search(q_n, k=1)[0][1]
    r2 = idx.search(q, k=1)[0][1]
    assert r1 == pytest.approx(r2, rel=1e-6)


# ── Serialization round-trip ────────────────────────────────────────────────


def test_to_from_dict_round_trip():
    idx = VectorIndex(metric="cosine")
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.0, 1.0, 0.0])
    idx.add("c", [0.0, 0.0, 1.0])
    blob = idx.to_dict()

    restored = VectorIndex.from_dict(blob)
    assert restored.size == 3
    assert restored.dim == 3
    assert restored.metric == "cosine"

    # Search results must match exactly.
    q = [1.0, 0.01, 0.0]
    a = idx.search(q, k=2)
    b = restored.search(q, k=2)
    assert [r[0] for r in a] == [r[0] for r in b]


def test_to_dict_empty_index_round_trips():
    idx = VectorIndex(metric="euclidean")
    blob = idx.to_dict()
    restored = VectorIndex.from_dict(blob)
    assert restored.size == 0
    assert restored.metric == "euclidean"


# ── Concurrency (best-effort, smoke test) ──────────────────────────────────


def test_concurrent_adds():
    import threading

    idx = VectorIndex()
    errors = []

    def worker(start):
        try:
            for i in range(100):
                idx.add(f"id-{start}-{i}", [float(i), 0.0, 0.0])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert idx.size == 400


# ── Real-world shape: agent memory retrieval ────────────────────────────────


def test_agent_memory_retrieval_pattern():
    """Simulate a small agent-memory store and verify nearest-neighbor
    retrieval finds the most semantically similar prior memory."""
    idx = VectorIndex(metric="cosine", dim=4)
    # Imagine these are 4-dim embeddings for past agent memories.
    memories = {
        "user-likes-coffee": [0.9, 0.1, 0.0, 0.0],
        "user-likes-tea":    [0.85, 0.15, 0.0, 0.0],
        "user-likes-python": [0.0, 0.9, 0.1, 0.0],
        "user-likes-rust":   [0.0, 0.85, 0.15, 0.0],
        "weather-rainy":     [0.1, 0.1, 0.9, 0.0],
        "weather-sunny":     [0.1, 0.1, 0.0, 0.9],
    }
    for k, v in memories.items():
        idx.add(k, v)

    # Query: "user likes coffee" embedding.
    query = [0.92, 0.08, 0.0, 0.0]
    results = idx.search(query, k=2)
    assert results[0][0] == "user-likes-coffee"
    assert results[1][0] == "user-likes-tea"


def test_repr_does_not_crash():
    idx = VectorIndex()
    idx.add("a", [1.0])
    assert "VectorIndex" in repr(idx)
