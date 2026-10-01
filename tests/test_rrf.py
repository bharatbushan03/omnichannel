"""
Unit test for the RRF fusion function — the piece most likely to be
implemented wrong (see architecture doc note on why RRF != graph fusion).
"""
from ingestion.hybrid_retrieval import reciprocal_rank_fusion


def test_rrf_favors_items_ranked_high_in_both_lists():
    dense = ["a", "b", "c", "d"]
    sparse = ["b", "a", "d", "c"]

    fused_ids, scores = reciprocal_rank_fusion([dense, sparse])

    # 'a' and 'b' are top-2 in both lists — they should fuse to the top.
    assert set(fused_ids[:2]) == {"a", "b"}


def test_rrf_handles_disjoint_lists():
    dense = ["a", "b"]
    sparse = ["c", "d"]

    fused_ids, scores = reciprocal_rank_fusion([dense, sparse])

    assert set(fused_ids) == {"a", "b", "c", "d"}
    # top-ranked items in either list should outscore bottom-ranked items
    assert scores["a"] == scores["c"]  # both rank 0 in their respective lists


def test_rrf_empty_lists():
    fused_ids, scores = reciprocal_rank_fusion([[], []])
    assert fused_ids == []
    assert scores == {}


def test_rrf_single_list_preserves_order():
    ranked = ["x", "y", "z"]
    fused_ids, _ = reciprocal_rank_fusion([ranked])
    assert fused_ids == ranked
