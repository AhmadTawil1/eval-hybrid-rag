import pytest

from app.retrieval.hybrid import rrf_fuse


def test_doc_ranked_high_in_both_lists_wins():
    dense = ["a", "b", "c", "d"]
    bm25 = ["x", "a", "y", "z"]
    fused = rrf_fuse(dense, bm25)
    assert fused[0][0] == "a"


def test_both_lists_beats_single_list_top_rank():
    dense = ["only_dense", "both"]
    bm25 = ["only_bm25", "both"]
    fused = rrf_fuse(dense, bm25)
    assert fused[0][0] == "both"


def test_missing_from_one_list_contributes_zero():
    fused = dict(rrf_fuse(["a"], ["b"]))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 61)


def test_score_formula_with_k_60():
    fused = dict(rrf_fuse(["a", "b"], ["b", "a"]))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
