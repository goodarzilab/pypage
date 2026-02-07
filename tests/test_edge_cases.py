"""Tests for hardened edge cases and behavioral changes."""

import warnings
import numpy as np
import pandas as pd
import pytest
from pypage import ExpressionProfile, GeneSets, PAGE


N_GENES = 200
N_BINS = 5
N_PATHWAYS = 10


def _make_expression():
    genes = np.array([f"g.{i}" for i in range(N_GENES)])
    scores = np.random.normal(size=N_GENES)
    return ExpressionProfile(genes, scores, n_bins=N_BINS)


def _make_genesets():
    genes = np.array([f"g.{i}" for i in range(N_GENES)])
    pathways = np.array([f"p.{i % N_PATHWAYS}" for i in range(N_GENES)])
    return GeneSets(genes, pathways)


# --- Item 1: PAGE.run() always returns a tuple ---

def test_run_returns_tuple_when_no_informative():
    exp = _make_expression()
    gs = _make_genesets()
    p = PAGE(exp, gs, n_shuffle=10, alpha=1e-10, k=1)
    result = p.run()
    assert isinstance(result, tuple) and len(result) == 2


def test_run_returns_tuple_2d():
    genes = np.array([f"g.{i}" for i in range(N_GENES)])
    scores = np.random.normal(size=(3, N_GENES))
    exp = ExpressionProfile(genes, scores, n_bins=N_BINS)
    gs = _make_genesets()
    p = PAGE(exp, gs, n_shuffle=10, k=1)
    result = p.run()
    assert isinstance(result, tuple) and len(result) == 2
    assert result[1] is None


# --- Item 3: Validation raises ValueError, not AssertionError ---

def test_expression_empty_raises_valueerror():
    with pytest.raises(ValueError, match="must not be empty"):
        ExpressionProfile(np.array([]), np.array([]))


def test_expression_shape_mismatch_raises_valueerror():
    with pytest.raises(ValueError, match="equally shaped"):
        ExpressionProfile(np.array(["a", "b"]), np.array([1.0]))


def test_genesets_empty_raises_valueerror():
    with pytest.raises(ValueError, match="must not be empty"):
        GeneSets(np.array([]), np.array([]))


def test_genesets_shape_mismatch_raises_valueerror():
    with pytest.raises(ValueError, match="equal sized"):
        GeneSets(np.array(["a", "b"]), np.array(["p1"]))


# --- Items 2, 6: Missing gene lookup raises ValueError ---

def test_expression_missing_gene_raises_valueerror():
    genes = np.array(["a", "b", "c"])
    scores = np.array([1.0, 2.0, 3.0])
    exp = ExpressionProfile(genes, scores, n_bins=3)
    with pytest.raises(ValueError, match="not found"):
        exp.get_gene_subset(np.array(["a", "z"]))


def test_genesets_missing_gene_raises_valueerror():
    gs = GeneSets(np.array(["a", "b"]), np.array(["p1", "p2"]))
    with pytest.raises(ValueError, match="not found"):
        gs.get_gene_subset(np.array(["a", "z"]))


# --- Item 4: PAGE.__init__ rejects invalid parameters ---

def test_page_invalid_alpha():
    exp = _make_expression()
    gs = _make_genesets()
    with pytest.raises(ValueError, match="alpha"):
        PAGE(exp, gs, alpha=0)
    with pytest.raises(ValueError, match="alpha"):
        PAGE(exp, gs, alpha=1)
    with pytest.raises(ValueError, match="alpha"):
        PAGE(exp, gs, alpha=-0.5)


def test_page_invalid_k():
    exp = _make_expression()
    gs = _make_genesets()
    with pytest.raises(ValueError, match="k must be"):
        PAGE(exp, gs, k=0)


def test_page_invalid_n_shuffle():
    exp = _make_expression()
    gs = _make_genesets()
    with pytest.raises(ValueError, match="n_shuffle"):
        PAGE(exp, gs, n_shuffle=0)


def test_page_invalid_function():
    exp = _make_expression()
    gs = _make_genesets()
    with pytest.raises(ValueError, match="function"):
        PAGE(exp, gs, function="bad")


def test_page_invalid_redundancy_ratio():
    exp = _make_expression()
    gs = _make_genesets()
    with pytest.raises(ValueError, match="redundancy_ratio"):
        PAGE(exp, gs, redundancy_ratio=-1)


# --- Items 2, 3: get_enriched_genes with nonexistent pathway ---

def test_get_enriched_genes_missing_pathway():
    exp = _make_expression()
    gs = _make_genesets()
    p = PAGE(exp, gs, n_shuffle=10, k=1)
    with pytest.raises(ValueError, match="pathway not present"):
        p.get_enriched_genes("nonexistent_pathway")


# --- Item 8: filter_pathways() with no args emits warning ---

def test_filter_pathways_no_args_warns():
    gs = _make_genesets()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gs.filter_pathways()
        assert len(w) == 1
        assert "No minimum or maximum" in str(w[0].message)


def test_filter_pathways_invalid_min_raises():
    gs = _make_genesets()
    with pytest.raises(ValueError, match="minimum must be >= 0"):
        gs.filter_pathways(min_size=-1)


def test_filter_pathways_invalid_max_raises():
    gs = _make_genesets()
    with pytest.raises(ValueError, match="maximum must be > min_size"):
        gs.filter_pathways(min_size=10, max_size=5)
