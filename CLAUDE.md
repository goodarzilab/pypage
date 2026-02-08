# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyPAGE is a Python implementation of the conditional-information PAGE (Pathway Analysis of Gene Expression) framework for gene-set enrichment analysis. It infers differential activity of pathways and regulons while accounting for annotation and membership biases using information-theoretic methods.

**Package name:** `bio-pypage` (on PyPI)
**Python:** 3.8+

## Commands

### Install (development)
```bash
pip install -e .
```

### Run tests
```bash
# Fast tests (excludes slow and network-dependent tests)
pytest -q -m "not slow and not online"

# Full test suite (includes network-dependent tests)
PYPAGE_RUN_ONLINE_TESTS=1 pytest -q

# Single test file
pytest tests/test_run.py -v

# Single test function
pytest tests/test_run.py::test_run -v
```

There is no linter or formatter configured for this project.

## Architecture

### Core Pipeline

The analysis flows through three main classes exposed via `pypage/__init__.py`:

1. **`ExpressionProfile`** (`pypage/io/expression.py`) — Holds gene expression data. Accepts continuous scores (auto-discretized into bins) or pre-binned labels. Lazily computes bin arrays on demand when `get_gene_subset()` is called.

2. **`GeneSets`** (`pypage/io/ontology.py`) — Holds pathway/regulon annotations as a binary membership matrix. Loads from tab-delimited annotation files (supports gzip). Tracks per-gene membership counts (how many pathways each gene belongs to) for bias correction.

3. **`PAGE`** (`pypage/page.py`) — Orchestrates the full analysis via `run()`:
   - Intersects genes between expression and gene-sets
   - Computes MI or CMI for each pathway (CMI conditions on membership to correct annotation bias)
   - Permutation tests to identify statistically significant pathways
   - Optional redundancy filtering via conditional mutual information
   - Hypergeometric enrichment testing per expression bin
   - Returns a results DataFrame and a `Heatmap` object

### Performance Layer

All information-theoretic computations are Numba JIT-compiled (`@nb.jit(nopython=True)`):
- **`pypage/information.py`** — Entropy, MI, CMI, permutation distributions, redundancy measurement. Some functions use `parallel=True`.
- **`pypage/hist.py`** — Fast 1D/2D/3D histogram functions used by the information module.

Changes to these files must stay compatible with Numba's `nopython` mode (no Python objects, no dynamic dispatch).

### Supporting Modules

- **`pypage/utils.py`** — Empirical p-value calculation, hypergeometric tests, Benjamini-Hochberg correction
- **`pypage/heatmap.py`** — Matplotlib-based visualization of enrichment results with optional regulator expression overlay
- **`pypage/io/accession_types.py`** — Gene ID conversion via pybiomart (requires network)

### Key Design Patterns

- **Input reuse:** `ExpressionProfile` and `GeneSets` objects can be reused across multiple `PAGE` runs. PAGE copies intersection data during `__init__`, preventing cross-contamination.
- **CMI as default:** The `function='cmi'` default conditions on membership bins to correct for annotation bias — the core contribution of this method over standard PAGE.
- **Early stopping:** Permutation testing iterates pathways by descending information; stops after `k` consecutive non-significant pathways.

## Test Markers

Defined in `pytest.ini`:
- `@pytest.mark.slow` — Long-running integration tests
- `@pytest.mark.online` — Tests requiring network access (pybiomart)

CI runs fast tests on every push; full suite runs on manual workflow dispatch.
