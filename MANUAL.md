# pyPAGE Manual

This manual is the detailed reference for `pyPAGE`.

## Scope

`pyPAGE` has two analysis modes:
- Bulk PAGE (`PAGE`): pathway enrichment from one ranked or pre-binned gene profile.
- Single-cell PAGE (`SingleCellPAGE`): per-cell pathway scoring + spatial consistency testing on a cell graph.

Core public imports:

```python
from pypage import PAGE, SingleCellPAGE, ExpressionProfile, GeneSets, GeneMapper
```

## Installation

```bash
pip install bio-pypage
```

From source:

```bash
git clone https://github.com/goodarzilab/pyPAGE
cd pyPAGE
pip install -e .
```

## Data Objects

### ExpressionProfile

Represents expression for bulk PAGE input.

```python
ExpressionProfile(genes, expression, is_bin=False, n_bins=10)
```

- `genes`: 1D array of gene names.
- `expression`:
  - 1D continuous scores (`is_bin=False`) or
  - pre-binned/discrete labels (`is_bin=True`), or
  - 2D matrix for advanced MI/CMI matrix use.
- `is_bin=False`: equal-frequency discretization into `n_bins`.
- `is_bin=True`: preserves discrete labels by encoding categories.
- Missing values are dropped from input vectors.

Useful attributes after construction/subsetting:
- `genes`, `raw_expression`, `n_genes`, `n_bins`
- `bin_edges` (continuous mode)
- `bin_labels` (discrete mode or generated labels)

Legacy ID conversion (network-dependent via BioMart):

```python
exp.convert_from_to("refseq", "ensg", "human")
```

### GeneSets

Represents pathway membership.

```python
GeneSets(genes=..., pathways=..., ann_file=None, n_bins=3, first_col_is_genes=False)
```

Construction paths:
- Long format arrays: `genes` + `pathways` (same length).
- Index file: `ann_file` where each line is either:
  - `pathway<TAB>gene1<TAB>gene2...` (default), or
  - `gene<TAB>pathway1<TAB>pathway2...` with `first_col_is_genes=True`.

GMT support:

```python
gs = GeneSets.from_gmt("pathways.gmt", min_size=15, max_size=500)
gs.to_gmt("filtered_pathways.gmt")
```

Key attributes:
- `genes`, `pathways`
- `bool_array` (pathway x gene membership matrix)
- `membership` (gene membership counts across pathways)

### GeneMapper (recommended for ID conversion)

Offline conversion with local cache (`~/.pypage/gene_map_<species>.tsv`).

```python
mapper = GeneMapper(species="human")
converted, unmapped = mapper.convert(ids, from_type="ensg", to_type="symbol")
```

Supported ID types: `ensg`, `symbol`, `entrez`.

Map full gene sets in-place:

```python
gs.map_genes(mapper, from_type="entrez", to_type="symbol")
```

`GeneMapper` downloads mapping on first run (via BioMart), then works from cache.

## Bulk PAGE API

### Minimal run

```python
import pandas as pd
from pypage import PAGE, ExpressionProfile, GeneSets

expr = pd.read_csv("example_data/test_DESeq_logFC.txt", sep="\t")
exp = ExpressionProfile(expr["GENE"], expr["log2FoldChange"], is_bin=False, n_bins=9)
gs = GeneSets.from_gmt("example_data/h.all.v2026.1.Hs.symbols.gmt")

p = PAGE(
    exp,
    gs,
    function="cmi",          # "cmi" or "mi"
    n_shuffle=10000,
    alpha=0.005,
    k=20,
    filter_redundant=True,
    redundancy_ratio=5.0,
    n_jobs=1,
)

results, heatmap = p.run()
```

Returned:
- `results` DataFrame columns:
  - `pathway`, `CMI`, `z-score`, `p-value`, `Regulation pattern`
- `heatmap`: `Heatmap` object or `None` if no informative pathways.

### Manual pathway mode

Bypasses permutation significance and redundancy filtering.

```python
manual_results, manual_hm = p.run_manual([
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_E2F_TARGETS",
])
```

In manual mode, `p-value` and `z-score` are `NaN`.

### Redundancy and introspection

```python
killed = p.get_redundancy_log()
full = p.full_results
es_matrix = p.get_es_matrix()
```

- `get_redundancy_log()` columns: `rejected_pathway`, `killed_by`, `min_ratio`
- `full_results` includes informative pathways before final redundancy exclusion with a `redundant` flag.
- `get_es_matrix()` returns pathway x bin enrichment scores used for plotting.

### Per-pathway enriched genes

```python
enriched_bins = p.get_enriched_genes("HALLMARK_MYC_TARGETS_V1")
```

Returns a list of gene arrays by expression bin.

## Heatmap Object

The bulk run heatmap object supports static, interactive, and matrix-based workflows.

```python
heatmap.save("heatmap.pdf", max_rows=50, max_val=5.0, min_val=-5.0)
heatmap.to_html("heatmap.html", max_rows=50)
heatmap.save_matrix("results.matrix.tsv")
```

Load later for draw-only rendering:

```python
from pypage.heatmap import Heatmap
hm = Heatmap.from_matrix("results.matrix.tsv")
hm.save("redrawn.pdf", max_rows=80)
```

Notes:
- `max_rows=-1` shows all pathways.
- `show_reg=True` shows regulator column when regulator expression is available.
- `bar_min`/`bar_max` control normalization of the continuous-mode bin-edge bar.

## Bulk CLI (`pypage`)

### Full run

```bash
pypage -e expression.tsv --gmt pathways.gmt --type continuous
```

Required for full run:
- `-e/--expression`
- one of `-g/--genesets`, `--genesets-long`, or `--gmt`

Common options:
- `--type continuous|discrete` (`--is-bin` is a legacy alias for discrete)
- `--cols GENE_COL,SCORE_COL` (names or 1-based indices)
- `--no-header`
- `--n-bins`
- `--function mi|cmi`
- `--n-shuffle`, `--alpha`, `-k`
- `--filter-redundant` / `--no-filter-redundant`
- `--redundancy-ratio`
- `--manual` (comma-separated names or file)
- visualization controls: `--max-rows`, `--min-val`, `--max-val`, `--bar-min`, `--bar-max`, `--cmap`, `--cmap-reg`, `--show-reg`, `--title`

### Draw-only mode

Re-render from matrix without recomputing enrichment:

```bash
pypage --draw-only --outdir my_run
# or
pypage --draw-only --matrix my_run/tables/results.matrix.tsv
```

### Resume mode

```bash
pypage --resume -e expression.tsv --gmt pathways.gmt --type continuous
```

`--resume` attempts matrix-based redraw first; if matrix is missing it falls back to full analysis.

### Default output layout

For default `outdir={expression_stem}_PAGE/`:
- `tables/results.tsv`
- `tables/results.matrix.tsv`
- `tables/results.killed.tsv`
- `plots/heatmap.pdf`
- `heatmap.html`
- `run/command.txt`
- `run/status.json`

## Single-Cell PAGE API

### Construct from AnnData (recommended)

```python
import anndata
from pypage import GeneSets, SingleCellPAGE

adata = anndata.read_h5ad("example_data/CRC.h5ad")
gs = GeneSets.from_gmt("example_data/c2.all.v2026.1.Hs.symbols.gmt")

sc = SingleCellPAGE(
    adata=adata,
    genesets=gs,
    function="cmi",          # "mi" or "cmi"
    n_bins=10,
    bin_axis="cell",         # "cell" or "gene"
    n_neighbors=None,         # defaults to ceil(sqrt(n_cells)), capped at 100
    fast_mode=False,
    n_jobs=1,
    filter_redundant=True,
    redundancy_ratio=5.0,
    redundancy_scope="fdr",  # "fdr" or "all"
    redundancy_fdr=0.05,
)

results = sc.run(n_permutations=1000)
```

### Construct from arrays

```python
sc = SingleCellPAGE(
    expression=X,       # shape (n_cells, n_genes)
    genes=gene_names,
    genesets=gs,
    connectivity=W,     # optional sparse graph
)
```

### Results and outputs

`sc.run(...)` returns DataFrame columns:
- `pathway`, `consistency`, `p-value`, `FDR`

Related attributes:
- `sc.scores`: per-cell pathway score matrix
- `sc.full_results`: includes `redundant` flag
- `sc.results`: filtered results (if redundancy filtering enabled)

### Manual mode

```python
manual = sc.run_manual(["REACTOME_M_PHASE", "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"])
```

Manual mode returns `p-value=NaN`, `FDR=NaN`.

### Redundancy and neighborhood analysis

```python
killed = sc.get_redundancy_log()
summary, group_results = sc.run_neighborhoods(labels=adata.obs["leiden"])
```

`run_neighborhoods` runs standard bulk PAGE on group-level pseudo-bulk profiles.

### Plot helpers

```python
sc.plot_consistency_ranking(top_n=30, fdr_threshold=0.05)
sc.plot_pathway_on_embedding("REACTOME_M_PHASE", embedding_key="X_umap")
sc.plot_pathway_heatmap(labels=adata.obs["leiden"])
```

## Single-Cell CLI (`pypage-sc`)

### Full run

```bash
pypage-sc --adata data.h5ad --gmt pathways.gmt
```

Required for full run:
- one of `--adata` or `--expression`
- one of `-g/--genesets`, `--genesets-long`, or `--gmt`
- `--genes` is required when using `--expression`

Common analysis options:
- `--function mi|cmi`
- `--n-bins`
- `--bin-axis cell|gene`
- `--n-neighbors`
- `--n-permutations`
- `--perm-chunk-size`
- `--score-chunk-size`
- `--fast-mode`
- `--filter-redundant` / `--no-filter-redundant`
- `--redundancy-ratio`, `--redundancy-scope`, `--redundancy-fdr`
- `--manual`

Visualization/report options:
- `--top-n`, `--fdr-threshold`, `--title`
- `--umap-top-n`, `--embedding-key`, `--sc-cmap`
- `--groupby`, `--group-enrichment-top-n`, `--no-group-enrichment`
- `--report`, `--no-report`, `--report-vmin`, `--report-vmax`
- `--ranking-pdf`, `--ranking-html`
- `--scores`
- `--no-save-adata`

### Draw-only mode

```bash
pypage-sc --draw-only --outdir data_scPAGE
```

Draw-only regenerates ranking artifacts (and report/UMAP/group artifacts when an annotated `adata.h5ad` is available).

### Resume mode

```bash
pypage-sc --adata data.h5ad --gmt pathways.gmt --resume
```

`--resume` can skip analysis and reuse prior outputs when input signatures and parameter signatures match a completed run manifest.

### Default output layout

For default `outdir={input_stem}_scPAGE/`:
- `tables/results.tsv`
- `tables/results.killed.tsv`
- `plots/ranking.pdf`
- `plots/ranking.html`
- `plots/umap_plots/*.pdf`
- `plots/group_enrichment/*.pdf`
- `plots/group_enrichment/sc_group_enrichment_stats.tsv`
- `sc_report.html`
- `adata.h5ad` (unless disabled)
- `run/command.txt`
- `run/command.json`
- `run/manifest.json`
- `run/artifacts.json`

## Reproducibility Guidance

- Use `--seed` for CLI runs.
- For Python API runs, set NumPy seed before constructing profiles:

```python
import numpy as np
np.random.seed(42)
```

- Use `n_jobs=1` for stricter run-to-run reproducibility.
- Keep gene IDs consistent between expression and gene-set inputs before running PAGE.

## Notebook Guide

Refreshed notebooks in `notebooks/`:
- `pyPAGE_tutorial.ipynb`: end-to-end bulk + single-cell walkthrough.
- `bulk_page_tutorial.ipynb`: focused bulk workflow.
- `sc_page_tutorial.ipynb`: CRC AnnData + `pypage-sc` style outputs.
- `single_cell_page_tutorial.ipynb`: synthetic single-cell walkthrough.
