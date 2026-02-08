"""Command-line interface for single-cell PAGE analysis.

Usage::

    pypage-sc --adata data.h5ad --gmt pathways.gmt [options]
    pypage-sc --expression matrix.tsv --genes genes.txt --genesets-long ann.txt.gz [options]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from .io import GeneSets
from .sc import SingleCellPAGE
from .cli import _parse_manual
from .plotting import plot_consistency_ranking, consistency_ranking_to_html


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="pypage-sc",
        description="pyPAGE single-cell: per-cell pathway scoring with spatial coherence testing",
    )

    # -- Input (expression) ---------------------------------------------------
    expr_group = parser.add_mutually_exclusive_group(required=False)
    expr_group.add_argument(
        "--adata",
        help="H5AD file (AnnData format)",
    )
    expr_group.add_argument(
        "--expression",
        help="Tab-delimited expression matrix (cells x genes, no header)",
    )

    parser.add_argument(
        "--genes",
        help="Gene names file (one per line). Required with --expression.",
    )

    # -- Gene sets ------------------------------------------------------------
    gs_group = parser.add_mutually_exclusive_group(required=False)
    gs_group.add_argument(
        "-g", "--genesets",
        help="Gene set index file (pathway<TAB>gene1<TAB>gene2..., supports .gz)",
    )
    gs_group.add_argument(
        "--genesets-long",
        help="Gene set long-format file (gene<TAB>pathway per line, supports .gz)",
    )
    gs_group.add_argument(
        "--gmt",
        help="Gene set GMT file (.gmt or .gmt.gz)",
    )

    # -- SC parameters --------------------------------------------------------
    parser.add_argument(
        "--function", choices=["mi", "cmi"], default="cmi",
        help="Information function: 'mi' or 'cmi' (default: cmi)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of bins for expression discretization (default: 10)",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=None,
        help="Number of KNN neighbors (default: ceil(sqrt(n_cells)), capped at 100)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for significance testing (default: 1000)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel threads (default: 1)",
    )

    # -- Manual mode ----------------------------------------------------------
    parser.add_argument(
        "--manual", default=None,
        help="Pathway names: comma-separated or a file (one per line). Bypasses significance testing.",
    )

    # -- Output ---------------------------------------------------------------
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output TSV path (default: stdout)",
    )
    parser.add_argument(
        "--scores", default=None,
        help="Save per-cell scores matrix as TSV (cells x pathways, with header)",
    )

    # -- Visualization options ------------------------------------------------
    parser.add_argument(
        "--heatmap", default=None,
        help="Save consistency ranking plot (PNG/PDF/SVG)",
    )
    parser.add_argument(
        "--html", default=None,
        help="Save consistency ranking as standalone HTML",
    )
    parser.add_argument(
        "--top-n", type=int, default=30,
        help="Top pathways to display (default: 30)",
    )
    parser.add_argument(
        "--fdr-threshold", type=float, default=0.05,
        help="FDR threshold for significance coloring (default: 0.05)",
    )
    parser.add_argument(
        "--title", default="",
        help="Plot title",
    )

    # -- Draw-only mode -------------------------------------------------------
    parser.add_argument(
        "--draw-only", action="store_true", default=False,
        help="Skip analysis, load saved results for visualization only",
    )
    parser.add_argument(
        "--results", default=None,
        help="Path to results TSV (required with --draw-only)",
    )

    # -- General --------------------------------------------------------------
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # -- Validate conditional requirements ------------------------------------
    if args.draw_only:
        if args.results is None:
            parser.error("--results is required when using --draw-only")
    else:
        if args.adata is None and args.expression is None:
            parser.error("one of --adata, --expression is required when not using --draw-only")
        if args.genesets is None and args.genesets_long is None and args.gmt is None:
            parser.error("one of -g/--genesets, --genesets-long, --gmt is required when not using --draw-only")

    if args.expression is not None and args.genes is None:
        parser.error("--genes is required when using --expression")

    # -- Seed -----------------------------------------------------------------
    if args.seed is not None:
        np.random.seed(args.seed)

    # -- Draw-only mode -------------------------------------------------------
    if args.draw_only:
        results = pd.read_csv(args.results, sep='\t')
        if args.heatmap is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            ax = plot_consistency_ranking(
                results, top_n=args.top_n, fdr_threshold=args.fdr_threshold)
            if args.title:
                ax.set_title(args.title)
            ax.figure.savefig(args.heatmap, bbox_inches='tight')
            plt.close(ax.figure)
        if args.html is not None:
            consistency_ranking_to_html(
                results, args.html, top_n=args.top_n,
                fdr_threshold=args.fdr_threshold, title=args.title)
        return

    # -- Load expression ------------------------------------------------------
    adata = None
    expression = None
    genes = None

    if args.adata is not None:
        import anndata
        adata = anndata.read_h5ad(args.adata)
    else:
        expression = np.loadtxt(args.expression, delimiter="\t")
        with open(args.genes) as f:
            genes = np.array([line.strip() for line in f if line.strip()])

    # -- Load gene sets -------------------------------------------------------
    if args.genesets is not None:
        gs = GeneSets(ann_file=args.genesets)
    elif args.genesets_long is not None:
        ann = pd.read_csv(
            args.genesets_long, sep="\t", header=None,
            names=["gene", "pathway"],
        )
        gs = GeneSets(ann["gene"], ann["pathway"])
    else:
        gs = GeneSets.from_gmt(args.gmt)

    # -- Create SingleCellPAGE ------------------------------------------------
    sc = SingleCellPAGE(
        adata=adata,
        expression=expression,
        genes=genes,
        genesets=gs,
        n_neighbors=args.n_neighbors,
        n_bins=args.n_bins,
        function=args.function,
        n_jobs=args.n_jobs,
    )

    # -- Run ------------------------------------------------------------------
    if args.manual is not None:
        pathway_names = _parse_manual(args.manual)
        results = sc.run_manual(pathway_names)
    else:
        results = sc.run(n_permutations=args.n_permutations)

    # -- Write results --------------------------------------------------------
    if args.output is not None:
        results.to_csv(args.output, sep="\t", index=False)
    else:
        results.to_csv(sys.stdout, sep="\t", index=False)

    # -- Write per-cell scores ------------------------------------------------
    if args.scores is not None:
        if args.manual is not None:
            pathway_names_out = pathway_names
        else:
            pathway_names_out = list(sc.pathway_names)
        scores_df = pd.DataFrame(sc.scores, columns=pathway_names_out)
        scores_df.to_csv(args.scores, sep="\t", index=False)

    # -- Visualization --------------------------------------------------------
    if args.heatmap is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ax = plot_consistency_ranking(
            results, top_n=args.top_n, fdr_threshold=args.fdr_threshold)
        if args.title:
            ax.set_title(args.title)
        ax.figure.savefig(args.heatmap, bbox_inches='tight')
        plt.close(ax.figure)

    if args.html is not None:
        consistency_ranking_to_html(
            results, args.html, top_n=args.top_n,
            fdr_threshold=args.fdr_threshold, title=args.title)


if __name__ == "__main__":
    main()
