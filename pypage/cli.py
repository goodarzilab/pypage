"""Command-line interface for pyPAGE.

Usage::

    pypage --expression expr.tab.gz --genesets annotations.txt.gz [options]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from .io import ExpressionProfile, GeneSets
from .page import PAGE
from .heatmap import Heatmap


def _parse_manual(value):
    """Parse --manual value as either a file path or comma-separated string."""
    import os
    if os.path.isfile(value):
        with open(value) as f:
            return [line.strip() for line in f if line.strip()]
    return [s.strip() for s in value.split(",")]


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="pypage",
        description="pyPAGE: Pathway Analysis of Gene Expression",
    )

    # -- Input files ----------------------------------------------------------
    parser.add_argument(
        "-e", "--expression", required=False, default=None,
        help="Expression file (tab-delimited: gene <TAB> score/bin)",
    )

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

    # -- Expression options ---------------------------------------------------
    parser.add_argument(
        "--is-bin", action="store_true", default=False,
        help="Treat expression values as pre-binned integer labels",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of bins for continuous expression (default: 10)",
    )

    # -- PAGE parameters ------------------------------------------------------
    parser.add_argument(
        "--function", choices=["mi", "cmi"], default="cmi",
        help="Information function: 'mi' or 'cmi' (default: cmi)",
    )
    parser.add_argument(
        "--n-shuffle", type=int, default=10000,
        help="Number of permutations (default: 10000)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.005,
        help="P-value threshold (default: 0.005)",
    )
    parser.add_argument(
        "-k", type=int, default=20,
        help="Early-stopping consecutive failures (default: 20)",
    )
    parser.add_argument(
        "--filter-redundant", action="store_true", dest="filter_redundant",
        default=True,
        help="Enable redundancy filtering (default)",
    )
    parser.add_argument(
        "--no-filter-redundant", action="store_false", dest="filter_redundant",
        help="Disable redundancy filtering",
    )
    parser.add_argument(
        "--redundancy-ratio", type=float, default=5.0,
        help="CMI/MI ratio threshold for redundancy (default: 5.0)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel threads (default: 1)",
    )

    # -- Output ---------------------------------------------------------------
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output TSV path (default: stdout)",
    )
    parser.add_argument(
        "--heatmap", default=None,
        help="Save heatmap image to path (PNG/PDF/SVG)",
    )
    parser.add_argument(
        "--manual", default=None,
        help="Pathway names: comma-separated or a file (one per line). Bypasses significance testing.",
    )
    parser.add_argument(
        "--killed", default=None,
        help="Save redundancy log to path (TSV)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )

    # -- Visualization options ------------------------------------------------
    parser.add_argument(
        "--html", default=None,
        help="Save heatmap as standalone HTML file",
    )
    parser.add_argument(
        "--cmap", default="viridis",
        help="Colormap for enrichment heatmap (default: viridis)",
    )
    parser.add_argument(
        "--cmap-reg", default="plasma",
        help="Colormap for regulator column (default: plasma)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=50,
        help="Max pathways displayed (default: 50, -1 for all)",
    )
    parser.add_argument(
        "--max-val", type=int, default=5,
        help="Color scale abs cap (default: 5)",
    )
    parser.add_argument(
        "--title", default="",
        help="Plot title",
    )
    parser.add_argument(
        "--show-reg", action="store_true", default=False,
        help="Show regulator expression column",
    )

    # -- Draw-only mode -------------------------------------------------------
    parser.add_argument(
        "--draw-only", action="store_true", default=False,
        help="Skip analysis, load saved matrix for visualization only",
    )
    parser.add_argument(
        "--matrix", default=None,
        help="Path to .matrix.tsv (required with --draw-only)",
    )

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # -- Validate conditional requirements ------------------------------------
    if args.draw_only:
        if args.matrix is None:
            parser.error("--matrix is required when using --draw-only")
    else:
        if args.expression is None:
            parser.error("-e/--expression is required when not using --draw-only")
        if args.genesets is None and args.genesets_long is None and args.gmt is None:
            parser.error("one of -g/--genesets, --genesets-long, --gmt is required when not using --draw-only")

    # -- Seed -----------------------------------------------------------------
    if args.seed is not None:
        np.random.seed(args.seed)

    # -- Draw-only mode -------------------------------------------------------
    if args.draw_only:
        heatmap = Heatmap.from_matrix(args.matrix)
        heatmap.cmap_main = args.cmap
        heatmap.cmap_reg = args.cmap_reg

        # Optionally load expression for regulator overlay
        if args.expression and args.show_reg:
            expr_df = pd.read_csv(
                args.expression, sep="\t", header=None, names=["gene", "score"],
            )
            heatmap.add_gene_expression(
                np.array(expr_df["gene"]), np.array(expr_df["score"]))

        if args.heatmap is not None:
            heatmap.save(args.heatmap, max_rows=args.max_rows,
                         show_reg=args.show_reg, max_val=args.max_val,
                         title=args.title)
        if args.html is not None:
            heatmap.to_html(args.html, max_rows=args.max_rows,
                            show_reg=args.show_reg, max_val=args.max_val,
                            title=args.title)
        return

    # -- Load expression ------------------------------------------------------
    expr_df = pd.read_csv(
        args.expression, sep="\t", header=None, names=["gene", "score"],
    )
    exp = ExpressionProfile(
        expr_df["gene"],
        expr_df["score"],
        is_bin=args.is_bin,
        n_bins=args.n_bins,
    )

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

    # -- Run PAGE -------------------------------------------------------------
    p = PAGE(
        exp, gs,
        function=args.function,
        n_shuffle=args.n_shuffle,
        alpha=args.alpha,
        k=args.k,
        filter_redundant=args.filter_redundant,
        redundancy_ratio=args.redundancy_ratio,
        n_jobs=args.n_jobs,
    )

    if args.manual is not None:
        pathway_names = _parse_manual(args.manual)
        results, heatmap = p.run_manual(pathway_names)
    else:
        results, heatmap = p.run()

    # -- Write results --------------------------------------------------------
    if args.output is not None:
        results.to_csv(args.output, sep="\t", index=False)
        # Auto-save companion matrix
        if heatmap is not None:
            if '.' in args.output:
                matrix_path = args.output.rsplit('.', 1)[0] + '.matrix.tsv'
            else:
                matrix_path = args.output + '.matrix.tsv'
            heatmap.save_matrix(matrix_path)
            print(f"Enrichment matrix saved to {matrix_path}", file=sys.stderr)
    else:
        results.to_csv(sys.stdout, sep="\t", index=False)

    # -- Apply viz params and save heatmap ------------------------------------
    if heatmap is not None:
        heatmap.cmap_main = args.cmap
        heatmap.cmap_reg = args.cmap_reg

    if args.heatmap is not None and heatmap is not None:
        heatmap.save(args.heatmap, max_rows=args.max_rows,
                     show_reg=args.show_reg, max_val=args.max_val,
                     title=args.title)

    if args.html is not None and heatmap is not None:
        heatmap.to_html(args.html, max_rows=args.max_rows,
                        show_reg=args.show_reg, max_val=args.max_val,
                        title=args.title)

    # -- Redundancy log -------------------------------------------------------
    if args.killed is not None:
        killed_df = p.get_redundancy_log()
        killed_df.to_csv(args.killed, sep="\t", index=False)


if __name__ == "__main__":
    main()
