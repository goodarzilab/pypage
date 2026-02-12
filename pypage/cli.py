"""Command-line interface for pyPAGE.

Usage::

    pypage --expression expr.tab.gz --genesets annotations.txt.gz [options]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .io import ExpressionProfile, GeneSets
from .page import PAGE
from .heatmap import Heatmap


def _parse_manual(value):
    """Parse --manual value as either a file path or comma-separated string."""
    if os.path.isfile(value):
        with open(value) as f:
            return [line.strip() for line in f if line.strip()]
    return [s.strip() for s in value.split(",")]


def _load_expression(path, cols=None, no_header=False, is_bin=False, n_bins=10):
    """Load an expression file with flexible column selection.

    Parameters
    ----------
    path : str
        Path to tab-delimited expression file.
    cols : str or None
        Comma-separated pair of column names or 1-based numbers.
    no_header : bool
        If True, file has no header row.
    is_bin : bool
        Treat values as pre-binned labels.
    n_bins : int
        Number of bins for continuous expression.
    """
    if no_header:
        df = pd.read_csv(path, sep="\t", header=None)
    else:
        df = pd.read_csv(path, sep="\t")

    if cols is not None:
        parts = [c.strip() for c in cols.split(",")]
        if len(parts) != 2:
            raise SystemExit("--cols requires exactly two values: GENE_COL,SCORE_COL")
        if parts[0].isdigit() and parts[1].isdigit():
            idx0, idx1 = int(parts[0]) - 1, int(parts[1]) - 1
            if idx0 < 0 or idx0 >= len(df.columns) or idx1 < 0 or idx1 >= len(df.columns):
                raise SystemExit(
                    f"Column index out of range. File has {len(df.columns)} columns (1-{len(df.columns)})")
            gene_col = df.columns[idx0]
            score_col = df.columns[idx1]
        else:
            if no_header:
                raise SystemExit("--cols with column names requires a header row (remove --no-header)")
            gene_col, score_col = parts
            if gene_col not in df.columns or score_col not in df.columns:
                available = ", ".join(str(c) for c in df.columns)
                raise SystemExit(
                    f"Column(s) not found. Available: {available}\n"
                    f"Requested: {gene_col}, {score_col}")
    else:
        gene_col = df.columns[0]
        score_col = df.columns[1]

    return ExpressionProfile(df[gene_col], df[score_col], is_bin=is_bin, n_bins=n_bins)


def _resolve_expression_input_mode(args, parser):
    """Resolve expression mode from --type and legacy --is-bin."""
    if args.type is None:
        return "discrete" if args.is_bin else "continuous"
    if args.type == "continuous" and args.is_bin:
        parser.error("--is-bin conflicts with --type continuous. Use --type discrete.")
    return args.type


def _stem(path):
    """Strip directory and common extensions to get a clean stem for naming."""
    stem = os.path.splitext(path)[0]
    # strip double extensions like .tab.gz, .txt.gz, .tsv.gz
    if stem.endswith(('.tab', '.txt', '.tsv')):
        stem = os.path.splitext(stem)[0]
    return stem


def _setup_outdir(args, source_path):
    """Determine output directory and populate default output paths."""
    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = _stem(source_path) + "_PAGE"

    os.makedirs(outdir, exist_ok=True)

    if args.output is None:
        args.output = os.path.join(outdir, "results.tsv")
    if args.heatmap is None:
        args.heatmap = os.path.join(outdir, "heatmap.pdf")
    if args.html is None:
        args.html = os.path.join(outdir, "heatmap.html")
    if args.killed is None:
        args.killed = os.path.join(outdir, "results.killed.tsv")


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
        "--cols", default=None,
        help="Column names or 1-based numbers for gene and score, "
             "e.g. --cols GENE,log2FoldChange or --cols 8,2",
    )
    parser.add_argument(
        "--no-header", action="store_true", default=False,
        help="Expression file has no header row (columns selected by position)",
    )
    parser.add_argument(
        "--type", choices=["continuous", "discrete"], default=None,
        help="Expression input type. 'continuous' quantizes values; "
             "'discrete' uses pre-binned labels as-is "
             "(default: continuous, or discrete when --is-bin is set)",
    )
    parser.add_argument(
        "--is-bin", action="store_true", default=False,
        help="Legacy alias for --type discrete",
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
        "--outdir", default=None,
        help="Output directory (default: {expression_stem}_PAGE)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output TSV path (default: outdir/results.tsv)",
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
        "--cmap", default="ipage",
        help="Colormap for enrichment heatmap (default: ipage)",
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
        "--min-val", type=float, default=None,
        help="Color scale lower cap (default: -max_val)",
    )
    parser.add_argument(
        "--max-val", type=float, default=5.0,
        help="Color scale upper cap (default: 5)",
    )
    parser.add_argument(
        "--bar-min", type=float, default=None,
        help="Global minimum for bin-edge bar normalization (default: auto)",
    )
    parser.add_argument(
        "--bar-max", type=float, default=None,
        help="Global maximum for bin-edge bar normalization (default: auto)",
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
        if args.matrix is None and args.expression is None:
            parser.error("--matrix or -e/--expression is required when using --draw-only")
        # Derive matrix and outdir from -e when --matrix not given
        if args.matrix is None:
            outdir = args.outdir if args.outdir is not None else _stem(args.expression) + "_PAGE"
            args.matrix = os.path.join(outdir, "results.matrix.tsv")
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
        # Determine outdir: explicit --outdir > derive from -e > derive from --matrix
        if args.outdir is not None:
            draw_outdir = args.outdir
        elif args.expression is not None:
            draw_outdir = _stem(args.expression) + "_PAGE"
        else:
            draw_outdir = _stem(args.matrix) + "_PAGE"
        os.makedirs(draw_outdir, exist_ok=True)
        if args.heatmap is None:
            args.heatmap = os.path.join(draw_outdir, "heatmap.pdf")
        if args.html is None:
            args.html = os.path.join(draw_outdir, "heatmap.html")

        heatmap = Heatmap.from_matrix(args.matrix)
        heatmap.cmap_main = args.cmap
        heatmap.cmap_reg = args.cmap_reg

        # Optionally load expression for regulator overlay
        if args.expression and args.show_reg:
            exp = _load_expression(args.expression, args.cols, args.no_header)
            heatmap.add_gene_expression(exp.genes, exp.raw_expression)

        heatmap.save(args.heatmap, max_rows=args.max_rows,
                     show_reg=args.show_reg, max_val=args.max_val,
                     min_val=args.min_val, title=args.title,
                     bar_min=args.bar_min, bar_max=args.bar_max)
        heatmap.to_html(args.html, max_rows=args.max_rows,
                        show_reg=args.show_reg, max_val=args.max_val,
                        min_val=args.min_val, title=args.title,
                        bar_min=args.bar_min, bar_max=args.bar_max)
        return

    # -- Set up output directory ----------------------------------------------
    _setup_outdir(args, args.expression)

    # -- Load expression ------------------------------------------------------
    expression_mode = _resolve_expression_input_mode(args, parser)
    exp = _load_expression(
        args.expression, args.cols, args.no_header,
        is_bin=(expression_mode == "discrete"), n_bins=args.n_bins,
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
    output_results = results.copy()
    output_results['p-value'] = output_results['p-value'].map(
        lambda x: f'{x:.4e}' if not np.isnan(x) else 'NaN')
    output_results.to_csv(args.output, sep="\t", index=False)
    print(f"Results saved to {args.output}", file=sys.stderr)

    # Auto-save companion matrix
    if heatmap is not None:
        if '.' in args.output:
            matrix_path = args.output.rsplit('.', 1)[0] + '.matrix.tsv'
        else:
            matrix_path = args.output + '.matrix.tsv'
        heatmap.save_matrix(matrix_path)
        print(f"Enrichment matrix saved to {matrix_path}", file=sys.stderr)

    # -- Apply viz params and save heatmap ------------------------------------
    if heatmap is not None:
        heatmap.cmap_main = args.cmap
        heatmap.cmap_reg = args.cmap_reg

        heatmap.save(args.heatmap, max_rows=args.max_rows,
                     show_reg=args.show_reg, max_val=args.max_val,
                     min_val=args.min_val, title=args.title,
                     bar_min=args.bar_min, bar_max=args.bar_max)
        print(f"Heatmap saved to {args.heatmap}", file=sys.stderr)

        heatmap.to_html(args.html, max_rows=args.max_rows,
                        show_reg=args.show_reg, max_val=args.max_val,
                        min_val=args.min_val, title=args.title,
                        bar_min=args.bar_min, bar_max=args.bar_max)
        print(f"HTML heatmap saved to {args.html}", file=sys.stderr)

    # -- Redundancy log -------------------------------------------------------
    killed_df = p.get_redundancy_log()
    killed_df.to_csv(args.killed, sep="\t", index=False)
    print(f"Redundancy log saved to {args.killed}", file=sys.stderr)


if __name__ == "__main__":
    main()
