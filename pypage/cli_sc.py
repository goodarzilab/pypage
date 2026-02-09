"""Command-line interface for single-cell PAGE analysis.

Usage::

    pypage-sc --adata data.h5ad --gmt pathways.gmt [options]
    pypage-sc --expression matrix.tsv --genes genes.txt --genesets-long ann.txt.gz [options]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .io import GeneSets
from .sc import SingleCellPAGE
from .cli import _parse_manual, _stem
from .plotting import (
    plot_consistency_ranking, consistency_ranking_to_html,
    plot_pathway_embedding, interactive_report_to_html,
)


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
    parser.add_argument(
        "--gene-column", default=None,
        help="Column in adata.var containing gene symbols (default: use adata.var_names)",
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
        "--outdir", default=None,
        help="Output directory (default: {input_stem}_scPAGE)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output TSV path (default: outdir/results.tsv)",
    )
    parser.add_argument(
        "--scores", default=None,
        help="Save per-cell scores matrix as TSV (cells x pathways, with header)",
    )

    # -- Visualization options ------------------------------------------------
    parser.add_argument(
        "--ranking-pdf", default=None,
        help="Save consistency ranking bar chart as PDF (default: outdir/ranking.pdf)",
    )
    parser.add_argument(
        "--ranking-html", default=None,
        help="Save consistency ranking as standalone HTML (default: outdir/ranking.html)",
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

    # -- Interactive report & extra outputs -----------------------------------
    parser.add_argument(
        "--report", default=None,
        help="Interactive HTML report path (default: outdir/report.html)",
    )
    parser.add_argument(
        "--no-report", action="store_true", default=False,
        help="Disable interactive report generation",
    )
    parser.add_argument(
        "--no-save-adata", action="store_true", default=False,
        help="Disable saving annotated AnnData with scPAGE scores",
    )
    parser.add_argument(
        "--umap-top-n", type=int, default=10,
        help="Number of top pathways for UMAP PDF plots (default: 10)",
    )
    parser.add_argument(
        "--embedding-key", default=None,
        help="Embedding key for UMAP plots (default: auto-detect X_umap > X_tsne > X_pca)",
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


def _resolve_embedding_key(embeddings, requested_key):
    """Pick the best available embedding key.

    Parameters
    ----------
    embeddings : dict
        Available embeddings from SingleCellPAGE.
    requested_key : str or None
        User-requested key, or None for auto-detection.

    Returns
    -------
    str or None
    """
    if requested_key is not None:
        return requested_key if requested_key in embeddings else None
    for key in ('X_umap', 'X_tsne', 'X_pca'):
        if key in embeddings:
            return key
    return None


def _generate_umap_pdfs(sc, results, outdir, args):
    """Generate per-pathway UMAP PDF plots for top pathways."""
    emb_key = _resolve_embedding_key(sc.embeddings, args.embedding_key)
    if emb_key is None:
        return

    embedding = sc.embeddings[emb_key]

    # Select top pathways: FDR < threshold first, then by consistency
    sig = results[results['FDR'] < args.fdr_threshold]
    if len(sig) >= args.umap_top_n:
        top_pw = sig.sort_values('consistency', ascending=False).head(args.umap_top_n)
    else:
        top_pw = results.sort_values('consistency', ascending=False).head(args.umap_top_n)

    if len(top_pw) == 0:
        return

    umap_dir = os.path.join(outdir, "umap_plots")
    os.makedirs(umap_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pw_name_to_idx = {name: i for i, name in enumerate(sc.pathway_names)}

    for _, row in top_pw.iterrows():
        pw_name = row['pathway']
        if pw_name not in pw_name_to_idx:
            continue
        idx = pw_name_to_idx[pw_name]
        scores = sc.scores[:, idx]

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plot_pathway_embedding(
            scores=scores,
            embedding=embedding,
            pathway_name=pw_name,
            ax=ax,
        )
        safe_name = pw_name.replace('/', '_').replace('\\', '_')
        pdf_path = os.path.join(umap_dir, f"{safe_name}.pdf")
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)

    print(f"UMAP plots saved to {umap_dir}/ ({len(top_pw)} pathways)", file=sys.stderr)


def _extract_draw_only_data(adata_path, gene_column=None):
    """Extract scores, embeddings, and metadata from a saved adata.h5ad.

    Parameters
    ----------
    adata_path : str
        Path to the annotated AnnData file.
    gene_column : str, optional
        Column in adata.var to use as gene names.

    Returns
    -------
    scores : np.ndarray
        Shape (n_cells, n_pathways).
    pw_names : list of str
        Pathway names extracted from scPAGE_ columns.
    embeddings : dict
        Mapping of embedding key to (n_cells, 2) arrays.
    metadata : dict
        Mapping of column name to list of string values.
    """
    import anndata
    adata = anndata.read_h5ad(adata_path)
    if gene_column:
        adata.var_names = adata.var[gene_column].astype(str).values
        adata.var_names_make_unique()

    # Extract scPAGE_ columns
    scpage_cols = [c for c in adata.obs.columns if c.startswith("scPAGE_")]
    pw_names = [c.replace("scPAGE_", "", 1) for c in scpage_cols]
    scores = adata.obs[scpage_cols].values  # (n_cells, n_pathways)

    # Extract embeddings
    embeddings = {}
    for key in ('X_umap', 'X_tsne', 'X_pca'):
        if key in adata.obsm:
            embeddings[key] = np.asarray(adata.obsm[key])

    # Extract categorical metadata
    metadata = {}
    for col in adata.obs.columns:
        if col.startswith("scPAGE_"):
            continue
        if adata.obs[col].dtype.name == 'category' or adata.obs[col].dtype == object:
            metadata[col] = adata.obs[col].astype(str).tolist()

    return scores, pw_names, embeddings, metadata


def _build_pathway_genes(args):
    """Build pathway_genes dict from gene set files if provided.

    Returns
    -------
    dict or None
        Mapping of pathway name to list of gene names, or None.
    """
    if args.genesets is not None:
        gs = GeneSets(ann_file=args.genesets)
    elif args.genesets_long is not None:
        ann = pd.read_csv(
            args.genesets_long, sep="\t", header=None,
            names=["gene", "pathway"],
        )
        gs = GeneSets(ann["gene"], ann["pathway"])
    elif args.gmt is not None:
        gs = GeneSets.from_gmt(args.gmt)
    else:
        return None

    pathway_genes = {}
    for pw_idx, pw_name in enumerate(gs.pathways):
        gene_mask = gs.bool_array[pw_idx] > 0
        pathway_genes[pw_name] = sorted(gs.genes[gene_mask].tolist())
    return pathway_genes


def _generate_umap_pdfs_from_arrays(scores, pw_names, embeddings, results, outdir, args):
    """Generate UMAP PDFs from pre-extracted arrays (for draw-only mode)."""
    emb_key = _resolve_embedding_key(embeddings, args.embedding_key)
    if emb_key is None:
        return

    embedding = embeddings[emb_key]

    sig = results[results['FDR'] < args.fdr_threshold]
    if len(sig) >= args.umap_top_n:
        top_pw = sig.sort_values('consistency', ascending=False).head(args.umap_top_n)
    else:
        top_pw = results.sort_values('consistency', ascending=False).head(args.umap_top_n)

    if len(top_pw) == 0:
        return

    umap_dir = os.path.join(outdir, "umap_plots")
    os.makedirs(umap_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pw_name_to_idx = {name: i for i, name in enumerate(pw_names)}

    for _, row in top_pw.iterrows():
        pw_name = row['pathway']
        if pw_name not in pw_name_to_idx:
            continue
        idx = pw_name_to_idx[pw_name]
        pw_scores = scores[:, idx]

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plot_pathway_embedding(
            scores=pw_scores,
            embedding=embedding,
            pathway_name=pw_name,
            ax=ax,
        )
        safe_name = pw_name.replace('/', '_').replace('\\', '_')
        pdf_path = os.path.join(umap_dir, f"{safe_name}.pdf")
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)

    print(f"UMAP plots saved to {umap_dir}/ ({len(top_pw)} pathways)", file=sys.stderr)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # -- Validate conditional requirements ------------------------------------
    if args.draw_only:
        if args.results is None and args.adata is None and args.expression is None and args.outdir is None:
            parser.error("--outdir, --results, --adata, or --expression is required when using --draw-only")
    else:
        if args.adata is None and args.expression is None:
            parser.error("one of --adata, --expression is required when not using --draw-only")
        if args.genesets is None and args.genesets_long is None and args.gmt is None:
            parser.error("one of -g/--genesets, --genesets-long, --gmt is required when not using --draw-only")

    if args.expression is not None and not args.draw_only and args.genes is None:
        parser.error("--genes is required when using --expression")

    # -- Seed -----------------------------------------------------------------
    if args.seed is not None:
        np.random.seed(args.seed)

    # -- Draw-only mode -------------------------------------------------------
    if args.draw_only:
        # Determine outdir and results path
        if args.outdir is not None:
            draw_outdir = args.outdir
        elif args.results is not None:
            draw_outdir = os.path.dirname(args.results) or "."
        elif args.adata is not None:
            draw_outdir = _stem(args.adata) + "_scPAGE"
        else:
            draw_outdir = _stem(args.expression) + "_scPAGE"
        os.makedirs(draw_outdir, exist_ok=True)

        if args.results is None:
            args.results = os.path.join(draw_outdir, "results.tsv")
        if args.ranking_pdf is None:
            args.ranking_pdf = os.path.join(draw_outdir, "ranking.pdf")
        if args.ranking_html is None:
            args.ranking_html = os.path.join(draw_outdir, "ranking.html")
        if args.report is None and not args.no_report:
            args.report = os.path.join(draw_outdir, "report.html")

        results = pd.read_csv(args.results, sep='\t')

        # Generate ranking plots (always)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ax = plot_consistency_ranking(
            results, top_n=args.top_n, fdr_threshold=args.fdr_threshold)
        if args.title:
            ax.set_title(args.title)
        ax.figure.savefig(args.ranking_pdf, bbox_inches='tight')
        plt.close(ax.figure)
        print(f"Ranking plot saved to {args.ranking_pdf}", file=sys.stderr)

        consistency_ranking_to_html(
            results, args.ranking_html, top_n=args.top_n,
            fdr_threshold=args.fdr_threshold, title=args.title)
        print(f"Ranking HTML saved to {args.ranking_html}", file=sys.stderr)

        # Try to load adata for report + UMAP PDFs
        adata_path = args.adata or os.path.join(draw_outdir, "adata.h5ad")
        if os.path.exists(adata_path):
            scores, pw_names, embeddings, metadata = _extract_draw_only_data(
                adata_path, gene_column=args.gene_column)

            # Build pathway_genes from gene sets if provided
            pathway_genes = _build_pathway_genes(args)

            # Generate UMAP PDFs
            if embeddings:
                _generate_umap_pdfs_from_arrays(
                    scores, pw_names, embeddings, results, draw_outdir, args)

            # Generate interactive report
            if args.report and not args.no_report and embeddings:
                interactive_report_to_html(
                    results=results,
                    scores=scores,
                    pathway_names=pw_names,
                    embeddings=embeddings,
                    output_path=args.report,
                    fdr_threshold=args.fdr_threshold,
                    title=args.title or 'pyPAGE-SC Interactive Report',
                    pathway_genes=pathway_genes,
                    metadata=metadata,
                )
                print(f"Interactive report saved to {args.report}", file=sys.stderr)
        else:
            print(f"No adata.h5ad found at {adata_path}; only regenerating ranking plots",
                  file=sys.stderr)
        return

    # -- Set up output directory ----------------------------------------------
    source_path = args.adata if args.adata is not None else args.expression
    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = _stem(source_path) + "_scPAGE"
    os.makedirs(outdir, exist_ok=True)

    if args.output is None:
        args.output = os.path.join(outdir, "results.tsv")
    if args.ranking_pdf is None:
        args.ranking_pdf = os.path.join(outdir, "ranking.pdf")
    if args.ranking_html is None:
        args.ranking_html = os.path.join(outdir, "ranking.html")
    if args.report is None and not args.no_report:
        args.report = os.path.join(outdir, "report.html")
    save_adata_path = None
    if not args.no_save_adata and args.adata is not None:
        save_adata_path = os.path.join(outdir, "adata.h5ad")

    # -- Load expression ------------------------------------------------------
    adata = None
    expression = None
    genes = None

    if args.adata is not None:
        import anndata
        adata = anndata.read_h5ad(args.adata)
        if args.gene_column is not None:
            if args.gene_column not in adata.var.columns:
                parser.error(
                    f"--gene-column '{args.gene_column}' not found in adata.var. "
                    f"Available columns: {list(adata.var.columns)}"
                )
            adata.var_names = adata.var[args.gene_column].astype(str).values
            adata.var_names_make_unique()
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
    results.to_csv(args.output, sep="\t", index=False)
    print(f"Results saved to {args.output}", file=sys.stderr)

    # -- Write per-cell scores ------------------------------------------------
    if args.scores is not None:
        if args.manual is not None:
            pathway_names_out = pathway_names
        else:
            pathway_names_out = list(sc.pathway_names)
        scores_df = pd.DataFrame(sc.scores, columns=pathway_names_out)
        scores_df.to_csv(args.scores, sep="\t", index=False)
        print(f"Scores saved to {args.scores}", file=sys.stderr)

    # -- Add scores to adata.obs ----------------------------------------------
    if adata is not None:
        if args.manual is not None:
            pw_names_out = pathway_names
        else:
            pw_names_out = list(sc.pathway_names)
        for i, pw_name in enumerate(pw_names_out):
            adata.obs[f"scPAGE_{pw_name}"] = sc.scores[:, i]

    # -- Save annotated adata -------------------------------------------------
    if save_adata_path is not None and adata is not None:
        adata.write_h5ad(save_adata_path)
        print(f"Annotated AnnData saved to {save_adata_path}", file=sys.stderr)

    # -- UMAP PDFs for top pathways -------------------------------------------
    _generate_umap_pdfs(sc, results, outdir, args)

    # -- Visualization --------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ax = plot_consistency_ranking(
        results, top_n=args.top_n, fdr_threshold=args.fdr_threshold)
    if args.title:
        ax.set_title(args.title)
    ax.figure.savefig(args.ranking_pdf, bbox_inches='tight')
    plt.close(ax.figure)
    print(f"Ranking plot saved to {args.ranking_pdf}", file=sys.stderr)

    consistency_ranking_to_html(
        results, args.ranking_html, top_n=args.top_n,
        fdr_threshold=args.fdr_threshold, title=args.title)
    print(f"Ranking HTML saved to {args.ranking_html}", file=sys.stderr)

    # -- Interactive report ---------------------------------------------------
    if args.report and not args.no_report:
        if sc.embeddings:
            # Build pathway_genes dict from GeneSets
            pathway_genes = {}
            pw_names_for_report = list(sc.pathway_names) if args.manual is None else pathway_names
            for i, pw_name in enumerate(sc.pathway_names):
                gene_mask = sc.ont_bool[i] > 0
                pathway_genes[pw_name] = list(sc.shared_genes[gene_mask])

            # Build metadata dict from adata.obs categorical columns
            report_metadata = None
            if adata is not None:
                report_metadata = {}
                for col in adata.obs.columns:
                    if col.startswith("scPAGE_"):
                        continue
                    if adata.obs[col].dtype.name == 'category' or adata.obs[col].dtype == object:
                        report_metadata[col] = adata.obs[col].astype(str).tolist()

            interactive_report_to_html(
                results=results,
                scores=sc.scores,
                pathway_names=pw_names_for_report,
                embeddings=sc.embeddings,
                output_path=args.report,
                fdr_threshold=args.fdr_threshold,
                title=args.title or 'pyPAGE-SC Interactive Report',
                pathway_genes=pathway_genes,
                metadata=report_metadata,
            )
            print(f"Interactive report saved to {args.report}", file=sys.stderr)
        else:
            print("Skipping interactive report: no embeddings available", file=sys.stderr)


if __name__ == "__main__":
    main()
