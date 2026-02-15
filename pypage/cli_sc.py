"""Command-line interface for single-cell PAGE analysis.

Usage::

    pypage-sc --adata data.h5ad --gmt pathways.gmt [options]
    pypage-sc --expression matrix.tsv --genes genes.txt --genesets-long ann.txt.gz [options]
"""

import argparse
import json
import os
import shlex
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from .io import GeneSets
from .sc import SingleCellPAGE
from .cli import _parse_manual, _stem
from .plotting import (
    plot_consistency_ranking, consistency_ranking_to_html,
    plot_pathway_embedding, interactive_report_to_html,
    compute_group_enrichment_stats, plot_group_enrichment_pdfs,
)


def _status(message):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", file=sys.stderr, flush=True)


def _shell_join(argv):
    return " ".join(shlex.quote(str(x)) for x in argv)


def _file_sig(path):
    if path is None:
        return None
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    return {
        "path": path,
        "exists": True,
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _build_resume_signature(args):
    gs_path = args.genesets or args.genesets_long or args.gmt
    return {
        "inputs": {
            "adata": _file_sig(args.adata),
            "expression": _file_sig(args.expression),
            "genes": _file_sig(args.genes),
            "genesets": _file_sig(gs_path),
        },
        "params": {
            "gene_column": args.gene_column,
            "function": args.function,
            "n_bins": args.n_bins,
            "bin_axis": args.bin_axis,
            "n_neighbors": args.n_neighbors,
            "n_permutations": args.n_permutations,
            "perm_chunk_size": args.perm_chunk_size,
            "score_chunk_size": args.score_chunk_size,
            "fast_mode": args.fast_mode,
            "n_jobs": args.n_jobs,
            "filter_redundant": args.filter_redundant,
            "redundancy_ratio": args.redundancy_ratio,
            "redundancy_scope": args.redundancy_scope,
            "redundancy_fdr": args.redundancy_fdr,
            "manual": args.manual,
            "seed": args.seed,
        },
    }


def _build_artifact_signatures(args):
    return {
        "ranking": {
            "top_n": args.top_n,
            "fdr_threshold": args.fdr_threshold,
            "title": args.title,
        },
        "ranking_html": {
            "top_n": args.top_n,
            "fdr_threshold": args.fdr_threshold,
            "title": args.title,
        },
        "report": {
            "fdr_threshold": args.fdr_threshold,
            "title": args.title,
            "groupby": args.groupby,
            "sc_cmap": args.sc_cmap,
            "report_vmin": args.report_vmin,
            "report_vmax": args.report_vmax,
        },
        "umap": {
            "umap_top_n": args.umap_top_n,
            "fdr_threshold": args.fdr_threshold,
            "embedding_key": args.embedding_key,
            "sc_cmap": args.sc_cmap,
        },
        "group_enrichment": {
            "groupby": args.groupby,
            "group_enrichment_top_n": args.group_enrichment_top_n,
            "fdr_threshold": args.fdr_threshold,
            "no_group_enrichment": args.no_group_enrichment,
        },
    }


def _manifest_matches(path, current_sig):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            saved = json.load(f)
    except Exception:
        return False
    return (
        isinstance(saved, dict) and
        saved.get("signature") == current_sig and
        saved.get("status") == "completed"
    )


def _load_artifact_signatures(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    sigs = payload.get("signatures")
    return sigs if isinstance(sigs, dict) else {}


def _save_artifact_signatures(path, signatures):
    with open(path, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "signatures": signatures,
            },
            f,
            indent=2,
        )


def _artifact_signature_matches(saved, key, current):
    return isinstance(saved, dict) and saved.get(key) == current


def _write_run_metadata(run_dir, argv, draw_only_cmd, manifest_path, signature):
    os.makedirs(run_dir, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    command_txt = os.path.join(run_dir, "command.txt")
    command_json = os.path.join(run_dir, "command.json")
    with open(command_txt, "w") as f:
        f.write(f"# Generated: {now}\n")
        f.write(f"# CWD: {os.getcwd()}\n")
        f.write("\nRun command:\n")
        f.write(_shell_join(argv) + "\n")
        f.write("\nDraw-only command:\n")
        f.write(draw_only_cmd + "\n")
    with open(command_json, "w") as f:
        json.dump(
            {
                "generated_at": now,
                "cwd": os.getcwd(),
                "run_command": argv,
                "draw_only_command": draw_only_cmd,
            },
            f,
            indent=2,
        )
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "generated_at": now,
                "status": "completed",
                "signature": signature,
            },
            f,
            indent=2,
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
    parser.add_argument(
        "--groupby", default=None,
        help="Metadata column in adata.obs for report group-enrichment bars.",
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
        "--bin-axis", choices=["cell", "gene"], default="cell",
        help="Discretize expression per cell (VISION-like) or per gene (default: cell)",
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
        "--perm-chunk-size", type=int, default=None,
        help="Chunk size for permutation null generation (default: auto, memory-aware)",
    )
    parser.add_argument(
        "--score-chunk-size", type=int, default=256,
        help="Number of pathways per scoring chunk (default: 256; lower improves Ctrl+C responsiveness)",
    )
    parser.add_argument(
        "--fast-mode", action="store_true", default=False,
        help="Apply redundancy filtering after Geary's C to reduce permutation workload",
    )
    parser.add_argument(
        "--filter-redundant",
        dest="filter_redundant",
        action="store_true",
        default=True,
        help="Enable redundancy filtering via CMI/MI ratio (default: enabled)",
    )
    parser.add_argument(
        "--no-filter-redundant",
        dest="filter_redundant",
        action="store_false",
        help="Disable redundancy filtering",
    )
    parser.add_argument(
        "--redundancy-ratio", type=float, default=5.0,
        help="CMI/MI ratio threshold for redundancy filtering (default: 5.0)",
    )
    parser.add_argument(
        "--redundancy-scope", choices=["fdr", "all"], default="fdr",
        help="Apply redundancy filtering to FDR-significant pathways or all pathways (default: fdr)",
    )
    parser.add_argument(
        "--redundancy-fdr", type=float, default=0.05,
        help="FDR threshold used when --redundancy-scope=fdr (default: 0.05)",
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
    parser.add_argument(
        "--killed", default=None,
        help="Save redundancy log as TSV (default: outdir/results.killed.tsv)",
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
        help="Interactive HTML report path (default: outdir/sc_report.html)",
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
    parser.add_argument(
        "--sc-cmap", default="ipage",
        help="Colormap for single-cell embedding plots/report (default: ipage)",
    )
    parser.add_argument(
        "--group-enrichment-top-n", type=int, default=10,
        help="Number of pathways for group-enrichment PDF/stat outputs (default: 10)",
    )
    parser.add_argument(
        "--no-group-enrichment", action="store_true", default=False,
        help="Disable grouped enrichment PDF/stat output",
    )
    parser.add_argument(
        "--report-vmin", type=float, default=None,
        help="Fixed lower bound for interactive report score colormap (default: auto)",
    )
    parser.add_argument(
        "--report-vmax", type=float, default=None,
        help="Fixed upper bound for interactive report score colormap (default: auto)",
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
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Reuse existing outputs when inputs/parameters match prior completed run",
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


def _generate_umap_pdfs(sc, results, umap_dir, args):
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
            cmap=args.sc_cmap,
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


def _generate_umap_pdfs_from_arrays(scores, pw_names, embeddings, results, umap_dir, args):
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
            cmap=args.sc_cmap,
        )
        safe_name = pw_name.replace('/', '_').replace('\\', '_')
        pdf_path = os.path.join(umap_dir, f"{safe_name}.pdf")
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)

    print(f"UMAP plots saved to {umap_dir}/ ({len(top_pw)} pathways)", file=sys.stderr)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _status("Starting pypage-sc")
    argv_used = list(sys.argv) if argv is None else ["pypage-sc", *argv]

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
    if args.groupby is not None and not args.draw_only and args.adata is None:
        parser.error("--groupby requires --adata (metadata comes from adata.obs)")
    if args.report_vmin is not None and args.report_vmax is not None:
        if args.report_vmax <= args.report_vmin:
            parser.error("--report-vmax must be greater than --report-vmin")

    # -- Seed -----------------------------------------------------------------
    if args.seed is not None:
        np.random.seed(args.seed)
        _status(f"Random seed set to {args.seed}")

    # -- Draw-only mode -------------------------------------------------------
    if args.draw_only:
        _status("Running in draw-only mode")
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
        _status(f"Draw output directory: {draw_outdir}")

        if args.results is None:
            results_candidates = [
                os.path.join(draw_outdir, "tables", "results.tsv"),
                os.path.join(draw_outdir, "results.tsv"),
            ]
            args.results = next((p for p in results_candidates if os.path.exists(p)), results_candidates[0])
        if args.ranking_pdf is None:
            args.ranking_pdf = os.path.join(draw_outdir, "plots", "ranking.pdf")
        if args.ranking_html is None:
            args.ranking_html = os.path.join(draw_outdir, "plots", "ranking.html")
        if args.report is None and not args.no_report:
            args.report = os.path.join(draw_outdir, "sc_report.html")
        group_enrich_dir = os.path.join(draw_outdir, "plots", "group_enrichment")
        group_stats_path = os.path.join(group_enrich_dir, "sc_group_enrichment_stats.tsv")

        results = pd.read_csv(args.results, sep='\t')
        _status(f"Loaded results table: {args.results} ({len(results)} pathways)")

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
        adata_candidates = [
            args.adata,
            os.path.join(draw_outdir, "adata.h5ad"),
            os.path.join(draw_outdir, "adata", "adata.h5ad"),
        ]
        adata_path = next((p for p in adata_candidates if p and os.path.exists(p)), None)
        if adata_path is not None and os.path.exists(adata_path):
            _status(f"Loading AnnData for draw-only assets: {adata_path}")
            scores, pw_names, embeddings, metadata = _extract_draw_only_data(
                adata_path, gene_column=args.gene_column)
            _status(
                f"Loaded draw-only data: {scores.shape[0]} cells, "
                f"{scores.shape[1]} pathways, {len(embeddings)} embeddings"
            )
            if args.groupby is not None and args.groupby not in metadata:
                parser.error(
                    f"--groupby '{args.groupby}' not found in adata.obs categorical columns. "
                    f"Available columns: {sorted(metadata.keys())}"
                )

            # Build pathway_genes from gene sets if provided
            pathway_genes = _build_pathway_genes(args)
            group_stats_df = None

            # Generate UMAP PDFs
            if embeddings:
                _status("Generating per-pathway embedding PDFs")
                umap_dir = os.path.join(draw_outdir, "plots", "umap_plots")
                _generate_umap_pdfs_from_arrays(
                    scores, pw_names, embeddings, results, umap_dir, args)

            if args.groupby is not None and not args.no_group_enrichment and args.groupby in metadata:
                _status("Generating group-enrichment PDFs and stats")
                group_stats_df = compute_group_enrichment_stats(
                    scores=scores,
                    pathway_names=pw_names,
                    results=results,
                    group_labels=np.asarray(metadata[args.groupby]),
                    group_name=args.groupby,
                    top_n=args.group_enrichment_top_n,
                    fdr_threshold=args.fdr_threshold,
                )
                os.makedirs(group_enrich_dir, exist_ok=True)
                group_stats_df.to_csv(group_stats_path, sep="\t", index=False)
                out_pdfs = plot_group_enrichment_pdfs(group_stats_df, group_enrich_dir)
                _status(
                    f"Group-enrichment outputs: {len(out_pdfs)} PDFs, stats={group_stats_path}"
                )

            # Generate interactive report
            if args.report and not args.no_report and embeddings:
                _status("Building interactive report")
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
                    groupby=args.groupby,
                    cmap=args.sc_cmap,
                    group_stats_df=group_stats_df,
                    score_vmin=args.report_vmin,
                    score_vmax=args.report_vmax,
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
    _status(f"Output directory: {outdir}")
    run_dir = os.path.join(outdir, "run")
    tables_dir = os.path.join(outdir, "tables")
    plots_dir = os.path.join(outdir, "plots")
    umap_dir = os.path.join(plots_dir, "umap_plots")
    group_enrich_dir = os.path.join(plots_dir, "group_enrichment")
    group_stats_path = os.path.join(group_enrich_dir, "sc_group_enrichment_stats.tsv")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if args.output is None:
        args.output = os.path.join(tables_dir, "results.tsv")
    if args.ranking_pdf is None:
        args.ranking_pdf = os.path.join(plots_dir, "ranking.pdf")
    if args.ranking_html is None:
        args.ranking_html = os.path.join(plots_dir, "ranking.html")
    if args.report is None and not args.no_report:
        args.report = os.path.join(outdir, "sc_report.html")
    if args.killed is None:
        args.killed = os.path.join(tables_dir, "results.killed.tsv")
    save_adata_path = None
    if not args.no_save_adata and args.adata is not None:
        save_adata_path = os.path.join(outdir, "adata.h5ad")
    manifest_path = os.path.join(run_dir, "manifest.json")
    artifacts_path = os.path.join(run_dir, "artifacts.json")

    draw_only_cmd = _shell_join([
        "pypage-sc", "--draw-only", "--outdir", outdir,
        "--top-n", args.top_n,
        "--fdr-threshold", args.fdr_threshold,
        "--umap-top-n", args.umap_top_n,
        *([] if args.groupby is None else ["--groupby", args.groupby]),
        *([] if args.gene_column is None else ["--gene-column", args.gene_column]),
    ])
    signature = _build_resume_signature(args)
    artifact_signatures = _build_artifact_signatures(args)
    saved_artifacts = _load_artifact_signatures(artifacts_path) if args.resume else {}
    updated_artifacts = dict(saved_artifacts)

    # -- Resume check ---------------------------------------------------------
    results = None
    skip_analysis = False
    if args.resume and _manifest_matches(manifest_path, signature):
        if os.path.exists(args.output) and os.path.exists(args.killed):
            if args.scores is None or os.path.exists(args.scores):
                if save_adata_path is None or os.path.exists(save_adata_path):
                    _status("Resume: manifest matched; skipping pathway analysis")
                    results = pd.read_csv(args.output, sep="\t")
                    skip_analysis = True

    # -- Load expression ------------------------------------------------------
    adata = None
    expression = None
    genes = None
    if not skip_analysis:
        if args.adata is not None:
            _status(f"Loading AnnData: {args.adata}")
            import anndata
            adata = anndata.read_h5ad(args.adata)
            _status(
                f"AnnData loaded: {adata.n_obs} cells, {adata.n_vars} genes"
            )
            if args.gene_column is not None:
                if args.gene_column not in adata.var.columns:
                    parser.error(
                        f"--gene-column '{args.gene_column}' not found in adata.var. "
                        f"Available columns: {list(adata.var.columns)}"
                    )
                adata.var_names = adata.var[args.gene_column].astype(str).values
                adata.var_names_make_unique()
                _status(f"Gene symbols mapped from adata.var['{args.gene_column}']")
            if args.groupby is not None and args.groupby not in adata.obs.columns:
                parser.error(
                    f"--groupby '{args.groupby}' not found in adata.obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )
        else:
            _status(f"Loading expression matrix: {args.expression}")
            expression = np.loadtxt(args.expression, delimiter="\t")
            with open(args.genes) as f:
                genes = np.array([line.strip() for line in f if line.strip()])
            _status(
                f"Expression loaded: {expression.shape[0]} cells, {expression.shape[1]} genes"
            )
    else:
        _status("Resume: skipping input data load for analysis")

    # -- Load gene sets -------------------------------------------------------
    gs = None
    sc = None
    if not skip_analysis:
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
        _status(
            f"Gene sets loaded: {len(gs.pathways)} pathways, {len(gs.genes)} unique genes"
        )

        # -- Create SingleCellPAGE --------------------------------------------
        sc = SingleCellPAGE(
            adata=adata,
            expression=expression,
            genes=genes,
            genesets=gs,
            n_neighbors=args.n_neighbors,
            n_bins=args.n_bins,
            function=args.function,
            bin_axis=args.bin_axis,
            permutation_chunk_size=args.perm_chunk_size,
            score_chunk_size=args.score_chunk_size,
            fast_mode=args.fast_mode,
            n_jobs=args.n_jobs,
            filter_redundant=args.filter_redundant,
            redundancy_ratio=args.redundancy_ratio,
            redundancy_scope=args.redundancy_scope,
            redundancy_fdr=args.redundancy_fdr,
        )
        _status(
            f"SingleCellPAGE initialized: {sc.n_cells} cells, {sc.n_pathways} pathways, "
            f"shared genes={len(sc.shared_genes)}"
        )
        if getattr(sc, "n_pathways_dropped_zero_shared", 0) > 0:
            _status(
                f"Dropped {sc.n_pathways_dropped_zero_shared} pathways with zero "
                "shared genes"
            )

    # -- Run ------------------------------------------------------------------
    pathway_names = None
    if not skip_analysis:
        if args.manual is not None:
            _status("Manual mode: scoring selected pathways (no permutation test)")
            pathway_names = _parse_manual(args.manual)
            results = sc.run_manual(pathway_names)
        else:
            _status(
                "Running full analysis: scoring pathways, Geary's C, permutation testing "
                f"({args.n_permutations} permutations)"
            )
            results = sc.run(
                n_permutations=args.n_permutations,
                progress_callback=_status,
            )
        _status("Analysis complete")

    # -- Write results --------------------------------------------------------
    if skip_analysis and os.path.exists(args.output):
        _status(f"Resume: using existing results at {args.output}")
    else:
        results.to_csv(args.output, sep="\t", index=False)
        print(f"Results saved to {args.output}", file=sys.stderr)
    if not skip_analysis:
        killed_df = sc.get_redundancy_log()
        killed_df.to_csv(args.killed, sep="\t", index=False)
        print(f"Redundancy log saved to {args.killed}", file=sys.stderr)
    elif os.path.exists(args.killed):
        _status(f"Resume: using existing redundancy log at {args.killed}")

    # -- Write per-cell scores ------------------------------------------------
    if args.scores is not None and not skip_analysis:
        if args.manual is not None:
            pathway_names_out = pathway_names
        else:
            pathway_names_out = list(sc.pathway_names)
        scores_df = pd.DataFrame(sc.scores, columns=pathway_names_out)
        scores_df.to_csv(args.scores, sep="\t", index=False)
        print(f"Scores saved to {args.scores}", file=sys.stderr)

    # -- Add scores to adata.obs ----------------------------------------------
    if adata is not None and not skip_analysis:
        if args.manual is not None:
            pw_names_out = pathway_names
        else:
            pw_names_out = list(sc.pathway_names)
        score_cols = [f"scPAGE_{pw_name}" for pw_name in pw_names_out]
        score_obs = pd.DataFrame(sc.scores, index=adata.obs.index, columns=score_cols)
        # Batch-join once to avoid pandas DataFrame fragmentation warnings.
        adata.obs = adata.obs.drop(columns=score_cols, errors="ignore")
        adata.obs = pd.concat([adata.obs, score_obs], axis=1)

    # -- Save annotated adata -------------------------------------------------
    if save_adata_path is not None and adata is not None:
        if skip_analysis and os.path.exists(save_adata_path):
            _status(f"Resume: using existing annotated AnnData at {save_adata_path}")
        elif not skip_analysis:
            adata.write_h5ad(save_adata_path)
            print(f"Annotated AnnData saved to {save_adata_path}", file=sys.stderr)

    # -- Group-enrichment PDFs/statistics -------------------------------------
    group_stats_df = None
    if args.groupby is not None and not args.no_group_enrichment:
        group_sig = artifact_signatures["group_enrichment"]
        if (
            args.resume
            and os.path.exists(group_stats_path)
            and _artifact_signature_matches(saved_artifacts, "group_enrichment", group_sig)
        ):
            _status(f"Resume: using existing group-enrichment stats at {group_stats_path}")
            group_stats_df = pd.read_csv(group_stats_path, sep="\t")
            updated_artifacts["group_enrichment"] = group_sig
        else:
            if skip_analysis:
                adata_for_draw = save_adata_path if save_adata_path and os.path.exists(save_adata_path) else args.adata
                if adata_for_draw and os.path.exists(adata_for_draw):
                    scores_arr, pw_names, _, metadata = _extract_draw_only_data(
                        adata_for_draw, gene_column=args.gene_column
                    )
                    if args.groupby in metadata:
                        group_stats_df = compute_group_enrichment_stats(
                            scores=scores_arr,
                            pathway_names=pw_names,
                            results=results,
                            group_labels=np.asarray(metadata[args.groupby]),
                            group_name=args.groupby,
                            top_n=args.group_enrichment_top_n,
                            fdr_threshold=args.fdr_threshold,
                        )
            else:
                if adata is not None and args.groupby in adata.obs.columns:
                    pw_names = pathway_names if args.manual is not None else list(sc.pathway_names)
                    group_stats_df = compute_group_enrichment_stats(
                        scores=sc.scores,
                        pathway_names=pw_names,
                        results=results,
                        group_labels=adata.obs[args.groupby].astype(str).values,
                        group_name=args.groupby,
                        top_n=args.group_enrichment_top_n,
                        fdr_threshold=args.fdr_threshold,
                    )
            if group_stats_df is not None and len(group_stats_df) > 0:
                os.makedirs(group_enrich_dir, exist_ok=True)
                group_stats_df.to_csv(group_stats_path, sep="\t", index=False)
                out_pdfs = plot_group_enrichment_pdfs(group_stats_df, group_enrich_dir)
                _status(
                    f"Group-enrichment outputs: {len(out_pdfs)} PDFs, stats={group_stats_path}"
                )
                updated_artifacts["group_enrichment"] = group_sig

    # -- UMAP PDFs for top pathways -------------------------------------------
    if skip_analysis:
        umap_sig = artifact_signatures["umap"]
        existing_umaps = []
        if os.path.isdir(umap_dir):
            existing_umaps = [x for x in os.listdir(umap_dir) if x.endswith(".pdf")]
        if (
            args.resume
            and len(existing_umaps) > 0
            and _artifact_signature_matches(saved_artifacts, "umap", umap_sig)
        ):
            _status(f"Resume: using existing UMAP plots in {umap_dir} ({len(existing_umaps)} files)")
            updated_artifacts["umap"] = umap_sig
        else:
            adata_for_draw = save_adata_path if save_adata_path and os.path.exists(save_adata_path) else args.adata
            if adata_for_draw and os.path.exists(adata_for_draw):
                scores, pw_names, embeddings, _ = _extract_draw_only_data(
                    adata_for_draw, gene_column=args.gene_column
                )
                _generate_umap_pdfs_from_arrays(
                    scores, pw_names, embeddings, results, umap_dir, args
                )
                updated_artifacts["umap"] = umap_sig
            else:
                _status("Resume: skipping UMAP PDFs (no annotated AnnData available)")
    else:
        _generate_umap_pdfs(sc, results, umap_dir, args)
        updated_artifacts["umap"] = artifact_signatures["umap"]

    # -- Visualization --------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ax = plot_consistency_ranking(
        results, top_n=args.top_n, fdr_threshold=args.fdr_threshold)
    if args.title:
        ax.set_title(args.title)
    ranking_sig = artifact_signatures["ranking"]
    if not (
        args.resume
        and os.path.exists(args.ranking_pdf)
        and _artifact_signature_matches(saved_artifacts, "ranking", ranking_sig)
    ):
        ax.figure.savefig(args.ranking_pdf, bbox_inches='tight')
        print(f"Ranking plot saved to {args.ranking_pdf}", file=sys.stderr)
        updated_artifacts["ranking"] = ranking_sig
    else:
        _status(f"Resume: using existing ranking plot at {args.ranking_pdf}")
        updated_artifacts["ranking"] = ranking_sig
    plt.close(ax.figure)

    ranking_html_sig = artifact_signatures["ranking_html"]
    if not (
        args.resume
        and os.path.exists(args.ranking_html)
        and _artifact_signature_matches(saved_artifacts, "ranking_html", ranking_html_sig)
    ):
        consistency_ranking_to_html(
            results, args.ranking_html, top_n=args.top_n,
            fdr_threshold=args.fdr_threshold, title=args.title)
        print(f"Ranking HTML saved to {args.ranking_html}", file=sys.stderr)
        updated_artifacts["ranking_html"] = ranking_html_sig
    else:
        _status(f"Resume: using existing ranking HTML at {args.ranking_html}")
        updated_artifacts["ranking_html"] = ranking_html_sig

    # -- Interactive report ---------------------------------------------------
    if args.report and not args.no_report:
        report_sig = artifact_signatures["report"]
        if (
            args.resume
            and os.path.exists(args.report)
            and _artifact_signature_matches(saved_artifacts, "report", report_sig)
        ):
            _status(f"Resume: using existing interactive report at {args.report}")
            updated_artifacts["report"] = report_sig
        elif skip_analysis:
            adata_for_draw = save_adata_path if save_adata_path and os.path.exists(save_adata_path) else args.adata
            if adata_for_draw and os.path.exists(adata_for_draw):
                scores, pw_names, embeddings, metadata = _extract_draw_only_data(
                    adata_for_draw, gene_column=args.gene_column
                )
                pathway_genes = _build_pathway_genes(args)
                if embeddings:
                    _status("Generating interactive report")
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
                        groupby=args.groupby,
                        cmap=args.sc_cmap,
                        group_stats_df=group_stats_df,
                        score_vmin=args.report_vmin,
                        score_vmax=args.report_vmax,
                    )
                    print(f"Interactive report saved to {args.report}", file=sys.stderr)
                    updated_artifacts["report"] = report_sig
                else:
                    print("Skipping interactive report: no embeddings available", file=sys.stderr)
            else:
                _status("Resume: skipping interactive report (no annotated AnnData available)")
        elif sc.embeddings:
            _status("Generating interactive report")
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
                groupby=args.groupby,
                cmap=args.sc_cmap,
                group_stats_df=group_stats_df,
                score_vmin=args.report_vmin,
                score_vmax=args.report_vmax,
            )
            print(f"Interactive report saved to {args.report}", file=sys.stderr)
            updated_artifacts["report"] = report_sig
        else:
            print("Skipping interactive report: no embeddings available", file=sys.stderr)

    _write_run_metadata(run_dir, argv_used, draw_only_cmd, manifest_path, signature)
    _save_artifact_signatures(artifacts_path, updated_artifacts)


if __name__ == "__main__":
    main()
