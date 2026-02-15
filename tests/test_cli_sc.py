import numpy as np
import pandas as pd
import pytest

from pypage.cli_sc import _build_parser, _build_artifact_signatures, main
from pypage.plotting import interactive_report_to_html


def test_sc_parser_accepts_groupby():
    parser = _build_parser()
    args = parser.parse_args(["--groupby", "PhenoGraph_clusters"])
    assert args.groupby == "PhenoGraph_clusters"


def test_sc_parser_accepts_redundancy_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--no-filter-redundant",
            "--redundancy-ratio", "3.0",
            "--redundancy-scope", "all",
            "--redundancy-fdr", "0.1",
            "--killed", "killed.tsv",
        ]
    )
    assert args.filter_redundant is False
    assert args.redundancy_ratio == 3.0
    assert args.redundancy_scope == "all"
    assert args.redundancy_fdr == 0.1
    assert args.killed == "killed.tsv"


def test_sc_parser_accepts_score_chunk_size():
    parser = _build_parser()
    args = parser.parse_args(["--score-chunk-size", "128"])
    assert args.score_chunk_size == 128


def test_sc_parser_accepts_fast_mode():
    parser = _build_parser()
    args = parser.parse_args(["--fast-mode"])
    assert args.fast_mode is True


def test_sc_parser_accepts_resume():
    parser = _build_parser()
    args = parser.parse_args(["--resume"])
    assert args.resume is True


def test_sc_parser_accepts_report_range():
    parser = _build_parser()
    args = parser.parse_args(["--report-vmin", "-0.2", "--report-vmax", "0.8"])
    assert args.report_vmin == -0.2
    assert args.report_vmax == 0.8


def test_sc_report_help_uses_sc_report_default():
    parser = _build_parser()
    report_action = next(a for a in parser._actions if a.dest == "report")
    assert "outdir/sc_report.html" in report_action.help


def test_artifact_signatures_change_for_report_knobs():
    parser = _build_parser()
    a = parser.parse_args(
        [
            "--top-n", "30",
            "--fdr-threshold", "0.05",
            "--groupby", "cluster",
            "--sc-cmap", "ipage",
            "--report-vmin", "-1",
            "--report-vmax", "1",
        ]
    )
    b = parser.parse_args(
        [
            "--top-n", "30",
            "--fdr-threshold", "0.05",
            "--groupby", "cluster",
            "--sc-cmap", "viridis",
            "--report-vmin", "-0.5",
            "--report-vmax", "0.5",
        ]
    )
    sig_a = _build_artifact_signatures(a)
    sig_b = _build_artifact_signatures(b)
    assert sig_a["report"] != sig_b["report"]
    assert sig_a["umap"] != sig_b["umap"]


def test_artifact_signatures_change_for_group_enrichment_knobs():
    parser = _build_parser()
    a = parser.parse_args(
        [
            "--groupby", "cluster",
            "--group-enrichment-top-n", "10",
            "--fdr-threshold", "0.05",
        ]
    )
    b = parser.parse_args(
        [
            "--groupby", "cluster",
            "--group-enrichment-top-n", "25",
            "--fdr-threshold", "0.1",
        ]
    )
    sig_a = _build_artifact_signatures(a)
    sig_b = _build_artifact_signatures(b)
    assert sig_a["group_enrichment"] != sig_b["group_enrichment"]


def test_sc_groupby_requires_adata():
    with pytest.raises(SystemExit):
        main(
            [
                "--expression",
                "dummy.tsv",
                "--genes",
                "genes.txt",
                "--gmt",
                "sets.gmt",
                "--groupby",
                "cluster",
            ]
        )


def test_interactive_report_includes_group_controls(tmp_path):
    results = pd.DataFrame(
        {
            "pathway": ["PW_A", "PW_B"],
            "consistency": [0.3, 0.2],
            "p-value": [0.01, 0.05],
            "FDR": [0.02, 0.08],
        }
    )
    scores = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.1],
            [0.2, 0.5],
            [0.1, 0.7],
        ],
        dtype=float,
    )
    embeddings = {"X_umap": np.array([[0.0, 0.0], [1.0, 0.2], [0.2, 1.0], [1.2, 1.1]], dtype=float)}
    metadata = {"PhenoGraph_clusters": ["0", "0", "1", "1"]}

    out = tmp_path / "report.html"
    interactive_report_to_html(
        results=results,
        scores=scores,
        pathway_names=["PW_A", "PW_B"],
        embeddings=embeddings,
        output_path=str(out),
        metadata=metadata,
        groupby="PhenoGraph_clusters",
    )
    html = out.read_text()
    assert 'id="groupBySelect"' in html
    assert "High-score fraction (>= P75)" in html
    assert '"groupby_preferred":"PhenoGraph_clusters"' in html
