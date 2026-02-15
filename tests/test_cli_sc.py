import numpy as np
import pandas as pd
import pytest

from pypage.cli_sc import _build_parser, main
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
