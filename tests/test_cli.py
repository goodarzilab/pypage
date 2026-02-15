import pytest

from pypage.cli import (
    _build_draw_only_command,
    _build_parser,
    _resolve_draw_matrix_path,
    _resolve_expression_input_mode,
    _setup_outdir,
    main,
)


def test_expression_mode_default_continuous():
    parser = _build_parser()
    args = parser.parse_args([])
    assert _resolve_expression_input_mode(args, parser) == "continuous"


def test_expression_mode_discrete_with_type():
    parser = _build_parser()
    args = parser.parse_args(["--type", "discrete"])
    assert _resolve_expression_input_mode(args, parser) == "discrete"


def test_expression_mode_discrete_with_is_bin_alias():
    parser = _build_parser()
    args = parser.parse_args(["--is-bin"])
    assert _resolve_expression_input_mode(args, parser) == "discrete"


def test_expression_mode_conflict_type_and_is_bin():
    parser = _build_parser()
    args = parser.parse_args(["--type", "continuous", "--is-bin"])
    with pytest.raises(SystemExit):
        _resolve_expression_input_mode(args, parser)


def test_parser_accepts_resume():
    parser = _build_parser()
    args = parser.parse_args(["--resume"])
    assert args.resume is True


def test_setup_outdir_uses_structured_defaults(tmp_path):
    parser = _build_parser()
    args = parser.parse_args(["-e", "expr.tsv", "--outdir", str(tmp_path / "out")])
    outdir, run_dir, tables_dir, plots_dir = _setup_outdir(args, args.expression)

    assert outdir == str(tmp_path / "out")
    assert run_dir == str(tmp_path / "out" / "run")
    assert tables_dir == str(tmp_path / "out" / "tables")
    assert plots_dir == str(tmp_path / "out" / "plots")
    assert args.output == str(tmp_path / "out" / "tables" / "results.tsv")
    assert args.killed == str(tmp_path / "out" / "tables" / "results.killed.tsv")
    assert args.heatmap == str(tmp_path / "out" / "plots" / "heatmap.pdf")
    assert args.html == str(tmp_path / "out" / "heatmap.html")


def test_resolve_draw_matrix_prefers_new_layout(tmp_path):
    parser = _build_parser()
    args = parser.parse_args(["--draw-only", "--expression", "expr.tsv"])
    draw_outdir = tmp_path / "out"
    new_path = draw_outdir / "tables" / "results.matrix.tsv"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text("x")

    resolved = _resolve_draw_matrix_path(args, str(draw_outdir))
    assert resolved == str(new_path)


def test_build_draw_only_command_includes_visualization_options():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--cmap", "viridis",
            "--cmap-reg", "magma",
            "--max-rows", "25",
            "--min-val", "-2.5",
            "--max-val", "4.5",
            "--bar-min", "-1.0",
            "--bar-max", "3.0",
            "--title", "My Plot",
            "--show-reg",
            "--expression", "expr.tsv",
            "--cols", "GENE,log2FC",
            "--no-header",
        ]
    )
    cmd = _build_draw_only_command(args, "outdir")
    assert "--min-val -2.5" in cmd
    assert "--max-val 4.5" in cmd
    assert "--bar-min -1.0" in cmd
    assert "--bar-max 3.0" in cmd
    assert "--cmap viridis" in cmd
    assert "--cmap-reg magma" in cmd
    assert "--max-rows 25" in cmd
    assert "--title 'My Plot'" in cmd
    assert "--show-reg" in cmd
    assert "--expression expr.tsv" in cmd
    assert "--cols GENE,log2FC" in cmd
    assert "--no-header" in cmd


def test_build_draw_only_command_includes_defaults_when_not_set():
    parser = _build_parser()
    args = parser.parse_args(["--max-val", "5.0"])
    cmd = _build_draw_only_command(args, "outdir")
    assert "--min-val -5.0" in cmd
    assert "--bar-min auto" in cmd
    assert "--bar-max auto" in cmd
    assert "--title ''" in cmd
    assert "--no-show-reg" in cmd


def test_resume_falls_back_to_full_run_when_matrix_missing(tmp_path):
    expr = tmp_path / "expr.tsv"
    gmt = tmp_path / "sets.gmt"
    outdir = tmp_path / "out"
    expr.write_text("g1\t1.0\ng2\t-0.2\ng3\t0.5\n")
    gmt.write_text("PW1\tna\tg1\tg2\n")

    main([
        "--resume",
        "-e",
        str(expr),
        "--gmt",
        str(gmt),
        "--n-shuffle",
        "2",
        "-k",
        "1",
        "--outdir",
        str(outdir),
    ])
    assert (outdir / "tables" / "results.tsv").exists()
    assert (outdir / "tables" / "results.killed.tsv").exists()
    assert (outdir / "run" / "status.json").exists()
