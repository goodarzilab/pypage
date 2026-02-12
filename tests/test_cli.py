import pytest

from pypage.cli import _build_parser, _resolve_expression_input_mode


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
