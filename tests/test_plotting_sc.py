import matplotlib
import pandas as pd

from pypage.plotting import plot_group_enrichment_pdfs


matplotlib.use("Agg")


def test_group_enrichment_stars_follow_negative_bar_endpoint(tmp_path, monkeypatch):
    stats_df = pd.DataFrame(
        {
            "pathway": ["PW1", "PW1"],
            "group_name": ["cluster", "cluster"],
            "group": ["A", "B"],
            "n_group": [20, 10],
            "n_rest": [10, 20],
            "high_activity_fraction_p75": [0.2, 0.8],
            "mean_score": [-2.0, 1.0],
            "kw_p_value": [0.01, 0.01],
            "one_vs_rest_p_value": [0.01, 0.01],
            "one_vs_rest_fdr": [0.01, 0.5],
        }
    )

    captured = []
    import matplotlib.axes

    orig_text = matplotlib.axes.Axes.text

    def _capture(self, x, y, s, *args, **kwargs):
        if s in {"*", "**", "***"}:
            captured.append((x, y, s))
        return orig_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", _capture)
    out = plot_group_enrichment_pdfs(stats_df, tmp_path)

    assert len(out) == 1
    assert len(captured) == 1
    x, y, star = captured[0]
    assert x == 0
    assert star == "*"
    assert y < stats_df.loc[stats_df["group"] == "A", "mean_score"].iloc[0]
