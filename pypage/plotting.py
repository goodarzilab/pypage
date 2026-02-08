"""Visualization functions for single-cell PAGE results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def plot_pathway_embedding(
    scores,
    embedding,
    pathway_name='',
    ax=None,
    cmap='viridis',
    size=5,
    **kwargs,
):
    """Scatter plot of pathway scores on a 2D embedding.

    Parameters
    ----------
    scores : np.ndarray
        1D array of shape (n_cells,) with pathway scores.
    embedding : np.ndarray
        2D array of shape (n_cells, 2) with coordinates.
    pathway_name : str
        Name for the title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.
    cmap : str
        Colormap name.
    size : float
        Point size.
    **kwargs
        Passed to ax.scatter.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=scores,
        s=size,
        cmap=cmap,
        edgecolors='none',
        rasterized=True,
        **kwargs,
    )
    plt.colorbar(sc, ax=ax, shrink=0.8, label='score')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    if pathway_name:
        ax.set_title(pathway_name)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_pathway_heatmap(
    results_df,
    scores_matrix,
    group_names,
    pathway_names,
    max_rows=50,
    cmap='viridis',
    ax=None,
    **kwargs,
):
    """Heatmap of pathway activity across cell groups.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from SingleCellPAGE.run() with 'pathway' and 'FDR' columns.
    scores_matrix : np.ndarray
        2D array of shape (n_groups, n_pathways) with mean scores.
    group_names : np.ndarray
        Labels for each group.
    pathway_names : np.ndarray
        Names for each pathway.
    max_rows : int
        Maximum number of pathways to show.
    cmap : str
        Colormap name.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.
    **kwargs
        Passed to ax.imshow.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # Select top pathways by FDR then consistency
    if 'FDR' in results_df.columns:
        sorted_df = results_df.sort_values(
            ['FDR', 'consistency'], ascending=[True, False]
        )
    else:
        sorted_df = results_df

    top_pathways = sorted_df['pathway'].values[:max_rows]

    # Map pathway names to column indices
    pw_to_idx = {pw: i for i, pw in enumerate(pathway_names)}
    col_idxs = [pw_to_idx[pw] for pw in top_pathways if pw in pw_to_idx]
    top_pathways = [pw for pw in top_pathways if pw in pw_to_idx]

    if len(col_idxs) == 0:
        raise ValueError("No matching pathways found")

    matrix = scores_matrix[:, col_idxs].T  # (n_pathways, n_groups)

    if ax is None:
        h = max(4, len(top_pathways) * 0.3)
        w = max(4, len(group_names) * 0.5 + 2)
        fig, ax = plt.subplots(1, 1, figsize=(w, h))

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', **kwargs)
    plt.colorbar(im, ax=ax, shrink=0.8, label='mean score')

    ax.set_xticks(np.arange(len(group_names)))
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(top_pathways)))
    ax.set_yticklabels(top_pathways, fontsize=8)
    ax.set_xlabel('Cell group')
    ax.set_ylabel('Pathway')
    return ax


def plot_consistency_ranking(
    results,
    top_n=30,
    fdr_threshold=0.05,
    ax=None,
):
    """Bar plot of top pathways ranked by consistency score (C').

    Parameters
    ----------
    results : pd.DataFrame
        From SingleCellPAGE.run(), with 'pathway', 'consistency', 'FDR'.
    top_n : int
        Number of pathways to show.
    fdr_threshold : float
        FDR threshold for coloring significant pathways.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = results.sort_values('consistency', ascending=False).head(top_n)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, max(4, len(df) * 0.3)))

    colors = [
        '#2166ac' if fdr < fdr_threshold else '#d6604d'
        for fdr in df['FDR']
    ]

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['consistency'].values, color=colors, edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['pathway'].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Consistency (C' = 1 - Geary's C)")
    ax.set_title('Pathway consistency ranking')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2166ac', label=f'FDR < {fdr_threshold}'),
        Patch(facecolor='#d6604d', label=f'FDR >= {fdr_threshold}'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    return ax
