"""Visualization functions for single-cell PAGE results."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from .utils import benjamini_hochberg


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


def _select_top_pathways(results, top_n=10, fdr_threshold=0.05):
    sig = results[results['FDR'] < fdr_threshold]
    if len(sig) >= top_n:
        top_df = sig.sort_values('consistency', ascending=False).head(top_n)
    else:
        top_df = results.sort_values('consistency', ascending=False).head(top_n)
    return top_df['pathway'].tolist()


def compute_group_enrichment_stats(
    scores,
    pathway_names,
    results,
    group_labels,
    group_name='group',
    top_n=10,
    fdr_threshold=0.05,
):
    """Compute per-pathway group-enrichment stats (Kruskal + one-vs-rest U test)."""
    from scipy.stats import kruskal, mannwhitneyu

    group_labels = np.asarray(group_labels).astype(str)
    if len(group_labels) != scores.shape[0]:
        raise ValueError("group_labels length must match number of cells")

    top_pathways = _select_top_pathways(results, top_n=top_n, fdr_threshold=fdr_threshold)
    pw_to_idx = {pw: i for i, pw in enumerate(pathway_names)}
    uniq = np.unique(group_labels)
    uniq = np.array(sorted(uniq, key=lambda g: int(np.sum(group_labels == g)), reverse=True))

    rows = []
    for pw in top_pathways:
        if pw not in pw_to_idx:
            continue
        vals = np.asarray(scores[:, pw_to_idx[pw]], dtype=np.float64)
        thr = float(np.quantile(vals, 0.75))

        group_arrays = []
        frac = np.zeros(len(uniq), dtype=np.float64)
        mean = np.zeros(len(uniq), dtype=np.float64)
        n_group = np.zeros(len(uniq), dtype=np.int64)
        for j, g in enumerate(uniq):
            idx = group_labels == g
            gv = vals[idx]
            group_arrays.append(gv)
            n_group[j] = int(gv.size)
            if gv.size:
                frac[j] = float(np.mean(gv >= thr))
                mean[j] = float(np.mean(gv))

        kw_p = np.nan
        valid = [a for a in group_arrays if a.size > 0]
        if len(valid) >= 2:
            try:
                kw_p = float(kruskal(*valid, nan_policy="omit").pvalue)
            except Exception:
                kw_p = np.nan

        p_ovr = np.full(len(uniq), np.nan, dtype=np.float64)
        for j, g in enumerate(uniq):
            in_g = group_arrays[j]
            out_g = vals[group_labels != g]
            if in_g.size == 0 or out_g.size == 0:
                continue
            try:
                p_ovr[j] = float(
                    mannwhitneyu(in_g, out_g, alternative="two-sided", method="asymptotic").pvalue
                )
            except Exception:
                p_ovr[j] = np.nan

        q_ovr = np.full(len(uniq), np.nan, dtype=np.float64)
        mask = np.isfinite(p_ovr)
        if np.any(mask):
            q_ovr[mask] = benjamini_hochberg(p_ovr[mask])

        for j, g in enumerate(uniq):
            rows.append({
                "pathway": str(pw),
                "group_name": str(group_name),
                "group": str(g),
                "n_group": int(n_group[j]),
                "n_rest": int(vals.size - n_group[j]),
                "high_activity_fraction_p75": float(frac[j]),
                "mean_score": float(mean[j]),
                "kw_p_value": float(kw_p) if np.isfinite(kw_p) else np.nan,
                "one_vs_rest_p_value": float(p_ovr[j]) if np.isfinite(p_ovr[j]) else np.nan,
                "one_vs_rest_fdr": float(q_ovr[j]) if np.isfinite(q_ovr[j]) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_group_enrichment_pdfs(stats_df, output_dir):
    """Save per-pathway grouped barplots (fraction + mean score with significance stars)."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    out_paths = []
    if stats_df is None or len(stats_df) == 0:
        return out_paths

    for pathway in stats_df['pathway'].drop_duplicates().tolist():
        sdf = stats_df[stats_df['pathway'] == pathway].copy()
        sdf = sdf.sort_values('n_group', ascending=False)
        if len(sdf) == 0:
            continue

        x = np.arange(len(sdf))
        frac = sdf['high_activity_fraction_p75'].values
        mean = sdf['mean_score'].values
        qvals = sdf['one_vs_rest_fdr'].values

        fig_w = max(7.0, 0.32 * len(sdf) + 2.8)
        fig, axes = plt.subplots(2, 1, figsize=(fig_w, 5.2), sharex=True)

        axes[0].bar(x, frac, color="#2f6da6", edgecolor='none', width=0.82)
        axes[0].set_ylabel("High-activity\nfraction (>=P75)", fontsize=8)
        axes[0].set_ylim(0.0, max(0.05, float(np.max(frac) * 1.08)))
        axes[0].grid(axis='y', alpha=0.2, linewidth=0.6)

        axes[1].bar(x, mean, color="#3a9d5d", edgecolor='none', width=0.82)
        axes[1].set_ylabel("Mean score", fontsize=8)
        axes[1].grid(axis='y', alpha=0.2, linewidth=0.6)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(sdf['group'].astype(str).tolist(), rotation=90, fontsize=7)
        axes[1].axhline(0, color="#666", linewidth=0.6, alpha=0.8)

        ymax = float(np.nanmax(mean)) if len(mean) else 0.0
        ymin = float(np.nanmin(mean)) if len(mean) else 0.0
        yspan = max(1e-9, ymax - ymin)
        pad = 0.12 * yspan + 0.02 * max(1.0, abs(ymax), abs(ymin))
        low = min(0.0, ymin) - pad
        high = max(0.0, ymax) + pad
        axes[1].set_ylim(low, high)
        yoff = 0.03 * (high - low)
        for i, qv in enumerate(qvals):
            if not np.isfinite(qv):
                continue
            star = ""
            if qv < 1e-3:
                star = "***"
            elif qv < 1e-2:
                star = "**"
            elif qv < 5e-2:
                star = "*"
            if star:
                val = mean[i]
                if val >= 0:
                    y = val + yoff
                    va = 'bottom'
                else:
                    y = val - yoff
                    va = 'top'
                axes[1].text(
                    i,
                    y,
                    star,
                    ha='center',
                    va=va,
                    fontsize=8,
                    color='#222',
                )

        kw_p = sdf['kw_p_value'].dropna().iloc[0] if sdf['kw_p_value'].notna().any() else np.nan
        fig.suptitle(pathway, fontsize=9)
        if np.isfinite(kw_p):
            fig.text(0.01, 0.98, f"Kruskal-Wallis p={kw_p:.2e}", fontsize=8, va='top')
        fig.text(0.60, 0.98, "Mann-Whitney one-vs-rest stars: BH-FDR", fontsize=7, va='top')

        safe = pathway.replace('/', '_').replace('\\', '_').replace(' ', '_')
        out = os.path.join(output_dir, f"sc_group_enrichment_{safe}.pdf")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def consistency_ranking_to_html(
    results,
    output_path,
    top_n=30,
    fdr_threshold=0.05,
    title='',
):
    """Generate a standalone HTML file with consistency ranking bars.

    Parameters
    ----------
    results : pd.DataFrame
        From SingleCellPAGE.run(), with 'pathway', 'consistency', 'FDR', 'p-value'.
    output_path : str
        Path to write the HTML file.
    top_n : int
        Number of top pathways to display.
    fdr_threshold : float
        FDR threshold for significance coloring.
    title : str
        Page title.
    """
    df = results.sort_values('consistency', ascending=False).head(top_n)
    max_c = df['consistency'].max() if len(df) > 0 else 1.0
    if max_c <= 0:
        max_c = 1.0

    html_parts = []
    html_parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html_parts.append(f"<title>{title or 'pyPAGE-SC Consistency Ranking'}</title>")
    html_parts.append("""<style>
body { font-family: Arial, sans-serif; margin: 20px; }
h1 { font-size: 1.3em; }
table { border-collapse: collapse; width: auto; }
td, th { padding: 4px 8px; text-align: left; font-size: 12px; border: 1px solid #ddd; }
th { background: #f5f5f5; }
.bar-cell { width: 300px; position: relative; }
.bar { height: 18px; display: inline-block; border-radius: 2px; }
.bar-sig { background: #2166ac; }
.bar-ns { background: #d6604d; }
#tooltip { position: fixed; background: #333; color: #fff; padding: 6px 10px;
  border-radius: 4px; font-size: 11px; pointer-events: none; display: none; z-index: 999; }
.legend { margin-top: 14px; font-size: 12px; }
.legend-box { display: inline-block; width: 18px; height: 12px; margin-right: 4px; vertical-align: middle; border: 1px solid #aaa; }
</style></head><body>""")
    if title:
        html_parts.append(f"<h1>{title}</h1>")

    html_parts.append("""<div id="tooltip"></div>
<script>
var tip = document.getElementById('tooltip');
function showTip(e, text) { tip.style.display='block'; tip.innerHTML=text;
  tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px'; }
function hideTip() { tip.style.display='none'; }
</script>""")

    html_parts.append("<table><tr><th>Pathway</th><th>Consistency</th><th>FDR</th></tr>")

    for _, row in df.iterrows():
        pw = row['pathway']
        c_val = row['consistency']
        fdr_val = row.get('FDR', 1.0)
        p_val = row.get('p-value', float('nan'))
        is_sig = fdr_val < fdr_threshold
        bar_class = 'bar-sig' if is_sig else 'bar-ns'
        bar_width = max(0, c_val / max_c * 100)
        tip_text = (f"{pw}<br>Consistency: {c_val:.4f}<br>"
                    f"p-value: {p_val:.4g}<br>FDR: {fdr_val:.4g}")

        html_parts.append(
            f'<tr><td>{pw}</td>'
            f'<td class="bar-cell" onmousemove="showTip(event,\'{tip_text}\')" '
            f'onmouseout="hideTip()">'
            f'<span class="bar {bar_class}" style="width:{bar_width:.1f}%"></span> '
            f'{c_val:.4f}</td>'
            f'<td>{fdr_val:.4g}</td></tr>')

    html_parts.append("</table>")

    # Legend
    html_parts.append('<div class="legend">')
    html_parts.append(f'<span class="legend-box" style="background:#2166ac"></span> FDR &lt; {fdr_threshold} &nbsp;')
    html_parts.append(f'<span class="legend-box" style="background:#d6604d"></span> FDR &ge; {fdr_threshold}')
    html_parts.append('</div>')

    html_parts.append("</body></html>")

    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))


def _generate_cmap_lut(cmap_name='viridis', n=256):
    """Generate a colormap lookup table as a list of [R, G, B] ints.

    Uses matplotlib colormaps at build time so the HTML report
    has no runtime dependency on matplotlib.
    """
    from matplotlib.cm import get_cmap
    cmap = get_cmap(cmap_name, n)
    lut = []
    for i in range(n):
        r, g, b, _ = cmap(i / (n - 1))
        lut.append([int(r * 255), int(g * 255), int(b * 255)])
    return lut


def interactive_report_to_html(
    results,
    scores,
    pathway_names,
    embeddings,
    output_path,
    fdr_threshold=0.05,
    title='pyPAGE-SC Interactive Report',
    pathway_genes=None,
    metadata=None,
    groupby=None,
    cmap='viridis',
    group_stats_df=None,
    score_vmin=None,
    score_vmax=None,
):
    """Generate a self-contained interactive HTML report (VISION-like).

    Three-panel layout: sortable signature table (upper-left), detail panel
    with histogram/stats/gene table (lower-left), scatter plot with hover
    tooltips, metadata coloring, and split view (right).

    Parameters
    ----------
    results : pd.DataFrame
        From SingleCellPAGE.run(), with 'pathway', 'consistency', 'FDR'.
    scores : np.ndarray
        Shape (n_cells, n_pathways) with per-cell pathway scores.
    pathway_names : array-like
        Pathway names matching columns of scores.
    embeddings : dict
        Mapping of embedding name (e.g. 'X_umap') to (n_cells, 2) arrays.
    output_path : str
        Path to write the HTML file.
    fdr_threshold : float
        FDR threshold for significance highlighting.
    title : str
        Report title.
    pathway_genes : dict, optional
        Mapping of pathway name to list of gene names.
    metadata : dict, optional
        Mapping of column name to list of values (one per cell, categorical).
    groupby : str, optional
        Preferred metadata column for group-enrichment bars.
    """
    pathway_names = list(pathway_names)
    n_cells = scores.shape[0]

    # Build pathway metadata sorted by consistency
    df = results.sort_values('consistency', ascending=False)
    pathways_json = []
    for _, row in df.iterrows():
        pw_name = row['pathway']
        genes_list = []
        if pathway_genes and pw_name in pathway_genes:
            genes_list = sorted(pathway_genes[pw_name])
        pathways_json.append({
            'name': pw_name,
            'consistency': round(float(row['consistency']), 6),
            'pvalue': round(float(row.get('p-value', float('nan'))), 6),
            'fdr': round(float(row.get('FDR', float('nan'))), 6),
            'genes': genes_list,
            'nGenes': len(genes_list),
        })

    # Build embeddings dict with rounded coordinates
    embeddings_json = {}
    for key, coords in embeddings.items():
        coords = np.asarray(coords)
        embeddings_json[key] = {
            'x': np.round(coords[:, 0], 4).tolist(),
            'y': np.round(coords[:, 1], 4).tolist(),
        }

    # Build scores dict (only for pathways in results)
    pw_name_to_idx = {name: i for i, name in enumerate(pathway_names)}
    scores_json = {}
    for pw_info in pathways_json:
        name = pw_info['name']
        if name in pw_name_to_idx:
            idx = pw_name_to_idx[name]
            scores_json[name] = np.round(scores[:, idx], 4).tolist()

    # Build metadata dict with integer encoding for categoricals
    metadata_json = {}
    if metadata:
        for col_name, values in metadata.items():
            str_vals = [str(v) for v in values]
            categories = sorted(set(str_vals))
            cat_to_idx = {c: i for i, c in enumerate(categories)}
            metadata_json[col_name] = {
                'values': [cat_to_idx[v] for v in str_vals],
                'categories': categories,
            }
    groupby_preferred = None
    if groupby and groupby in metadata_json:
        groupby_preferred = groupby
    elif metadata_json:
        groupby_preferred = sorted(metadata_json.keys())[0]

    # Colormap LUT
    viridis_lut = _generate_cmap_lut(cmap_name=cmap, n=256)

    group_stats = {}
    if group_stats_df is not None and len(group_stats_df) > 0:
        for _, row in group_stats_df.iterrows():
            pw = str(row.get('pathway', ''))
            gb = str(row.get('group_name', ''))
            grp = str(row.get('group', ''))
            if not pw or not gb or not grp:
                continue
            group_stats.setdefault(pw, {}).setdefault(gb, {"kw_p_value": None, "groups": {}})
            kw = row.get('kw_p_value', np.nan)
            if np.isfinite(kw):
                group_stats[pw][gb]["kw_p_value"] = float(kw)
            qv = row.get('one_vs_rest_fdr', np.nan)
            if np.isfinite(qv):
                group_stats[pw][gb]["groups"][grp] = float(qv)

    data_obj = {
        'pathways': pathways_json,
        'embeddings': embeddings_json,
        'scores': scores_json,
        'metadata': metadata_json,
        'groupby_preferred': groupby_preferred,
        'group_stats': group_stats,
        'score_vmin': None if score_vmin is None else float(score_vmin),
        'score_vmax': None if score_vmax is None else float(score_vmax),
        'n_cells': n_cells,
        'fdr_threshold': fdr_threshold,
    }

    data_json = json.dumps(data_obj, separators=(',', ':'))
    viridis_json = json.dumps(viridis_lut)

    embedding_keys = list(embeddings.keys())
    default_embedding = embedding_keys[0] if embedding_keys else ''

    html = _REPORT_TEMPLATE.format(
        title=title,
        n_cells=f"{n_cells:,}",
        n_pathways=len(pathways_json),
        data_json=data_json,
        viridis_json=viridis_json,
        emb_keys_json=json.dumps(embedding_keys),
        default_embedding=default_embedding,
    )

    with open(output_path, 'w') as f:
        f.write(html)


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#f8f9fa;color:#333;height:100vh;display:flex;flex-direction:column}}

.title-bar{{background:#1a1a2e;color:#fff;padding:10px 20px;font-size:16px;
  font-weight:600;flex-shrink:0;display:flex;align-items:center;justify-content:space-between}}
.title-bar .sub{{font-size:12px;color:#aaa;font-weight:400}}

.main{{display:flex;flex:1;overflow:hidden}}

/* Left panel */
.left-panel{{width:340px;min-width:280px;display:flex;flex-direction:column;
  border-right:1px solid #ddd;background:#fff;flex-shrink:0}}

/* Upper-left: signature table */
.sig-panel{{flex:1;display:flex;flex-direction:column;border-bottom:1px solid #ddd;min-height:0}}
.sig-header{{padding:8px 10px;border-bottom:1px solid #eee}}
.search-input{{width:100%;padding:6px 10px;border:1px solid #ccc;border-radius:4px;
  font-size:13px;outline:none}}
.search-input:focus{{border-color:#4a90d9}}
.sig-table-wrap{{flex:1;overflow-y:auto;overflow-x:hidden}}
table.sig-table{{width:100%;border-collapse:collapse}}
.sig-table th{{position:sticky;top:0;background:#f5f5f5;border-bottom:2px solid #ddd;
  padding:5px 8px;font-size:11px;cursor:pointer;user-select:none;white-space:nowrap;text-align:left}}
.sig-table th:hover{{background:#e8e8e8}}
.sig-table th .arrow{{font-size:9px;margin-left:3px;color:#888}}
.sig-table td{{padding:5px 8px;font-size:12px;border-bottom:1px solid #f0f0f0;cursor:pointer}}
.sig-table tr:hover td{{background:#f0f4ff}}
.sig-table tr.active td{{background:#e3ecf7;font-weight:600}}
.sig-table tr.sig td:first-child{{color:#2166ac}}
.sig-table td.num{{text-align:right;font-variant-numeric:tabular-nums}}

/* Lower-left: detail panel */
.detail-panel{{height:280px;min-height:200px;display:flex;flex-direction:column;
  overflow-y:auto;padding:10px;background:#fafafa}}
.detail-panel.empty{{display:flex;align-items:center;justify-content:center;color:#999;font-size:13px}}
.detail-title{{font-size:13px;font-weight:600;margin-bottom:8px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.hist-canvas{{width:100%;height:100px;border:1px solid #ddd;border-radius:3px;
  background:#fff;margin-bottom:8px}}
.stats-box{{display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;font-size:11px;
  margin-bottom:8px;padding:6px 8px;background:#fff;border:1px solid #eee;border-radius:3px}}
.stats-box .lbl{{color:#888}}
.stats-box .val{{text-align:right;font-variant-numeric:tabular-nums}}
.gene-section{{flex:1;min-height:0;display:flex;flex-direction:column}}
.gene-section h4{{font-size:11px;color:#666;margin-bottom:4px}}
.gene-list{{flex:1;overflow-y:auto;font-size:11px;padding:4px 6px;background:#fff;
  border:1px solid #eee;border-radius:3px;max-height:120px;column-count:2;column-gap:10px}}
.gene-list span{{display:block;padding:1px 0}}

/* Right panel */
.right-panel{{flex:1;display:flex;flex-direction:column;padding:10px;overflow:hidden}}
.controls{{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap;flex-shrink:0}}
.tab-btn{{padding:4px 12px;border:1px solid #ccc;border-radius:4px 4px 0 0;
  background:#eee;cursor:pointer;font-size:12px}}
.tab-btn.active{{background:#fff;border-bottom-color:#fff;font-weight:600}}
.color-select{{padding:4px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px;
  background:#fff;max-width:200px}}
.split-btn{{padding:4px 10px;border:1px solid #ccc;border-radius:4px;font-size:11px;
  background:#fff;cursor:pointer}}
.split-btn.active{{background:#e3ecf7;border-color:#4a90d9}}

.scatter-area{{flex:1;display:flex;gap:6px;min-height:0}}
.scatter-col{{flex:1;display:flex;flex-direction:column;min-width:0}}
.scatter-col .color-select{{margin-bottom:4px}}
.canvas-wrap{{flex:1;position:relative;background:#fff;border:1px solid #ddd;border-radius:4px;overflow:hidden}}
canvas.scatter{{display:block}}

.legend-bar{{display:flex;align-items:center;gap:6px;padding:6px 0;flex-shrink:0;min-height:24px;flex-wrap:wrap}}
.legend-grad{{width:180px;height:12px;border:1px solid #ccc;border-radius:2px}}
.legend-lbl{{font-size:11px;color:#666}}
.cat-legend{{display:flex;flex-wrap:wrap;gap:4px 10px;font-size:11px}}
.cat-legend-item{{display:flex;align-items:center;gap:3px}}
.cat-swatch{{width:10px;height:10px;border-radius:2px;border:1px solid rgba(0,0,0,0.15)}}
.group-panel{{margin-top:6px;border:1px solid #ddd;border-radius:4px;background:#fff;padding:8px;
  flex-shrink:0}}
.group-controls{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}}
.group-controls label{{font-size:12px;color:#555}}
.group-select{{padding:4px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px;background:#fff}}
.group-canvas{{width:100%;height:210px;border:1px solid #ddd;border-radius:3px;background:#fff;display:block}}

/* Tooltip */
.tooltip{{position:fixed;background:rgba(30,30,30,0.92);color:#fff;padding:6px 10px;
  border-radius:4px;font-size:11px;pointer-events:none;display:none;z-index:999;
  max-width:280px;line-height:1.4}}
</style>
</head>
<body>

<div class="title-bar">
  <span>{title}</span>
  <span class="sub">pyPAGE single-cell &middot; {n_cells} cells &middot; {n_pathways} pathways</span>
</div>

<div class="main">
  <div class="left-panel">
    <div class="sig-panel">
      <div class="sig-header">
        <input type="text" class="search-input" id="search" placeholder="Search pathways..." autocomplete="off">
      </div>
      <div class="sig-table-wrap">
        <table class="sig-table">
          <thead><tr>
            <th data-col="name">Name <span class="arrow"></span></th>
            <th data-col="consistency">C&#x2032; <span class="arrow"></span></th>
            <th data-col="fdr">FDR <span class="arrow"></span></th>
          </tr></thead>
          <tbody id="sigBody"></tbody>
        </table>
      </div>
    </div>
    <div class="detail-panel empty" id="detailPanel">
      Select a pathway from the table above.
    </div>
  </div>

  <div class="right-panel">
    <div class="controls" id="controls">
      <div id="tabBar" style="display:flex;gap:4px"></div>
      <label style="font-size:12px;margin-left:8px">Color by:</label>
      <select class="color-select" id="colorBy"></select>
      <label style="font-size:12px">vmin:</label>
      <input class="color-select" id="colorVmin" type="number" step="any" placeholder="auto" style="max-width:95px">
      <label style="font-size:12px">vmax:</label>
      <input class="color-select" id="colorVmax" type="number" step="any" placeholder="auto" style="max-width:95px">
      <button class="split-btn" id="rangeAutoBtn" title="Reset color range to auto">Auto range</button>
      <button class="split-btn" id="splitBtn" title="Split view">Split</button>
    </div>
    <div class="scatter-area" id="scatterArea">
      <div class="scatter-col" id="col0">
        <div class="canvas-wrap" id="wrap0"><canvas class="scatter" id="cv0"></canvas></div>
      </div>
    </div>
    <div class="legend-bar" id="legendBar"></div>
    <div class="group-panel" id="groupPanel">
      <div class="group-controls">
        <label>Group by:</label>
        <select class="group-select" id="groupBySelect"></select>
        <label>Metric:</label>
        <select class="group-select" id="groupMetric">
          <option value="fraction">High-score fraction (>= P75)</option>
          <option value="mean">Mean pathway score</option>
        </select>
      </div>
      <canvas class="group-canvas" id="groupCanvas"></canvas>
    </div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
(function(){{
"use strict";
var DATA={data_json};
var VIRIDIS={viridis_json};
var embKeys={emb_keys_json};
var curEmb=embKeys[0]||"";

// Categorical palette (20 distinct colors, d3-like)
var CAT_COLORS=[
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
  "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5"];

var curPw=null;
var sortCol="consistency",sortAsc=false;
var splitMode=false;
var colorBy=["_pw","_pw"];  // per-panel color source
var dpr=window.devicePixelRatio||1;

// Build color-by options
var colorOptions=[];
colorOptions.push({{value:"_pw",label:"Selected pathway"}});
DATA.pathways.forEach(function(pw){{
  colorOptions.push({{value:"pw:"+pw.name,label:pw.name}});
}});
var metaKeys=Object.keys(DATA.metadata||{{}});
metaKeys.forEach(function(k){{
  colorOptions.push({{value:"meta:"+k,label:k+" (metadata)"}});
}});

// DOM
var sigBody=document.getElementById("sigBody");
var searchInput=document.getElementById("search");
var tabBar=document.getElementById("tabBar");
var detailPanel=document.getElementById("detailPanel");
var legendBar=document.getElementById("legendBar");
var scatterArea=document.getElementById("scatterArea");
var splitBtn=document.getElementById("splitBtn");
var colorBySelect=document.getElementById("colorBy");
var colorVmin=document.getElementById("colorVmin");
var colorVmax=document.getElementById("colorVmax");
var rangeAutoBtn=document.getElementById("rangeAutoBtn");
var tooltip=document.getElementById("tooltip");
var groupPanel=document.getElementById("groupPanel");
var groupBySelect=document.getElementById("groupBySelect");
var groupMetric=document.getElementById("groupMetric");
var groupCanvas=document.getElementById("groupCanvas");

// Populate color-by dropdown
function fillColorSelect(sel,idx){{
  sel.innerHTML="";
  colorOptions.forEach(function(o){{
    var opt=document.createElement("option");
    opt.value=o.value;opt.textContent=o.label;
    sel.appendChild(opt);
  }});
  sel.value=colorBy[idx];
  sel.onchange=function(){{colorBy[idx]=sel.value;drawAll();}};
}}
fillColorSelect(colorBySelect,0);
if(colorVmin && DATA.score_vmin !== null && DATA.score_vmin !== undefined){{
  colorVmin.value = Number(DATA.score_vmin);
}}
if(colorVmax && DATA.score_vmax !== null && DATA.score_vmax !== undefined){{
  colorVmax.value = Number(DATA.score_vmax);
}}

// Embedding tabs
embKeys.forEach(function(key){{
  var btn=document.createElement("div");
  btn.className="tab-btn"+(key===curEmb?" active":"");
  btn.textContent=key.replace("X_","").toUpperCase();
  btn.onclick=function(){{
    curEmb=key;
    document.querySelectorAll(".tab-btn").forEach(function(b){{b.className="tab-btn"}});
    btn.className="tab-btn active";
    drawAll();
  }};
  tabBar.appendChild(btn);
}});

// Sortable signature table
var filteredPws=DATA.pathways.slice();

function sortPws(){{
  filteredPws.sort(function(a,b){{
    var va,vb;
    if(sortCol==="name"){{va=a.name.toLowerCase();vb=b.name.toLowerCase();return sortAsc?(va<vb?-1:va>vb?1:0):(va>vb?-1:va<vb?1:0);}}
    va=a[sortCol];vb=b[sortCol];
    return sortAsc?va-vb:vb-va;
  }});
}}

function buildTable(){{
  var lc=(searchInput.value||"").toLowerCase();
  filteredPws=DATA.pathways.filter(function(pw){{return !lc||pw.name.toLowerCase().indexOf(lc)!==-1;}});
  sortPws();
  sigBody.innerHTML="";
  filteredPws.forEach(function(pw){{
    var tr=document.createElement("tr");
    if(pw.fdr<DATA.fdr_threshold)tr.className="sig";
    if(curPw===pw.name)tr.className+=" active";
    tr.innerHTML='<td title="'+pw.name+'">'+pw.name+'</td>'+
      '<td class="num">'+pw.consistency.toFixed(4)+'</td>'+
      '<td class="num">'+(pw.fdr<0.0001?pw.fdr.toExponential(2):pw.fdr.toFixed(4))+'</td>';
    tr.onclick=function(){{selectPw(pw.name);}};
    sigBody.appendChild(tr);
  }});
  // Update sort arrows
  document.querySelectorAll(".sig-table th").forEach(function(th){{
    var col=th.getAttribute("data-col");
    var arrow=th.querySelector(".arrow");
    if(col===sortCol)arrow.textContent=sortAsc?"\u25B2":"\u25BC";
    else arrow.textContent="";
  }});
}}

document.querySelectorAll(".sig-table th").forEach(function(th){{
  th.onclick=function(){{
    var col=th.getAttribute("data-col");
    if(sortCol===col)sortAsc=!sortAsc;
    else{{sortCol=col;sortAsc=col==="name";}}
    buildTable();
  }};
}});
searchInput.addEventListener("input",function(){{buildTable();}});
buildTable();

// Detail panel
var histCanvas=null;

function updateDetail(){{
  if(!curPw){{
    detailPanel.className="detail-panel empty";
    detailPanel.innerHTML="Select a pathway from the table above.";
    return;
  }}
  var pw=DATA.pathways.find(function(p){{return p.name===curPw;}});
  if(!pw)return;
  detailPanel.className="detail-panel";

  var vals=DATA.scores[curPw];
  var mean=0,min=Infinity,max=-Infinity;
  if(vals){{
    for(var i=0;i<vals.length;i++){{mean+=vals[i];if(vals[i]<min)min=vals[i];if(vals[i]>max)max=vals[i];}}
    mean/=vals.length;
  }}
  var med=0,std=0;
  if(vals){{
    var sorted=vals.slice().sort(function(a,b){{return a-b;}});
    med=sorted.length%2?sorted[Math.floor(sorted.length/2)]:(sorted[sorted.length/2-1]+sorted[sorted.length/2])/2;
    var ss=0;for(var i=0;i<vals.length;i++){{var d=vals[i]-mean;ss+=d*d;}}
    std=Math.sqrt(ss/vals.length);
  }}

  var sigLabel=pw.fdr<DATA.fdr_threshold?'<span style="color:#2166ac;font-weight:600">Significant</span>':'<span style="color:#d6604d">Not significant</span>';

  var h='<div class="detail-title" title="'+pw.name+'">'+pw.name+' '+sigLabel+'</div>';
  h+='<canvas class="hist-canvas" id="histCv"></canvas>';
  h+='<div class="stats-box">';
  h+='<span class="lbl">C\u2032</span><span class="val">'+pw.consistency.toFixed(4)+'</span>';
  h+='<span class="lbl">p-value</span><span class="val">'+pw.pvalue.toFixed(4)+'</span>';
  h+='<span class="lbl">FDR</span><span class="val">'+(pw.fdr<0.0001?pw.fdr.toExponential(2):pw.fdr.toFixed(4))+'</span>';
  h+='<span class="lbl"># genes</span><span class="val">'+pw.nGenes+'</span>';
  if(vals){{
    h+='<span class="lbl">Mean</span><span class="val">'+mean.toFixed(4)+'</span>';
    h+='<span class="lbl">Median</span><span class="val">'+med.toFixed(4)+'</span>';
    h+='<span class="lbl">Std</span><span class="val">'+std.toFixed(4)+'</span>';
    h+='<span class="lbl">Range</span><span class="val">'+min.toFixed(3)+' .. '+max.toFixed(3)+'</span>';
  }}
  h+='</div>';

  // Gene list
  h+='<div class="gene-section"><h4>Genes ('+pw.nGenes+')</h4><div class="gene-list">';
  if(pw.genes&&pw.genes.length>0){{
    pw.genes.forEach(function(g){{h+='<span>'+g+'</span>';}});
  }}else{{
    h+='<span style="color:#999">Gene list not available</span>';
  }}
  h+='</div></div>';

  detailPanel.innerHTML=h;

  // Draw histogram
  if(vals){{
    var cv=document.getElementById("histCv");
    if(cv)drawHistogram(cv,vals,min,max);
  }}
}}

function drawHistogram(cv,vals,vmin,vmax){{
  var rect=cv.getBoundingClientRect();
  cv.width=Math.floor(rect.width*dpr);
  cv.height=Math.floor(rect.height*dpr);
  cv.style.width=rect.width+"px";
  cv.style.height=rect.height+"px";
  var ctx2=cv.getContext("2d");
  ctx2.setTransform(dpr,0,0,dpr,0,0);
  var w=rect.width,h=rect.height;
  var nBins=30;
  var range=vmax-vmin||1;
  var bins=new Array(nBins).fill(0);
  for(var i=0;i<vals.length;i++){{
    var bi=Math.floor((vals[i]-vmin)/range*(nBins-0.001));
    if(bi<0)bi=0;if(bi>=nBins)bi=nBins-1;
    bins[bi]++;
  }}
  var maxCount=0;for(var i=0;i<nBins;i++)if(bins[i]>maxCount)maxCount=bins[i];
  if(maxCount===0)maxCount=1;
  var padL=4,padR=4,padT=4,padB=14;
  var bw=(w-padL-padR)/nBins;
  for(var i=0;i<nBins;i++){{
    var bh=bins[i]/maxCount*(h-padT-padB);
    var ci=Math.round(i/(nBins-1)*255);
    var c=VIRIDIS[ci];
    ctx2.fillStyle="rgb("+c[0]+","+c[1]+","+c[2]+")";
    ctx2.fillRect(padL+i*bw,h-padB-bh,bw-1,bh);
  }}
  ctx2.fillStyle="#888";ctx2.font="9px sans-serif";ctx2.textAlign="left";
  ctx2.fillText(vmin.toFixed(2),padL,h-1);
  ctx2.textAlign="right";
  ctx2.fillText(vmax.toFixed(2),w-padR,h-1);
}}

// Select pathway
function selectPw(name){{
  curPw=name;
  // If color-by is "_pw", keep it synced
  buildTable();
  updateDetail();
  drawAll();
}}

// Split view
splitBtn.onclick=function(){{
  splitMode=!splitMode;
  splitBtn.className="split-btn"+(splitMode?" active":"");
  setupScatterPanels();
  drawAll();
}};

var canvases=[],ctxs=[],wraps=[];
var grids=[];  // spatial grids for hover

function setupScatterPanels(){{
  scatterArea.innerHTML="";
  var nPanels=splitMode?2:1;
  canvases=[];ctxs=[];wraps=[];grids=[];
  for(var p=0;p<nPanels;p++){{
    var col=document.createElement("div");
    col.className="scatter-col";
    if(splitMode){{
      var sel=document.createElement("select");
      sel.className="color-select";
      (function(idx){{fillColorSelect(sel,idx)}})(p);
      col.appendChild(sel);
    }}
    var cw=document.createElement("div");
    cw.className="canvas-wrap";
    cw.id="wrap"+p;
    var cv=document.createElement("canvas");
    cv.className="scatter";cv.id="cv"+p;
    cw.appendChild(cv);
    col.appendChild(cw);
    scatterArea.appendChild(col);
    canvases.push(cv);
    ctxs.push(cv.getContext("2d"));
    wraps.push(cw);
    grids.push(null);

    // Hover events
    (function(panelIdx){{
      cw.addEventListener("mousemove",function(e){{onHover(e,panelIdx);}});
      cw.addEventListener("mouseleave",function(){{tooltip.style.display="none";}});
    }})(p);
  }}
  // reset colorBy for panels
  if(!splitMode)colorBy[0]=colorBySelect.value;
  if(splitMode&&colorBy.length<2)colorBy.push("_pw");
}}
setupScatterPanels();
initGroupBars();

// Color-by change for single mode
colorBySelect.onchange=function(){{colorBy[0]=colorBySelect.value;drawAll();}};
if(colorVmin) colorVmin.oninput=function(){{drawAll();}};
if(colorVmax) colorVmax.oninput=function(){{drawAll();}};
if(rangeAutoBtn) rangeAutoBtn.onclick=function(){{
  if(colorVmin) colorVmin.value="";
  if(colorVmax) colorVmax.value="";
  drawAll();
}};

function initGroupBars(){{
  if(!groupBySelect||!groupPanel) return;
  groupBySelect.innerHTML="";
  if(metaKeys.length===0){{
    groupPanel.style.display="none";
    return;
  }}
  metaKeys.forEach(function(k){{
    var opt=document.createElement("option");
    opt.value=k;
    opt.textContent=k;
    groupBySelect.appendChild(opt);
  }});
  var preferred=DATA.groupby_preferred;
  if(preferred&&metaKeys.indexOf(preferred)!==-1){{
    groupBySelect.value=preferred;
  }}else{{
    groupBySelect.value=metaKeys[0];
  }}
  groupBySelect.onchange=drawGroupBars;
  groupMetric.onchange=drawGroupBars;
}}

function drawGroupBars(){{
  if(!groupCanvas||!groupBySelect) return;
  var ctx=groupCanvas.getContext("2d");
  var rect=groupCanvas.getBoundingClientRect();
  var w=Math.max(10,Math.floor(rect.width));
  var h=Math.max(10,Math.floor(rect.height));
  groupCanvas.width=Math.floor(w*dpr);
  groupCanvas.height=Math.floor(h*dpr);
  groupCanvas.style.width=w+"px";
  groupCanvas.style.height=h+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);

  if(!curPw||!DATA.scores[curPw]){{
    ctx.fillStyle="#888";
    ctx.font="12px sans-serif";
    ctx.fillText("Select a pathway to show group enrichment.",10,22);
    return;
  }}
  var gb=groupBySelect.value;
  var md=DATA.metadata[gb];
  if(!md||!md.values||!md.categories){{
    ctx.fillStyle="#888";
    ctx.font="12px sans-serif";
    ctx.fillText("No metadata available for grouped bars.",10,22);
    return;
  }}
  var vals=DATA.scores[curPw];
  var n=Math.min(vals.length, md.values.length);
  if(n===0){{
    ctx.fillStyle="#888";
    ctx.font="12px sans-serif";
    ctx.fillText("No cells available for grouped bars.",10,22);
    return;
  }}

  var sortedVals=vals.slice(0,n).slice().sort(function(a,b){{return a-b;}});
  var qIdx=Math.floor(0.75*Math.max(0,sortedVals.length-1));
  var thr=sortedVals[qIdx];
  var agg={{}};
  for(var i=0;i<n;i++){{
    var gid=md.values[i];
    if(agg[gid]===undefined)agg[gid]={{n:0,pos:0,sum:0}};
    var a=agg[gid];
    var v=vals[i];
    a.n+=1;
    if(v>=thr)a.pos+=1;
    a.sum+=v;
  }}

  var rows=[];
  Object.keys(agg).forEach(function(k){{
    var gid=parseInt(k,10);
    var a=agg[k];
    var label=(gid>=0&&gid<md.categories.length)?md.categories[gid]:String(gid);
    rows.push({{
      label:label,
      n:a.n,
      fraction:a.n>0?(a.pos/a.n):0,
      mean:a.n>0?(a.sum/a.n):0
    }});
  }});
  if(rows.length===0){{
    ctx.fillStyle="#888";
    ctx.font="12px sans-serif";
    ctx.fillText("No groups available.",10,22);
    return;
  }}

  var metric=(groupMetric&&groupMetric.value==="mean")?"mean":"fraction";
  var pathwayStats=null;
  if(DATA.group_stats && DATA.group_stats[curPw] && DATA.group_stats[curPw][gb]){{
    pathwayStats=DATA.group_stats[curPw][gb];
  }}
  rows.sort(function(a,b){{return b[metric]-a[metric];}});
  rows=rows.slice(0,40);

  var padL=58,padR=12,padT=16,padB=72;
  var pw=w-padL-padR,ph=h-padT-padB;
  if(pw<=10||ph<=10)return;
  var yMin=(metric==="fraction")?0:Math.min.apply(null,rows.map(function(r){{return r.mean;}}));
  var yMax=Math.max.apply(null,rows.map(function(r){{return metric==="fraction"?r.fraction:r.mean;}}));
  if(yMax<=yMin)yMax=yMin+1e-9;
  var bw=pw/rows.length;
  ctx.strokeStyle="#888";
  ctx.beginPath();
  ctx.moveTo(padL,padT+ph);
  ctx.lineTo(padL+pw,padT+ph);
  ctx.stroke();
  ctx.fillStyle="#222";
  ctx.font="10px sans-serif";
  ctx.textAlign="right";
  ctx.fillText(yMax.toFixed(metric==="fraction"?2:3), padL-4, padT+9);
  ctx.fillText(yMin.toFixed(metric==="fraction"?2:3), padL-4, padT+ph);

  rows.forEach(function(r,idx){{
    var val=(metric==="fraction")?r.fraction:r.mean;
    var t=(val-yMin)/(yMax-yMin);
    var bh=Math.max(1,ph*t);
    var x=padL+idx*bw+bw*0.12;
    var y=padT+ph-bh;
    var rw=Math.max(1,bw*0.76);
    ctx.fillStyle=(metric==="fraction")?"#2f6da6":"#3a9d5d";
    ctx.fillRect(x,y,rw,bh);
    if(metric==="mean" && pathwayStats && pathwayStats.groups){{
      var qv = pathwayStats.groups[String(r.label)];
      if(qv !== undefined && !isNaN(qv)){{
        var star="";
        if(qv<1e-3)star="***";
        else if(qv<1e-2)star="**";
        else if(qv<5e-2)star="*";
        if(star){{
          ctx.fillStyle="#111";
          ctx.font="10px sans-serif";
          ctx.textAlign="center";
          ctx.fillText(star, x+rw*0.5, y-4);
        }}
      }}
    }}
    ctx.save();
    ctx.translate(x+rw*0.5,padT+ph+6);
    ctx.rotate(-Math.PI/2);
    ctx.textAlign="left";
    ctx.fillStyle="#333";
    ctx.fillText(r.label,0,0);
    ctx.restore();
  }});
  ctx.fillStyle="#111";
  ctx.font="11px sans-serif";
  ctx.textAlign="left";
  ctx.fillText(
    gb+" \u2022 "+(metric==="fraction"?"High-score fraction (>=P75)":"Mean pathway score"),
    padL,
    h-10
  );
  if(pathwayStats && pathwayStats.kw_p_value !== null && pathwayStats.kw_p_value !== undefined){{
    ctx.fillStyle="#444";
    ctx.textAlign="right";
    ctx.fillText("KW p="+Number(pathwayStats.kw_p_value).toExponential(2), w-padR, h-10);
  }}
}}

// Resolve which values+colors to use for a color-by key
function resolveColor(key){{
  if(key==="_pw"){{
    if(!curPw||!DATA.scores[curPw])return null;
    return {{type:"continuous",vals:DATA.scores[curPw]}};
  }}
  if(key.indexOf("pw:")===0){{
    var pn=key.substring(3);
    if(!DATA.scores[pn])return null;
    return {{type:"continuous",vals:DATA.scores[pn]}};
  }}
  if(key.indexOf("meta:")===0){{
    var mn=key.substring(5);
    var md=DATA.metadata[mn];
    if(!md)return null;
    return {{type:"categorical",vals:md.values,categories:md.categories}};
  }}
  return null;
}}

// Resize and draw a single scatter panel
function drawPanel(idx){{
  var cv=canvases[idx],ctx=ctxs[idx],wrap=wraps[idx];
  var rect=wrap.getBoundingClientRect();
  cv.width=Math.floor(rect.width*dpr);
  cv.height=Math.floor(rect.height*dpr);
  cv.style.width=rect.width+"px";
  cv.style.height=rect.height+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0);
  var w=rect.width,h=rect.height;
  ctx.clearRect(0,0,w,h);

  if(!curEmb||!DATA.embeddings[curEmb]){{
    ctx.fillStyle="#999";ctx.font="14px sans-serif";ctx.textAlign="center";
    ctx.fillText("No embedding available",w/2,h/2);
    grids[idx]=null;return;
  }}

  var emb=DATA.embeddings[curEmb];
  var xs=emb.x,ys=emb.y,n=xs.length;

  var xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
  for(var i=0;i<n;i++){{
    if(xs[i]<xmin)xmin=xs[i];if(xs[i]>xmax)xmax=xs[i];
    if(ys[i]<ymin)ymin=ys[i];if(ys[i]>ymax)ymax=ys[i];
  }}
  var pad=20,xrange=xmax-xmin||1,yrange=ymax-ymin||1;
  var scale=Math.min((w-2*pad)/xrange,(h-2*pad)/yrange);
  var ox=pad+((w-2*pad)-xrange*scale)/2;
  var oy=pad+((h-2*pad)-yrange*scale)/2;
  var r=Math.max(1,Math.min(4,800/Math.sqrt(n)));

  var colorInfo=resolveColor(splitMode?colorBy[idx]:colorBy[0]);

  // Build spatial grid for hover
  var gridSize=50,cellW=w/gridSize,cellH=h/gridSize;
  var grid=new Array(gridSize*gridSize);
  for(var i=0;i<grid.length;i++)grid[i]=[];

  // Compute screen positions
  var px_arr=new Float32Array(n),py_arr=new Float32Array(n);
  for(var i=0;i<n;i++){{
    px_arr[i]=ox+(xs[i]-xmin)*scale;
    py_arr[i]=oy+(ys[i]-ymin)*scale;
    var gx=Math.floor(px_arr[i]/cellW),gy=Math.floor(py_arr[i]/cellH);
    if(gx>=0&&gx<gridSize&&gy>=0&&gy<gridSize)grid[gy*gridSize+gx].push(i);
  }}
  grids[idx]={{grid:grid,gridSize:gridSize,cellW:cellW,cellH:cellH,px:px_arr,py:py_arr}};

  if(!colorInfo){{
    ctx.fillStyle="rgba(150,150,150,0.4)";
    for(var i=0;i<n;i++){{ctx.beginPath();ctx.arc(px_arr[i],py_arr[i],r,0,6.283);ctx.fill();}}
    return;
  }}

  if(colorInfo.type==="continuous"){{
    var vals=colorInfo.vals;
    var dataMin=Infinity,dataMax=-Infinity;
    for(var i=0;i<n;i++){{if(vals[i]<dataMin)dataMin=vals[i];if(vals[i]>dataMax)dataMax=vals[i];}}
    var inMin = colorVmin ? parseFloat(colorVmin.value) : NaN;
    var inMax = colorVmax ? parseFloat(colorVmax.value) : NaN;
    var vmin = isFinite(inMin) ? inMin : dataMin;
    var vmax = isFinite(inMax) ? inMax : dataMax;
    if(vmax < vmin){{var tmp=vmin;vmin=vmax;vmax=tmp;}}
    var vrange=vmax-vmin||1;

    var order=new Array(n);
    for(var i=0;i<n;i++)order[i]=i;
    order.sort(function(a,b){{return vals[a]-vals[b];}});

    for(var j=0;j<n;j++){{
      var i=order[j];
      var ci=Math.round((vals[i]-vmin)/vrange*255);
      if(ci<0)ci=0;if(ci>255)ci=255;
      var c=VIRIDIS[ci];
      ctx.fillStyle="rgb("+c[0]+","+c[1]+","+c[2]+")";
      ctx.beginPath();ctx.arc(px_arr[i],py_arr[i],r,0,6.283);ctx.fill();
    }}
    return {{type:"continuous",vmin:vmin,vmax:vmax,dataMin:dataMin,dataMax:dataMax}};
  }}

  if(colorInfo.type==="categorical"){{
    var cvals=colorInfo.vals,cats=colorInfo.categories;
    for(var i=0;i<n;i++){{
      ctx.fillStyle=CAT_COLORS[cvals[i]%CAT_COLORS.length];
      ctx.beginPath();ctx.arc(px_arr[i],py_arr[i],r,0,6.283);ctx.fill();
    }}
    return {{type:"categorical",categories:cats}};
  }}
}}

function drawAll(){{
  var legendInfos=[];
  for(var i=0;i<canvases.length;i++){{
    var info=drawPanel(i);
    legendInfos.push(info);
  }}
  updateLegend(legendInfos);
  drawGroupBars();
}}

function updateLegend(infos){{
  legendBar.innerHTML="";
  var info=infos[0]||null;
  // In split mode show labels
  for(var p=0;p<infos.length;p++){{
    var inf=infos[p];
    if(!inf)continue;
    if(splitMode){{
      var lbl=document.createElement("span");
      lbl.className="legend-lbl";lbl.style.fontWeight="600";
      lbl.textContent="Panel "+(p+1)+":";
      legendBar.appendChild(lbl);
    }}
    if(inf.type==="continuous"){{
      var minL=document.createElement("span");minL.className="legend-lbl";
      minL.textContent=inf.vmin.toFixed(4);
      var cv=document.createElement("canvas");cv.className="legend-grad";
      cv.width=180;cv.height=12;cv.style.width="180px";cv.style.height="12px";
      var gctx=cv.getContext("2d");
      for(var i=0;i<180;i++){{
        var ci=Math.round(i/179*255);var c=VIRIDIS[ci];
        gctx.fillStyle="rgb("+c[0]+","+c[1]+","+c[2]+")";
        gctx.fillRect(i,0,1,12);
      }}
      var maxL=document.createElement("span");maxL.className="legend-lbl";
      maxL.textContent=inf.vmax.toFixed(4);
      legendBar.appendChild(minL);legendBar.appendChild(cv);legendBar.appendChild(maxL);
    }}
    if(inf.type==="categorical"){{
      var cDiv=document.createElement("div");cDiv.className="cat-legend";
      inf.categories.forEach(function(cat,ci){{
        var item=document.createElement("span");item.className="cat-legend-item";
        item.innerHTML='<span class="cat-swatch" style="background:'+CAT_COLORS[ci%CAT_COLORS.length]+'"></span>'+cat;
        cDiv.appendChild(item);
      }});
      legendBar.appendChild(cDiv);
    }}
  }}
}}

// Hover tooltip with spatial grid
var hoverRaf=false;

function onHover(e,panelIdx){{
  if(hoverRaf)return;
  hoverRaf=true;
  requestAnimationFrame(function(){{
    hoverRaf=false;
    var g=grids[panelIdx];
    if(!g){{tooltip.style.display="none";return;}}
    var rect=wraps[panelIdx].getBoundingClientRect();
    var mx=e.clientX-rect.left,my=e.clientY-rect.top;
    var gx=Math.floor(mx/g.cellW),gy=Math.floor(my/g.cellH);
    var bestDist=64,bestIdx=-1;  // 8px threshold squared
    for(var dx=-1;dx<=1;dx++)for(var dy=-1;dy<=1;dy++){{
      var cx=gx+dx,cy=gy+dy;
      if(cx<0||cx>=g.gridSize||cy<0||cy>=g.gridSize)continue;
      var bucket=g.grid[cy*g.gridSize+cx];
      for(var k=0;k<bucket.length;k++){{
        var i=bucket[k];
        var dd=(g.px[i]-mx)*(g.px[i]-mx)+(g.py[i]-my)*(g.py[i]-my);
        if(dd<bestDist){{bestDist=dd;bestIdx=i;}}
      }}
    }}
    if(bestIdx<0){{tooltip.style.display="none";return;}}
    var lines=["<b>Cell "+bestIdx+"</b>"];
    if(curPw&&DATA.scores[curPw])lines.push("Score: "+DATA.scores[curPw][bestIdx].toFixed(4));
    metaKeys.forEach(function(mk){{
      var md=DATA.metadata[mk];
      lines.push(mk+": "+md.categories[md.values[bestIdx]]);
    }});
    tooltip.innerHTML=lines.join("<br>");
    tooltip.style.display="block";
    tooltip.style.left=(e.clientX+14)+"px";
    tooltip.style.top=(e.clientY+14)+"px";
  }});
}}

// Auto-select first pathway
if(DATA.pathways.length>0)selectPw(DATA.pathways[0].name);

window.addEventListener("resize",drawAll);
}})();
</script>
</body>
</html>"""
