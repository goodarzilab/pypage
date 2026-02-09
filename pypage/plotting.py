"""Visualization functions for single-cell PAGE results."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List


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


def _generate_viridis_lut(n=256):
    """Generate a viridis colormap lookup table as a list of [R, G, B] ints.

    Uses matplotlib's viridis colormap at build time so the HTML report
    has no runtime dependency on matplotlib.
    """
    from matplotlib.cm import get_cmap
    cmap = get_cmap('viridis', n)
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
):
    """Generate a self-contained interactive HTML report (VISION-like).

    The report includes a sidebar with clickable pathway list, a canvas
    scatter plot colored by per-cell pathway scores, embedding tabs,
    search filtering, and a viridis color legend. No external dependencies.

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
    """
    pathway_names = list(pathway_names)
    n_cells = scores.shape[0]

    # Build pathway metadata sorted by consistency
    df = results.sort_values('consistency', ascending=False)
    pathways_json = []
    for _, row in df.iterrows():
        pathways_json.append({
            'name': row['pathway'],
            'consistency': round(float(row['consistency']), 6),
            'pvalue': round(float(row.get('p-value', float('nan'))), 6),
            'fdr': round(float(row.get('FDR', float('nan'))), 6),
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
            scores_json[name] = np.round(scores[:, idx], 6).tolist()

    # Viridis LUT
    viridis_lut = _generate_viridis_lut(256)

    data_obj = {
        'pathways': pathways_json,
        'embeddings': embeddings_json,
        'scores': scores_json,
        'n_cells': n_cells,
        'fdr_threshold': fdr_threshold,
    }

    data_json = json.dumps(data_obj, separators=(',', ':'))
    viridis_json = json.dumps(viridis_lut)

    embedding_keys = list(embeddings.keys())
    default_embedding = embedding_keys[0] if embedding_keys else ''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background:#f8f9fa; color:#333; height:100vh; display:flex; flex-direction:column; }}

/* Title bar */
.title-bar {{ background:#1a1a2e; color:#fff; padding:10px 20px; font-size:16px;
  font-weight:600; flex-shrink:0; display:flex; align-items:center; justify-content:space-between; }}
.title-bar .subtitle {{ font-size:12px; color:#aaa; font-weight:400; }}

/* Main layout */
.main {{ display:flex; flex:1; overflow:hidden; }}

/* Sidebar */
.sidebar {{ width:320px; min-width:260px; background:#fff; border-right:1px solid #ddd;
  display:flex; flex-direction:column; flex-shrink:0; }}
.sidebar-header {{ padding:10px 12px; border-bottom:1px solid #eee; }}
.search-input {{ width:100%; padding:7px 10px; border:1px solid #ccc; border-radius:4px;
  font-size:13px; outline:none; }}
.search-input:focus {{ border-color:#4a90d9; }}
.pathway-list {{ flex:1; overflow-y:auto; }}
.pw-item {{ padding:8px 12px; cursor:pointer; border-bottom:1px solid #f0f0f0;
  font-size:12px; display:flex; justify-content:space-between; align-items:center; }}
.pw-item:hover {{ background:#f0f4ff; }}
.pw-item.active {{ background:#e3ecf7; font-weight:600; }}
.pw-item.sig .pw-name {{ color:#2166ac; }}
.pw-name {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-right:8px; }}
.pw-stat {{ font-size:11px; color:#888; white-space:nowrap; }}

/* Content area */
.content {{ flex:1; display:flex; flex-direction:column; padding:12px; overflow:hidden; }}
.tab-bar {{ display:flex; gap:4px; margin-bottom:8px; flex-shrink:0; }}
.tab-btn {{ padding:5px 14px; border:1px solid #ccc; border-radius:4px 4px 0 0;
  background:#eee; cursor:pointer; font-size:12px; }}
.tab-btn.active {{ background:#fff; border-bottom-color:#fff; font-weight:600; }}
.canvas-wrap {{ flex:1; position:relative; background:#fff; border:1px solid #ddd; border-radius:4px;
  overflow:hidden; }}
canvas {{ display:block; }}

/* Legend */
.legend {{ display:flex; align-items:center; gap:8px; padding:8px 0; flex-shrink:0; }}
.legend-gradient {{ width:200px; height:14px; border:1px solid #ccc; border-radius:2px; }}
.legend-label {{ font-size:11px; color:#666; }}

/* Info panel */
.info-panel {{ font-size:12px; color:#555; padding:6px 0; flex-shrink:0; }}
.info-panel b {{ color:#333; }}

/* No embedding message */
.no-data {{ display:flex; align-items:center; justify-content:center; flex:1;
  color:#999; font-size:14px; }}
</style>
</head>
<body>

<div class="title-bar">
  <span>{title}</span>
  <span class="subtitle">pyPAGE single-cell &middot; {n_cells:,} cells &middot; {len(pathways_json)} pathways</span>
</div>

<div class="main">
  <div class="sidebar">
    <div class="sidebar-header">
      <input type="text" class="search-input" id="search" placeholder="Search pathways..." autocomplete="off">
    </div>
    <div class="pathway-list" id="pwList"></div>
  </div>
  <div class="content">
    <div class="tab-bar" id="tabBar"></div>
    <div class="info-panel" id="infoPanel">Select a pathway from the sidebar.</div>
    <div class="canvas-wrap" id="canvasWrap">
      <canvas id="scatter"></canvas>
    </div>
    <div class="legend" id="legendBar">
      <span class="legend-label" id="legendMin"></span>
      <canvas class="legend-gradient" id="legendGrad" width="200" height="14"></canvas>
      <span class="legend-label" id="legendMax"></span>
    </div>
  </div>
</div>

<script>
(function() {{
"use strict";
var DATA = {data_json};
var VIRIDIS = {viridis_json};
var embKeys = {json.dumps(embedding_keys)};
var curEmb = "{default_embedding}";
var curPw = null;
var dpr = window.devicePixelRatio || 1;

// DOM refs
var canvas = document.getElementById('scatter');
var ctx = canvas.getContext('2d');
var wrap = document.getElementById('canvasWrap');
var pwList = document.getElementById('pwList');
var searchInput = document.getElementById('search');
var tabBar = document.getElementById('tabBar');
var infoPanel = document.getElementById('infoPanel');
var legendMin = document.getElementById('legendMin');
var legendMax = document.getElementById('legendMax');
var legendGrad = document.getElementById('legendGrad');

// Draw legend gradient once
(function() {{
  var gctx = legendGrad.getContext('2d');
  for (var i = 0; i < 200; i++) {{
    var ci = Math.round(i / 199 * 255);
    var c = VIRIDIS[ci];
    gctx.fillStyle = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
    gctx.fillRect(i, 0, 1, 14);
  }}
}})();

// Build tab bar
embKeys.forEach(function(key) {{
  var btn = document.createElement('div');
  btn.className = 'tab-btn' + (key === curEmb ? ' active' : '');
  btn.textContent = key.replace('X_', '').toUpperCase();
  btn.onclick = function() {{
    curEmb = key;
    document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.className = 'tab-btn'; }});
    btn.className = 'tab-btn active';
    draw();
  }};
  tabBar.appendChild(btn);
}});

// Build pathway list
function buildList(filter) {{
  pwList.innerHTML = '';
  var lc = (filter || '').toLowerCase();
  DATA.pathways.forEach(function(pw) {{
    if (lc && pw.name.toLowerCase().indexOf(lc) === -1) return;
    var div = document.createElement('div');
    div.className = 'pw-item' + (pw.fdr < DATA.fdr_threshold ? ' sig' : '') +
      (curPw === pw.name ? ' active' : '');
    div.innerHTML = '<span class="pw-name" title="' + pw.name + '">' + pw.name + '</span>' +
      '<span class="pw-stat">C\\u2032=' + pw.consistency.toFixed(3) + '</span>';
    div.onclick = function() {{ selectPw(pw.name); }};
    pwList.appendChild(div);
  }});
}}
buildList('');

searchInput.addEventListener('input', function() {{ buildList(this.value); }});

function selectPw(name) {{
  curPw = name;
  buildList(searchInput.value);
  // Update info panel
  var pw = DATA.pathways.find(function(p) {{ return p.name === name; }});
  if (pw) {{
    var sigLabel = pw.fdr < DATA.fdr_threshold
      ? '<span style="color:#2166ac;font-weight:600">Significant</span>'
      : '<span style="color:#d6604d">Not significant</span>';
    infoPanel.innerHTML = '<b>' + pw.name + '</b> &nbsp; C\\u2032=' +
      pw.consistency.toFixed(4) + ' &nbsp; p=' + pw.pvalue.toFixed(4) +
      ' &nbsp; FDR=' + pw.fdr.toFixed(4) + ' &nbsp; ' + sigLabel;
  }}
  draw();
}}

// Resize canvas
function resizeCanvas() {{
  var rect = wrap.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}

function draw() {{
  resizeCanvas();
  var w = canvas.width / dpr;
  var h = canvas.height / dpr;
  ctx.clearRect(0, 0, w, h);

  if (!curEmb || !DATA.embeddings[curEmb]) {{
    ctx.fillStyle = '#999';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No embedding available', w / 2, h / 2);
    return;
  }}

  var emb = DATA.embeddings[curEmb];
  var xs = emb.x, ys = emb.y;
  var n = xs.length;

  // Compute bounds with padding
  var xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (var i = 0; i < n; i++) {{
    if (xs[i] < xmin) xmin = xs[i];
    if (xs[i] > xmax) xmax = xs[i];
    if (ys[i] < ymin) ymin = ys[i];
    if (ys[i] > ymax) ymax = ys[i];
  }}
  var pad = 20;
  var xrange = xmax - xmin || 1;
  var yrange = ymax - ymin || 1;
  var scale = Math.min((w - 2 * pad) / xrange, (h - 2 * pad) / yrange);
  var ox = pad + ((w - 2 * pad) - xrange * scale) / 2;
  var oy = pad + ((h - 2 * pad) - yrange * scale) / 2;

  // Point size based on cell count
  var r = Math.max(1, Math.min(4, 800 / Math.sqrt(n)));

  if (!curPw || !DATA.scores[curPw]) {{
    // Grey dots when no pathway selected
    ctx.fillStyle = 'rgba(150,150,150,0.4)';
    for (var i = 0; i < n; i++) {{
      var px = ox + (xs[i] - xmin) * scale;
      var py = oy + (ys[i] - ymin) * scale;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 6.283);
      ctx.fill();
    }}
    legendMin.textContent = '';
    legendMax.textContent = '';
    return;
  }}

  var vals = DATA.scores[curPw];
  var vmin = Infinity, vmax = -Infinity;
  for (var i = 0; i < n; i++) {{
    if (vals[i] < vmin) vmin = vals[i];
    if (vals[i] > vmax) vmax = vals[i];
  }}
  var vrange = vmax - vmin || 1;

  legendMin.textContent = vmin.toFixed(4);
  legendMax.textContent = vmax.toFixed(4);

  // Build sorted indices (low scores first so high scores draw on top)
  var order = new Array(n);
  for (var i = 0; i < n; i++) order[i] = i;
  order.sort(function(a, b) {{ return vals[a] - vals[b]; }});

  // Draw points
  for (var j = 0; j < n; j++) {{
    var i = order[j];
    var px = ox + (xs[i] - xmin) * scale;
    var py = oy + (ys[i] - ymin) * scale;
    var ci = Math.round((vals[i] - vmin) / vrange * 255);
    if (ci < 0) ci = 0;
    if (ci > 255) ci = 255;
    var c = VIRIDIS[ci];
    ctx.fillStyle = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
    ctx.beginPath();
    ctx.arc(px, py, r, 0, 6.283);
    ctx.fill();
  }}
}}

// Auto-select first pathway
if (DATA.pathways.length > 0) {{
  selectPw(DATA.pathways[0].name);
}}

window.addEventListener('resize', draw);
}})();
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
