import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib
from typing import Optional
import copy
from .io.accession_types import change_accessions


class Heatmap:
    """
    An object which is used to build a heatmap, representing pathway deregulation
    Attributes
    ==========
    pathways: np.ndarray
    graphical_ar: np.ndarray
    regulator_exp: np.ndarray
    cmap_main: str
    cmap_reg: str
    n_pathways: int
    n_bins: int
    isreg: bool
    bin_edges: np.ndarray or None
    fig: matplotlib.figure.Figure
    ax: matplotlib.figure.Axes
    ==========
    Methods
    ----------
    save
        saves a heatmap
    show
        shows a heatmap
    to_html
        saves a standalone HTML file with colored table and hover tooltips
    save_matrix
        saves enrichment matrix as TSV with metadata header
    from_matrix
        loads a Heatmap from a saved matrix TSV
    """
    def __init__(self,
                 pathways: np.ndarray,
                 graphical_ar: np.ndarray,
                 regulator_exp: Optional[np.ndarray] = None,
                 cmap_main: Optional[str] = 'viridis',
                 cmap_reg: Optional[str] = 'plasma',
                 bin_edges: Optional[np.ndarray] = None):
        self.pathways = pathways
        self.graphical_ar = graphical_ar
        self.regulator_exp = regulator_exp
        self.cmap_main = cmap_main
        self.cmap_reg = cmap_reg
        self.n_pathways = self.graphical_ar.shape[0]
        self.n_bins = self.graphical_ar.shape[1]
        self.expression = None
        self.genes = None
        self.isreg = False
        self.bin_edges = bin_edges
        self.is_continuous = bin_edges is not None

        self.regulator_names = [gene.split('_')[0] for gene in self.pathways]

    def _subset_and_sort_pathways(self,
                                  max_rows: int = 50) -> (np.ndarray, np.ndarray):
        """

        Parameters
        ----------
        max_rows
            maximum number of rows to use on a heatmap
        Returns
        -------

        """
        upregulated_mask = self.graphical_ar[:, :self.n_bins // 2].sum(1) <= self.graphical_ar[:, self.n_bins // 2:].sum(1)
        upregulated_idxs = np.where(upregulated_mask)[0]
        downregulated_idxs = np.where(~upregulated_mask)[0]
        if 0 < max_rows < self.n_pathways:
            additional_up = max(0, max_rows//2 - len(downregulated_idxs))
            additional_down = max(0, max_rows//2 - len(upregulated_idxs))
            upregulated_idxs = upregulated_idxs[:max_rows//2 + additional_up]
            downregulated_idxs = downregulated_idxs[:max_rows//2 + additional_down]
        sub_idxs = np.concatenate((upregulated_idxs, downregulated_idxs))

        sub_graphical_ar = self.graphical_ar[sub_idxs]
        sub_pathways = self.pathways[sub_idxs]
        if self.isreg:
            sub_regulator_exp = self.regulator_exp[sub_idxs]

        sort_idxs = np.arange(sub_graphical_ar.shape[0])
        for i in range(self.n_bins-2):
            sort_idxs = sorted(sort_idxs, key=lambda x: sum(sub_graphical_ar[x][i:i+3]), reverse=True)
        sorted_pathways = sub_pathways[sort_idxs]
        sorted_graphical_ar = sub_graphical_ar[sort_idxs]
        if self.isreg:
            sorted_regulator_exp = sub_regulator_exp[sort_idxs]
        else:
            sorted_regulator_exp = None

        return sorted_pathways, sorted_graphical_ar, sorted_regulator_exp

    def _draw_bin_edge_bar(self, ax_bar, n_bins):
        """Draw a continuous data bar above the heatmap showing expression range per bin."""
        edges = self.bin_edges
        cmap = matplotlib.cm.get_cmap('RdYlBu_r')
        norm = Normalize(vmin=edges[0], vmax=edges[-1])
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mid = (lo + hi) / 2.0
            color = cmap(norm(mid))
            ax_bar.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='white', linewidth=0.5))
            label = f"{lo:.2g}\n{hi:.2g}"
            ax_bar.text(i + 0.5, 0.5, label, ha='center', va='center', fontsize=6)
        ax_bar.set_xlim(0, n_bins)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        ax_bar.set_ylabel('Range', fontsize=7, rotation=0, labelpad=20, va='center')

    def _columnwise_heatmap(self,
                            graphical_ar: np.ndarray,
                            regulator_exp: np.ndarray,
                            max_val: int):
        self.ims = []
        if self.isreg:
            current_cmap = copy.copy(matplotlib.cm.get_cmap(self.cmap_reg))
            current_cmap.set_bad(color='black')
            im = self.ax_reg.imshow(np.atleast_2d(regulator_exp).T, cmap=current_cmap, aspect="auto", vmin=-1, vmax=1)
            self.ims.append(im)
            im = self.ax_main.imshow(np.atleast_2d(graphical_ar),
                                     cmap=self.cmap_main, aspect="auto", vmin=-max_val, vmax=max_val)
            self.ims.append(im)
        else:
            im = self.ax_main.imshow(np.atleast_2d(graphical_ar),
                                     cmap=self.cmap_main, aspect="auto", vmin=-max_val, vmax=max_val)
            self.ims.append(im)

    def _add_colorbar(self):
        self.fig.subplots_adjust(left=0.06, right=0.65)
        rows = 1 + self.isreg
        cols = 1
        gs = GridSpec(rows, cols)
        gs.update(left=0.8, right=0.85, wspace=1, hspace=0.3, bottom=0.2, top=0.5)

        if self.isreg:
            colorbar_names = ['Regulator\'s \n expression', 'Regulon\'s \n enrichment']
            colorbar_images = [0, 1]
        else:
            colorbar_names = ['Regulon\'s \n enrichment']
            colorbar_images = [-1]

        for i in colorbar_images:
            cax = self.fig.add_subplot(gs[i // cols, i % cols])
            self.fig.colorbar(self.ims[i], cax=cax)
            cax.set_title(colorbar_names[i], fontsize=10)

    def _make_heatmap(self, max_rows, max_val, title):
        pathways, graphical_ar, regulator_exp = self._subset_and_sort_pathways(max_rows)
        n_pathways = len(pathways)
        plt.rcParams.update({'font.weight': 'roman'})
        plt.rcParams.update({'ytick.labelsize': 10})
        fontsize_pt = plt.rcParams['ytick.labelsize']
        dpi = 72.27
        matrix_height_pt = (fontsize_pt + 30 / 2) * n_pathways
        matrix_height_in = matrix_height_pt / dpi
        matrix_width_pt = (fontsize_pt + 50 / 2) * self.n_bins
        matrix_width_in = matrix_width_pt / dpi
        figure_height = matrix_height_in
        figure_width = matrix_width_in

        has_bar = self.is_continuous and self.bin_edges is not None and len(self.bin_edges) == self.n_bins + 1

        # Build GridSpec layout
        n_rows = 2 if has_bar else 1
        n_cols = 2 if self.isreg else 1
        height_ratios = [0.5, n_pathways] if has_bar else [n_pathways]
        width_ratios = [1, self.n_bins] if self.isreg else [self.n_bins]

        if has_bar:
            figure_height += 0.5  # extra space for bar

        self.fig = plt.figure(figsize=(figure_width, figure_height))
        gs = GridSpec(n_rows, n_cols, figure=self.fig, height_ratios=height_ratios, width_ratios=width_ratios,
                      hspace=0.05, wspace=0.05)

        heatmap_row = 1 if has_bar else 0
        heatmap_col = 1 if self.isreg else 0

        self.ax_main = self.fig.add_subplot(gs[heatmap_row, heatmap_col])

        if self.isreg:
            self.ax_reg = self.fig.add_subplot(gs[heatmap_row, 0])
        else:
            self.ax_reg = None

        if has_bar:
            if self.isreg:
                self.ax_bar = self.fig.add_subplot(gs[0, 1])
                # Hide unused top-left cell
                ax_empty = self.fig.add_subplot(gs[0, 0])
                ax_empty.set_visible(False)
            else:
                self.ax_bar = self.fig.add_subplot(gs[0, 0])
            self._draw_bin_edge_bar(self.ax_bar, self.n_bins)

        # For backward compatibility, set self.ax
        if self.isreg:
            self.ax = [self.ax_reg, self.ax_main]
        else:
            self.ax = self.ax_main

        self._columnwise_heatmap(graphical_ar, regulator_exp, max_val)

        if self.isreg:
            self.ax_reg.set(xticks=[], yticks=np.arange(n_pathways), yticklabels=pathways)
            self.ax_reg.xaxis.set_label_position('top')
            self.ax_main.set(xticks=[], yticks=[])
            self.ax_main.xaxis.set_label_position('top')
        else:
            self.ax_main.set(xticks=[], yticks=np.arange(n_pathways), yticklabels=pathways)
            self.ax_main.xaxis.set_label_position('top')
        self.fig.suptitle(title)
        self._add_colorbar()

    def add_gene_expression(self,
                            genes: np.ndarray,
                            expression: np.ndarray):
        self.expression = expression
        self.genes = genes

    def _calculate_regulator_expression(self):
        self.regulator_exp = np.array([np.nan] * len(self.regulator_names))
        for i, gene in enumerate(self.regulator_names):
            if gene in self.genes:
                self.regulator_exp[i] = self.expression[np.where(self.genes == gene)[0][0]]

    def save(self,
             output_name: str,
             max_rows: Optional[int] = 50,
             show_reg: Optional[bool] = False,
             max_val: Optional[int] = 5,
             title=''):
        """
        saves a heatmap
        Parameters
        ----------
        max_rows
            maximum number of rows in a heatmap, if -1 provided then the number of rows equals the number of pathways
        output_name: str
            name of the output file, if extension provided the file is saved using that extension,
            if not the picture is saved using svg format
        Returns
        -------
        creates a file containing a picture of the heatmap
        """
        if self.expression is not None:
            self._calculate_regulator_expression()
            self.isreg = show_reg and (sum(~np.isnan(self.regulator_exp)) != 0)
        self._make_heatmap(max_rows, max_val, title)
        if output_name.split('.')[-1] not in ['svg', 'jpg', 'png', 'pdf']:
            output_name += '.svg'
        self.fig.savefig(output_name, bbox_inches='tight')

    def show(self,
             max_rows: Optional[int] = 50,
             show_reg: Optional[bool] = False,
             max_val: Optional[int] = 5,
             title=''):
        """
        shows a heatmap
        Returns
        -------
        """
        if self.expression is not None:
            self._calculate_regulator_expression()
            self.isreg = show_reg and (sum(~np.isnan(self.regulator_exp)) != 0)
        self._make_heatmap(max_rows, max_val, title)
        self.fig.show()

    def save_matrix(self, output_path: str):
        """Save enrichment matrix as TSV with a JSON metadata header line.

        Parameters
        ----------
        output_path : str
            Path to write the matrix TSV file.
        """
        meta = {
            "n_bins": int(self.n_bins),
            "n_pathways": int(self.n_pathways),
            "cmap_main": self.cmap_main,
            "cmap_reg": self.cmap_reg,
        }
        if self.bin_edges is not None:
            meta["bin_edges"] = [round(float(v), 6) for v in self.bin_edges]

        header_cols = ["pathway"] + [f"bin_{i}" for i in range(self.n_bins)]
        lines = []
        lines.append("#META\t" + json.dumps(meta))
        lines.append("\t".join(header_cols))
        for i in range(self.n_pathways):
            row = [str(self.pathways[i])] + [f"{v:.4f}" for v in self.graphical_ar[i]]
            lines.append("\t".join(row))
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    @classmethod
    def from_matrix(cls, path: str) -> 'Heatmap':
        """Load a Heatmap from a saved matrix TSV file.

        Parameters
        ----------
        path : str
            Path to the matrix TSV file (with #META header).

        Returns
        -------
        Heatmap
        """
        with open(path) as f:
            first_line = f.readline().rstrip("\n")
            if not first_line.startswith("#META\t"):
                raise ValueError(f"Expected #META header, got: {first_line[:40]}")
            meta = json.loads(first_line.split("\t", 1)[1])
            # skip column header
            f.readline()
            pathways = []
            rows = []
            for line in f:
                parts = line.rstrip("\n").split("\t")
                pathways.append(parts[0])
                rows.append([float(v) for v in parts[1:]])

        pathways = np.array(pathways)
        graphical_ar = np.array(rows)
        bin_edges = None
        if "bin_edges" in meta:
            bin_edges = np.array(meta["bin_edges"])

        hm = cls(
            pathways=pathways,
            graphical_ar=graphical_ar,
            cmap_main=meta.get("cmap_main", "viridis"),
            cmap_reg=meta.get("cmap_reg", "plasma"),
            bin_edges=bin_edges,
        )
        return hm

    def to_html(self,
                output_path: str,
                max_rows: int = 50,
                show_reg: bool = False,
                max_val: int = 5,
                title: str = ''):
        """Generate a standalone HTML file with colored enrichment table.

        Parameters
        ----------
        output_path : str
            Path to write the HTML file.
        max_rows : int
            Maximum number of pathways to display.
        show_reg : bool
            Show regulator expression column.
        max_val : int
            Color scale absolute cap.
        title : str
            Page title.
        """
        if self.expression is not None:
            self._calculate_regulator_expression()
            use_reg = show_reg and (sum(~np.isnan(self.regulator_exp)) != 0)
        else:
            use_reg = False

        pathways, graphical_ar, regulator_exp = self._subset_and_sort_pathways(max_rows)
        n_pathways = len(pathways)

        def _enrichment_color(val):
            """Map value to RGB: negative->green, zero->white, positive->red."""
            clamped = max(-max_val, min(max_val, val))
            t = clamped / max_val  # -1 to 1
            if t < 0:
                r = int(255 + t * 255)
                g = int(128 + (1 + t) * 127)
                b = int(255 + t * 255)
            else:
                r = 255
                g = int(255 - t * 255)
                b = int(255 - t * 255)
            return f"rgb({r},{g},{b})"

        def _direction_label(val):
            if val > 0:
                return "over-represented"
            elif val < 0:
                return "under-represented"
            return "neutral"

        html_parts = []
        html_parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        html_parts.append(f"<title>{title or 'pyPAGE Enrichment Heatmap'}</title>")
        html_parts.append("""<style>
body { font-family: Arial, sans-serif; margin: 20px; }
h1 { font-size: 1.3em; }
table { border-collapse: collapse; }
td, th { padding: 4px 8px; text-align: center; font-size: 12px; border: 1px solid #ddd; }
th.pathway { text-align: left; min-width: 180px; }
td.pathway { text-align: left; font-size: 11px; }
.bin-bar-cell { font-size: 9px; white-space: pre-line; }
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

        html_parts.append("<table>")

        # Optional bin-edge header row
        has_bar = self.is_continuous and self.bin_edges is not None and len(self.bin_edges) == self.n_bins + 1
        if has_bar:
            html_parts.append("<tr>")
            html_parts.append("<th></th>")
            if use_reg:
                html_parts.append("<th></th>")
            cmap = matplotlib.cm.get_cmap('RdYlBu_r')
            norm = Normalize(vmin=self.bin_edges[0], vmax=self.bin_edges[-1])
            for i in range(self.n_bins):
                lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
                mid = (lo + hi) / 2.0
                rgba = cmap(norm(mid))
                r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
                html_parts.append(
                    f'<td class="bin-bar-cell" style="background:rgb({r},{g},{b})">'
                    f'{lo:.2g}\n{hi:.2g}</td>')
            html_parts.append("</tr>")

        # Column header
        html_parts.append("<tr><th class='pathway'>Pathway</th>")
        if use_reg:
            html_parts.append("<th>Reg</th>")
        for i in range(self.n_bins):
            html_parts.append(f"<th>Bin {i}</th>")
        html_parts.append("</tr>")

        # Data rows
        for row_idx in range(n_pathways):
            pw = pathways[row_idx]
            html_parts.append(f"<tr><td class='pathway'>{pw}</td>")
            if use_reg and regulator_exp is not None:
                rv = regulator_exp[row_idx]
                if np.isnan(rv):
                    html_parts.append("<td style='background:black;color:white'>NA</td>")
                else:
                    # Blue to red scale for regulator
                    t = max(-1, min(1, rv))
                    if t < 0:
                        rr = int(255 + t * 255)
                        gg = int(255 + t * 255)
                        bb = 255
                    else:
                        rr = 255
                        gg = int(255 - t * 255)
                        bb = int(255 - t * 255)
                    html_parts.append(
                        f'<td style="background:rgb({rr},{gg},{bb})" '
                        f'onmousemove="showTip(event,\'Regulator: {rv:.3f}\')" '
                        f'onmouseout="hideTip()">{rv:.2f}</td>')

            for bin_idx in range(self.n_bins):
                val = graphical_ar[row_idx, bin_idx]
                bg = _enrichment_color(val)
                direction = _direction_label(val)
                tip_text = f"{pw}<br>Bin {bin_idx}: {val:.3f}<br>{direction}"
                html_parts.append(
                    f'<td style="background:{bg}" '
                    f'onmousemove="showTip(event,\'{tip_text}\')" '
                    f'onmouseout="hideTip()">{val:.2f}</td>')
            html_parts.append("</tr>")

        html_parts.append("</table>")

        # Legend
        html_parts.append('<div class="legend">')
        html_parts.append(f'<span class="legend-box" style="background:rgb(0,128,0)"></span> Under-represented &nbsp;')
        html_parts.append(f'<span class="legend-box" style="background:rgb(255,255,255)"></span> Neutral &nbsp;')
        html_parts.append(f'<span class="legend-box" style="background:rgb(255,0,0)"></span> Over-represented &nbsp;')
        html_parts.append(f'<br>Color scale capped at [{-max_val}, {max_val}]')
        html_parts.append('</div>')

        html_parts.append("</body></html>")

        with open(output_path, "w") as f:
            f.write("\n".join(html_parts))

    def convert_from_to(self,
                        input_format: str,
                        output_format: str,
                        species: Optional[str] = 'human'):
        """
        A function which changes accessions
        Parameters
        ----------
        input_format
            input accession type, takes 'enst', 'ensg', 'refseq', 'entrez', 'gs', 'ext'
        output_format
            output accession type, takes 'enst', 'ensg', 'refseq', 'entrez', 'gs', 'ext'
        species
            analyzed species, takes either 'human' or 'mouse'
        """
        if input_format in ['ensg', 'enst', 'refseq']:  # for ex., remove '.1' from 'ENSG00000128016.1'
            self.regulator_names = np.array([gene.split('.')[0] for gene in self.regulator_names])
        self.regulator_names = change_accessions(self.regulator_names,
                                                 input_format,
                                                 output_format,
                                                 species)
