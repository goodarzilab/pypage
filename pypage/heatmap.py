import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, ListedColormap, TwoSlopeNorm
import matplotlib

# Editable fonts in PDF/SVG (for Adobe Illustrator compatibility)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
from typing import Optional
import copy
from .io.accession_types import change_accessions

# iPAGE blue→grey→red diverging colormap (from cmap_dens.txt)
_IPAGE_RGB = np.array([
    [0.000000, 0.607843, 0.901961],
    [0.018301, 0.603922, 0.892272],
    [0.036601, 0.600000, 0.882584],
    [0.054902, 0.596078, 0.872895],
    [0.073203, 0.592157, 0.863206],
    [0.091503, 0.588235, 0.853518],
    [0.109804, 0.584314, 0.843829],
    [0.128105, 0.580392, 0.834141],
    [0.146405, 0.576471, 0.824452],
    [0.164706, 0.572549, 0.814764],
    [0.183007, 0.568627, 0.805075],
    [0.201307, 0.564706, 0.795386],
    [0.219608, 0.560784, 0.785698],
    [0.237908, 0.556863, 0.776009],
    [0.256209, 0.552941, 0.766321],
    [0.274510, 0.549020, 0.756632],
    [0.292810, 0.545098, 0.746943],
    [0.311111, 0.541176, 0.737255],
    [0.329412, 0.537255, 0.727566],
    [0.347712, 0.533333, 0.717878],
    [0.366013, 0.529412, 0.708189],
    [0.384314, 0.525490, 0.698501],
    [0.402614, 0.521569, 0.688812],
    [0.420915, 0.517647, 0.679123],
    [0.439216, 0.513725, 0.669435],
    [0.457516, 0.509804, 0.659746],
    [0.475817, 0.505882, 0.650058],
    [0.494118, 0.501961, 0.640369],
    [0.512418, 0.498039, 0.630681],
    [0.530719, 0.494118, 0.620992],
    [0.549020, 0.490196, 0.611303],
    [0.567320, 0.486275, 0.601615],
    [0.585621, 0.482353, 0.591926],
    [0.603922, 0.478431, 0.582238],
    [0.622222, 0.474510, 0.572549],
    [0.640523, 0.470588, 0.562860],
    [0.658824, 0.466667, 0.553172],
    [0.677124, 0.462745, 0.543483],
    [0.695425, 0.458824, 0.533795],
    [0.713726, 0.454902, 0.524106],
    [0.732026, 0.450980, 0.514418],
    [0.750327, 0.447059, 0.504729],
    [0.768627, 0.443137, 0.495040],
    [0.786928, 0.439216, 0.485352],
    [0.805229, 0.435294, 0.475663],
    [0.823529, 0.431373, 0.465975],
    [0.841830, 0.427451, 0.456286],
    [0.860131, 0.423529, 0.446597],
    [0.878431, 0.419608, 0.436909],
    [0.896732, 0.415686, 0.427220],
    [0.915033, 0.411765, 0.417532],
    [0.933333, 0.407843, 0.407843],
    [0.944444, 0.369281, 0.369281],
    [0.955556, 0.330719, 0.330719],
    [0.966667, 0.292157, 0.292157],
    [0.977778, 0.253595, 0.253595],
    [0.988889, 0.215033, 0.215033],
    [1.000000, 0.176471, 0.176471],
    [0.964052, 0.147059, 0.147059],
    [0.928105, 0.117647, 0.117647],
    [0.892157, 0.088235, 0.088235],
    [0.856209, 0.058824, 0.058824],
    [0.820261, 0.029412, 0.029412],
    [0.784314, 0.000000, 0.000000],
])
_ipage_cmap = ListedColormap(_IPAGE_RGB, name="ipage")
try:
    matplotlib.colormaps.register(_ipage_cmap)
except ValueError:
    pass  # already registered


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
                 cmap_main: Optional[str] = 'ipage',
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

    def _draw_bin_edge_bar(self, ax_bar, n_bins, bar_min=None, bar_max=None):
        """Draw iPAGE-style bin range bar: black cells with red range indicators."""
        edges = self.bin_edges
        if bar_min is None:
            bar_min = float(edges[0])
        if bar_max is None:
            bar_max = float(edges[-1])

        global_range = bar_max - bar_min
        if global_range == 0:
            global_range = 1.0

        for i in range(n_bins):
            lo, hi = float(edges[i]), float(edges[i + 1])
            # Black background
            ax_bar.add_patch(plt.Rectangle((i, 0), 1, 1,
                             facecolor='black', edgecolor='none'))
            # Red bar showing where bin range falls in global range
            y_lo = max(0.0, (lo - bar_min) / global_range)
            y_hi = min(1.0, (hi - bar_min) / global_range)
            bar_h = max(y_hi - y_lo, 0.015)
            ax_bar.add_patch(plt.Rectangle((i, y_lo), 1, bar_h,
                             facecolor='red', edgecolor='none'))

        ax_bar.set_xlim(0, n_bins)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        # Global min/max labels on left
        ax_bar.text(-0.15, 1.0, f"{bar_max:.4g}",
                    ha='right', va='top', fontsize=7, clip_on=False)
        ax_bar.text(-0.15, 0.0, f"{bar_min:.4g}",
                    ha='right', va='bottom', fontsize=7, clip_on=False)

    @staticmethod
    def _make_norm(min_val, max_val):
        """Build a colormap norm that keeps 0 at the center of a diverging map."""
        if min_val < 0 < max_val:
            return TwoSlopeNorm(vcenter=0, vmin=min_val, vmax=max_val)
        return Normalize(vmin=min_val, vmax=max_val)

    def _columnwise_heatmap(self,
                            graphical_ar: np.ndarray,
                            regulator_exp: np.ndarray,
                            min_val: float,
                            max_val: float):
        norm = self._make_norm(min_val, max_val)
        self.ims = []
        if self.isreg:
            current_cmap = copy.copy(matplotlib.colormaps.get_cmap(self.cmap_reg))
            current_cmap.set_bad(color='black')
            im = self.ax_reg.imshow(np.atleast_2d(regulator_exp).T, cmap=current_cmap, aspect="auto", vmin=-1, vmax=1)
            self.ims.append(im)
            im = self.ax_main.imshow(np.atleast_2d(graphical_ar),
                                     cmap=self.cmap_main, aspect="auto", norm=norm)
            self.ims.append(im)
        else:
            im = self.ax_main.imshow(np.atleast_2d(graphical_ar),
                                     cmap=self.cmap_main, aspect="auto", norm=norm)
            self.ims.append(im)

    def _add_colorbar(self, min_val, max_val):
        self.fig.subplots_adjust(left=0.18, right=0.58)

        # Enrichment colorbar on the far left
        cax = self.fig.add_axes([0.02, 0.15, 0.02, 0.7])
        # Build gradient using TwoSlopeNorm so 0 stays at colormap center
        values = np.linspace(max_val, min_val, 256)
        norm = self._make_norm(min_val, max_val)
        gradient = norm(values).reshape(-1, 1)
        cax.imshow(gradient, aspect='auto', cmap=self.cmap_main,
                   extent=[0, 1, min_val, max_val], vmin=0, vmax=1)
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.set_yticks([max_val, 0, min_val])
        cax.tick_params(axis='y', which='both', length=0, labelsize=7, pad=2)
        for spine in cax.spines.values():
            spine.set_visible(False)
        # Rotated label
        cax.set_ylabel('gene-set enrichment', rotation=90, fontsize=8, labelpad=5)

        if self.isreg:
            cax_reg = self.fig.add_axes([0.02, 0.02, 0.02, 0.10])
            reg_cmap = matplotlib.colormaps.get_cmap(self.cmap_reg)
            reg_gradient = np.linspace(1, 0, 256).reshape(-1, 1)
            cax_reg.imshow(reg_gradient, aspect='auto', cmap=reg_cmap,
                           extent=[0, 1, -1, 1])
            cax_reg.set_xticks([])
            cax_reg.yaxis.tick_right()
            cax_reg.set_yticks([-1, 0, 1])
            cax_reg.tick_params(axis='y', which='both', length=0, labelsize=6)
            for spine in cax_reg.spines.values():
                spine.set_visible(False)
            cax_reg.set_ylabel('regulator', rotation=90, fontsize=7, labelpad=5)

    def _make_heatmap(self, max_rows, min_val, max_val, title, bar_min=None, bar_max=None):
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
        height_ratios = [1.2, n_pathways] if has_bar else [n_pathways]
        width_ratios = [1, self.n_bins] if self.isreg else [self.n_bins]

        if has_bar:
            figure_height += 0.6  # extra space for bar

        self.fig = plt.figure(figsize=(figure_width, figure_height))
        gs = GridSpec(n_rows, n_cols, figure=self.fig, height_ratios=height_ratios, width_ratios=width_ratios,
                      hspace=0.02, wspace=0.05)

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
            self._draw_bin_edge_bar(self.ax_bar, self.n_bins,
                                    bar_min=bar_min, bar_max=bar_max)

        # For backward compatibility, set self.ax
        if self.isreg:
            self.ax = [self.ax_reg, self.ax_main]
        else:
            self.ax = self.ax_main

        self._columnwise_heatmap(graphical_ar, regulator_exp, min_val, max_val)

        # Remove borders and ticks from all heatmap axes
        for ax in [self.ax_main, self.ax_reg]:
            if ax is not None:
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(axis='both', which='both', length=0)

        # Pathway labels on the right side of the main heatmap
        if self.isreg:
            self.ax_reg.set(xticks=[], yticks=[])
            self.ax_main.set(xticks=[], yticks=np.arange(n_pathways),
                             yticklabels=pathways)
            self.ax_main.yaxis.tick_right()
        else:
            self.ax_main.set(xticks=[], yticks=np.arange(n_pathways),
                             yticklabels=pathways)
            self.ax_main.yaxis.tick_right()

        self.fig.suptitle(title)
        self._add_colorbar(min_val, max_val)

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
             max_val: float = 5.0,
             min_val: Optional[float] = None,
             title='',
             bar_min: Optional[float] = None,
             bar_max: Optional[float] = None):
        """
        saves a heatmap
        Parameters
        ----------
        max_rows
            maximum number of rows in a heatmap, if -1 provided then the number of rows equals the number of pathways
        output_name: str
            name of the output file, if extension provided the file is saved using that extension,
            if not the picture is saved using svg format
        max_val : float
            Color scale upper cap.
        min_val : float or None
            Color scale lower cap (default: -max_val).
        bar_min : float or None
            Global minimum for bin-edge bar normalization.
        bar_max : float or None
            Global maximum for bin-edge bar normalization.
        Returns
        -------
        creates a file containing a picture of the heatmap
        """
        if min_val is None:
            min_val = -max_val
        if self.expression is not None:
            self._calculate_regulator_expression()
            self.isreg = show_reg and (sum(~np.isnan(self.regulator_exp)) != 0)
        self._make_heatmap(max_rows, min_val, max_val, title,
                           bar_min=bar_min, bar_max=bar_max)
        if output_name.split('.')[-1] not in ['svg', 'jpg', 'png', 'pdf']:
            output_name += '.svg'
        self.fig.savefig(output_name, bbox_inches='tight')

    def show(self,
             max_rows: Optional[int] = 50,
             show_reg: Optional[bool] = False,
             max_val: float = 5.0,
             min_val: Optional[float] = None,
             title='',
             bar_min: Optional[float] = None,
             bar_max: Optional[float] = None):
        """
        shows a heatmap
        Returns
        -------
        """
        if min_val is None:
            min_val = -max_val
        if self.expression is not None:
            self._calculate_regulator_expression()
            self.isreg = show_reg and (sum(~np.isnan(self.regulator_exp)) != 0)
        self._make_heatmap(max_rows, min_val, max_val, title,
                           bar_min=bar_min, bar_max=bar_max)
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
            cmap_main=meta.get("cmap_main", "ipage"),
            cmap_reg=meta.get("cmap_reg", "plasma"),
            bin_edges=bin_edges,
        )
        return hm

    def to_html(self,
                output_path: str,
                max_rows: int = 50,
                show_reg: bool = False,
                max_val: float = 5.0,
                min_val: Optional[float] = None,
                title: str = '',
                bar_min: Optional[float] = None,
                bar_max: Optional[float] = None):
        """Generate a standalone HTML file with colored enrichment table.

        Parameters
        ----------
        output_path : str
            Path to write the HTML file.
        max_rows : int
            Maximum number of pathways to display.
        show_reg : bool
            Show regulator expression column.
        max_val : float
            Color scale upper cap.
        min_val : float or None
            Color scale lower cap (default: -max_val).
        title : str
            Page title.
        bar_min : float or None
            Global minimum for bin-edge bar normalization.
        bar_max : float or None
            Global maximum for bin-edge bar normalization.
        """
        if min_val is None:
            min_val = -max_val

        if self.expression is not None:
            self._calculate_regulator_expression()
            use_reg = show_reg and (sum(~np.isnan(self.regulator_exp)) != 0)
        else:
            use_reg = False

        pathways, graphical_ar, regulator_exp = self._subset_and_sort_pathways(max_rows)
        n_pathways = len(pathways)

        cmap_obj = matplotlib.colormaps[self.cmap_main]
        norm = self._make_norm(min_val, max_val)

        def _enrichment_color(val):
            """Map value to RGB using the active colormap with TwoSlopeNorm."""
            clamped = max(min_val, min(max_val, val))
            t = float(norm(clamped))
            r, g, b, _ = cmap_obj(t)
            return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        def _direction_label(val):
            if val > 0:
                return "over-represented"
            elif val < 0:
                return "under-represented"
            return "neutral"

        # Build vertical colorbar legend (64 color steps)
        n_legend_steps = 64
        legend_html = []
        legend_html.append('<div style="display:flex; align-items:center; gap:4px;">')
        legend_html.append(
            '<div style="writing-mode:vertical-rl; transform:rotate(180deg); '
            'font-size:11px; font-family:Helvetica,Arial,sans-serif;">'
            'gene-set enrichment</div>')
        legend_html.append('<div>')
        legend_html.append(
            f'<div style="font-size:9px; text-align:center;">'
            f'{max_val:.4g}</div>')
        for j in range(n_legend_steps):
            val = max_val - j * (max_val - min_val) / (n_legend_steps - 1)
            t = float(norm(val))
            r, g, b, _ = cmap_obj(t)
            legend_html.append(
                f'<div style="width:18px; height:3px; '
                f'background:rgb({int(r*255)},{int(g*255)},{int(b*255)});"></div>')
        legend_html.append(
            f'<div style="font-size:9px; text-align:center;">'
            f'{min_val:.4g}</div>')
        legend_html.append('</div></div>')

        html_parts = []
        html_parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        html_parts.append(f"<title>{title or 'pyPAGE Enrichment Heatmap'}</title>")
        html_parts.append("""<style>
body { font-family: Helvetica, Arial, sans-serif; margin: 20px; }
h1 { font-size: 1.3em; }
table.main { border-collapse: collapse; }
td.grid { width: 55px; height: 30px; border: none; }
td.gridheader { width: 55px; background: #000; border: 1px solid #333;
  position: relative; overflow: hidden; }
td.gridlabel { white-space: nowrap; padding-left: 5px; font-size: 11px; border: none; }
td.gridreg { width: 30px; height: 30px; border: none; text-align: center; font-size: 10px; }
#tooltip { position: fixed; background: #333; color: #fff; padding: 6px 10px;
  border-radius: 4px; font-size: 11px; pointer-events: none; display: none; z-index: 999; }
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

        # Flex container: legend on left, table on right
        html_parts.append('<div style="display:flex; align-items:flex-start; gap:20px;">')
        html_parts.append("\n".join(legend_html))
        html_parts.append('<table class="main" cellpadding="0" cellspacing="0">')

        # iPAGE-style bin-edge header row (black cells with red range bars)
        has_bar = self.is_continuous and self.bin_edges is not None and len(self.bin_edges) == self.n_bins + 1
        if has_bar:
            bmin = bar_min if bar_min is not None else float(self.bin_edges[0])
            bmax = bar_max if bar_max is not None else float(self.bin_edges[-1])
            global_range = bmax - bmin
            if global_range == 0:
                global_range = 1.0
            cell_height = 45

            html_parts.append(f'<tr style="height:{cell_height}px">')
            if use_reg:
                html_parts.append('<td style="border:none;"></td>')
            for i in range(self.n_bins):
                lo, hi = float(self.bin_edges[i]), float(self.bin_edges[i + 1])
                y_lo_frac = max(0.0, (lo - bmin) / global_range)
                y_hi_frac = min(1.0, (hi - bmin) / global_range)
                bar_bottom_pct = y_lo_frac * 100
                bar_height_pct = max((y_hi_frac - y_lo_frac) * 100, 2)
                html_parts.append(
                    f'<td class="gridheader" '
                    f'title="lower bound = {lo:.2f}, upper bound = {hi:.2f}">'
                    f'<div style="position:absolute; left:0; right:0; '
                    f'bottom:{bar_bottom_pct:.1f}%; height:{bar_height_pct:.1f}%; '
                    f'background:#F00;"></div></td>')
            # Min/max labels
            html_parts.append(
                f'<td style="border:none; font-size:9px; vertical-align:top; '
                f'padding-left:4px; white-space:nowrap;">{bmax:.4g}</td>')
            html_parts.append('</tr>')
            # Second label row for bar_min
            html_parts.append('<tr style="height:0px;">')
            if use_reg:
                html_parts.append('<td style="border:none;"></td>')
            for _ in range(self.n_bins):
                html_parts.append('<td style="border:none; height:0;"></td>')
            html_parts.append(
                f'<td style="border:none; font-size:9px; vertical-align:bottom; '
                f'padding-left:4px; white-space:nowrap; position:relative; top:-14px;">'
                f'{bmin:.4g}</td>')
            html_parts.append('</tr>')

        # Data rows
        for row_idx in range(n_pathways):
            pw = pathways[row_idx]
            html_parts.append("<tr>")
            if use_reg and regulator_exp is not None:
                rv = regulator_exp[row_idx]
                if np.isnan(rv):
                    html_parts.append(
                        '<td class="gridreg" style="background:black;color:white">NA</td>')
                else:
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
                        f'<td class="gridreg" style="background:rgb({rr},{gg},{bb})" '
                        f'onmousemove="showTip(event,\'Regulator: {rv:.3f}\')" '
                        f'onmouseout="hideTip()">{rv:.2f}</td>')

            for bin_idx in range(self.n_bins):
                val = graphical_ar[row_idx, bin_idx]
                bg = _enrichment_color(val)
                direction = _direction_label(val)
                tip_text = f"{pw}<br>Bin {bin_idx}: {val:.3f}<br>{direction}"
                html_parts.append(
                    f'<td class="grid" style="background:{bg}" '
                    f'onmousemove="showTip(event,\'{tip_text}\')" '
                    f'onmouseout="hideTip()">&nbsp;</td>')
            html_parts.append(f'<td class="gridlabel">{pw}</td>')
            html_parts.append("</tr>")

        html_parts.append("</table>")
        html_parts.append("</div>")  # close flex container

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
