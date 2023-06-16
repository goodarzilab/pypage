import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    fig: matplotlib.figure.Figure
    ax: matplotlib.figure.Axes
    ==========
    Methods
    ----------
    save
        saves a heatmap
    show
        shows a heatmap
    """
    def __init__(self,
                 pathways: np.ndarray,
                 graphical_ar: np.ndarray,
                 regulator_exp: Optional[np.ndarray] = None,
                 cmap_main: Optional[str] = 'viridis',
                 cmap_reg: Optional[str] = 'plasma'):
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

    def _columnwise_heatmap(self,
                            graphical_ar: np.ndarray,
                            regulator_exp: np.ndarray,
                            max_val: int):
        self.ims = []
        if self.isreg:
            current_cmap = copy.copy(matplotlib.cm.get_cmap(self.cmap_reg))
            current_cmap.set_bad(color='black')
            im = self.ax[0].imshow(np.atleast_2d(regulator_exp).T, cmap=current_cmap, aspect="auto", vmin=-1, vmax=1)
            self.ims.append(im)
            im = self.ax[1].imshow(np.atleast_2d(graphical_ar),
                                   cmap=self.cmap_main, aspect="auto", vmin=-max_val, vmax=max_val)
            self.ims.append(im)
        else:
            im = self.ax.imshow(np.atleast_2d(graphical_ar),
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

        if self.isreg:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(figure_width, figure_height),
                                             gridspec_kw={'width_ratios': [1, self.n_bins]})
            self.fig.subplots_adjust(wspace=0.05)
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(figure_width, figure_height))

        self._columnwise_heatmap(graphical_ar, regulator_exp, max_val)

        if self.isreg:
            self.ax[0].set(xticks=[], yticks=np.arange(n_pathways), yticklabels=pathways)
            self.ax[0].xaxis.set_label_position('top')
            self.ax[1].set(xticks=[], yticks=[])
            self.ax[1].xaxis.set_label_position('top')
        else:
            self.ax.set(xticks=[], yticks=np.arange(n_pathways), yticklabels=pathways)
            self.ax.xaxis.set_label_position('top')
            plt.xticks(rotation=90)
        plt.title(title)
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
        if output_name.split('.')[-1] not in ['svg', 'jpg', 'png']:
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
