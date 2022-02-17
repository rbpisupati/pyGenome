import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import BrokenBarHCollection, PathCollection, LineCollection
import seaborn as sns

from . import genome


def smooth_sum(arr, n_times= 100):
    arr = np.array(arr, dtype = float)
    for e_ind in np.arange(n_times):
        arr = np.insert(arr, 0, arr[0])
        arr = np.append(arr, arr[-1] )
        arr = sp.signal.convolve( arr, [0.25,0.5,0.25], mode = "same" )[1:-1]
#         arr = sp.signal.convolve( arr, [0.25,0.5,0.25] )[1:-1]
    return(arr)

"""
Demonstrates plotting chromosome ideograms and genes (or any features, really)
using matplotlib.
1) Assumes a file from UCSC's Table Browser from the "cytoBandIdeo" table,
saved as "ideogram.txt". Lines look like this::
    #chrom  chromStart  chromEnd  name    gieStain
    chr1    0           2300000   p36.33  gneg
    chr1    2300000     5300000   p36.32  gpos25
    chr1    5300000     7100000   p36.31  gneg
2) Assumes another file, "ucsc_genes.txt", which is a BED format file
   downloaded from UCSC's Table Browser. This script will work with any
   BED-format file.
"""
# Here's the function that we'll call for each dataframe (once for chromosome
# ideograms, once for genes).  The rest of this script will be prepping data
# for input to this function
#
def chromosome_collections(df, y_positions, height,  **kwargs):
    """
    Yields BrokenBarHCollection of features that can be added to an Axes
    object.
    Parameters
    ----------
    df : pandas.DataFrame
        Must at least have columns ['chrom', 'start', 'end', 'color']. If no
        column 'width', it will be calculated from start/end.
    y_positions : dict
        Keys are chromosomes, values are y-value at which to anchor the
        BrokenBarHCollection
    height : float
        Height of each BrokenBarHCollection
    Additional kwargs are passed to BrokenBarHCollection
    """
    del_width = False
    if 'width' not in df.columns:
        del_width = True
        df['width'] = df['end'] - df['start']
    for chrom, group in df.groupby('chrom'):
        yrange = (y_positions[chrom], height)
        xranges = group[['start', 'width']].values
        yield(BrokenBarHCollection(xranges, yrange, facecolors=group['colors'], **kwargs))
    if del_width:
        del df['width']
        
def line_collections(df, y_positions, line_plt_options = None, **kwargs):
    """
    Yields BrokenBarHCollection of features that can be added to an Axes
    object.
    Parameters
    ----------
    df : pandas.DataFrame
        Must at least have columns ['chrom', 'start']
    y_positions : dict
        Keys are chromosomes, values are y-value at which to anchor the
        PathCollection
    Additional kwargs are passed to BrokenBarHCollection
    """
    if line_plt_options is None:
        line_plt_options = {}
    if "smooth" not in line_plt_options.keys():
            line_plt_options['smooth'] = 1
    if "bins" not in line_plt_options.keys():
            line_plt_options['bins'] = 100
    # if "max_height" not in line_plt_options.keys():

    for chrom, group in df.groupby('chrom'):
        if chrom in y_positions.keys():
            inds = np.histogram( group['start'], bins = line_plt_options['bins'])
            # import ipdb; ipdb.set_trace()
            relative_y = inds[0]/float(np.sum(inds[0]))
            relative_y = smooth_sum( relative_y/np.max(relative_y), line_plt_options['smooth'] )
            y_ind = y_positions[chrom] + (line_plt_options['max_height'] * relative_y)
            x_ind = inds[1][0:-1]
            yield(LineCollection([np.column_stack([x_ind, y_ind])], color=group['colors'], **kwargs))



class PlottingGenomeWide(genome.GenomeClass):
    """
    Class function to different plotting options

    1. Density plot for given positions.
    2. Heatmap

    """

    def __init__(self, ref_genome = "at_tair10"):
        super().__init__( ref_genome )

    def density_plot_chr_wide(self, bed_df, axs = None, plt_options = None):
        if plt_options is None:
            plt_options = {}
        if "chrom_height" not in plt_options.keys():
            plt_options['chrom_height'] = 0.3
        if "chrom_spacing" not in plt_options.keys():
            plt_options['chrom_spacing'] = 0.4
        if "gene_height" not in plt_options.keys():
            plt_options['gene_height'] = 0.1
        if "gene_padding" not in plt_options.keys():
            plt_options['gene_padding'] = 0.1
        if "density_bins" not in plt_options.keys():
            plt_options['density_bins'] = 100
        if "line_smooth" not in plt_options.keys():
            plt_options['line_smooth'] = 10
        if "line_kwargs" not in plt_options.keys():
            plt_options['line_kwargs'] = {}
        if "line_height" not in plt_options.keys():
            plt_options['line_height'] = plt_options['chrom_height'] - 0.01

        plt_ylim_min = -plt_options['gene_height'] - plt_options['gene_padding'] - 0.01
        plt_ylim_max = (len(self.chrs) * plt_options['chrom_height']) + ((len(self.chrs)-1) * plt_options['chrom_spacing']) + plt_options['gene_padding'] 

        if axs is None:
            axs = plt.gca()

        # Keep track of the y positions for ideograms and genes for each chromosome,
        # and the center of each ideogram (which is where we'll put the ytick labels)
        ybase = 0
        chrom_ybase = {}
        gene_ybase = {}
        chrom_centers = {}

        # Iterate in reverse so that items in the beginning of `chromosome_list` will
        # appear at the top of the plot
        for chrom in self.chrs[::-1]:
            chrom_ybase[chrom] = ybase
            chrom_centers[chrom] = ybase + plt_options['chrom_height'] / 2.
            gene_ybase[chrom] = ybase - plt_options['gene_height'] - plt_options['gene_padding']
            ybase += plt_options['chrom_height'] + plt_options['chrom_spacing']

        chromosome_bars = pd.DataFrame({'chrom': self.chrs, 'start': self.centro_start, 'end': self.centro_end, 'colors': "#525252"})
        chromosome_bars = chromosome_bars.append(pd.DataFrame({'chrom': self.chrs, 'start': 1, 'end': self.golden_chrlen, 'colors': "#bdbdbd"}))

        for collection in chromosome_collections(chromosome_bars, chrom_ybase, plt_options['chrom_height'], alpha = 0.5):
            axs.add_collection(collection)

        for collection in line_collections(bed_df, chrom_ybase, { 'max_height': plt_options['line_height'], 'smooth': plt_options['line_smooth'], 'bins': plt_options['density_bins'] }, plt_options['line_kwargs']):
            axs.add_collection(collection)

        axs.set_yticks([chrom_centers[i] for i in self.chrs] )
        axs.set_yticklabels(self.chrs)
        axs.axis('tight')
        axs.set_ylim(plt_ylim_min, plt_ylim_max)
        # plt.xticks( size = 10 )
        # plt.yticks( size = 10 )

        return((axs, {'chrom_ybase': chrom_ybase, 'gene_ybase': gene_ybase, 'chrom_centers': chrom_centers}))
        




