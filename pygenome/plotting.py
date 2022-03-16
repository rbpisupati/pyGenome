import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import BrokenBarHCollection, PathCollection, LineCollection
import seaborn as sns

from . import genome
import logging
log = logging.getLogger(__name__)

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



class PlottingGenomeWide(object):
    """
    Class function to different plotting options

    1. Density plot for given positions.
    2. Heatmap

    """

    def __init__(self, ref_genome_class):
        assert type(ref_genome_class) is genome.GenomeClass, "provide a genome class"
        self.genome = ref_genome_class

        self.chr_info = pd.DataFrame({"chr":self.genome.chrs})
        self.chr_info['len'] = self.genome.golden_chrlen
        self.chr_info['mid'] = (np.array(self.genome.golden_chrlen)/2).astype(int)
        self.chr_info['chr_ind_start'] = np.array(self.genome.chr_inds)[:-1]
        self.chr_info['chr_ind_end'] = np.array(self.genome.chr_inds)[1::]

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
        plt_ylim_max = (len(self.genome.chrs) * plt_options['chrom_height']) + ((len(self.genome.chrs)-1) * plt_options['chrom_spacing']) + plt_options['gene_padding'] 

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
        for chrom in self.genome.chrs[::-1]:
            chrom_ybase[chrom] = ybase
            chrom_centers[chrom] = ybase + plt_options['chrom_height'] / 2.
            gene_ybase[chrom] = ybase - plt_options['gene_height'] - plt_options['gene_padding']
            ybase += plt_options['chrom_height'] + plt_options['chrom_spacing']

        chromosome_bars = pd.DataFrame({'chrom': self.genome.chrs, 'start': self.genome.centro_start, 'end': self.genome.centro_end, 'colors': "#525252"})
        chromosome_bars = chromosome_bars.append(pd.DataFrame({'chrom': self.genome.chrs, 'start': 1, 'end': self.genome.golden_chrlen, 'colors': "#bdbdbd"}))

        for collection in chromosome_collections(chromosome_bars, chrom_ybase, plt_options['chrom_height'], alpha = 0.5):
            axs.add_collection(collection)

        for collection in line_collections(bed_df, chrom_ybase, { 'max_height': plt_options['line_height'], 'smooth': plt_options['line_smooth'], 'bins': plt_options['density_bins'] }, plt_options['line_kwargs']):
            axs.add_collection(collection)

        axs.set_yticks( chr_mid_ix[np.where(np.array(self.genome.golden_chrlen) > 1000000)[0]] )
        axs.set_yticklabels( np.array(self.genome.chrs)[np.where(np.array(self.genome.golden_chrlen) > 1000000)[0]] )
        # axs.set_yticks([chrom_centers[i] for i in self.genome.chrs] )
        # axs.set_yticklabels(self.genome.chrs)
        axs.axis('tight')
        axs.set_ylim(plt_ylim_min, plt_ylim_max)
        # plt.xticks( size = 10 )
        # plt.yticks( size = 10 )

        return((axs, {'chrom_ybase': chrom_ybase, 'gene_ybase': gene_ybase, 'chrom_centers': chrom_centers}))

    def density_line_plot_echr(self, bed_pos_df, echr = "Chr1", axs = None, plt_options = {}):
        """
        Function to generate a density plot for given a list of positions

        """
        assert isinstance(bed_pos_df, pd.DataFrame), "provide a dataframe object"

        # if "lims" not in plt_options.keys():
        #     plt_options['lims'] = (np.nanmin(score_df.values), np.nanmax(score_df.values))
        if "window_size" not in plt_options.keys():
            plt_options['window_size'] = 100000
        if "color" not in plt_options.keys():
            plt_options['color'] = "#238443"
        if "nsmooth" not in plt_options.keys():
            plt_options['nsmooth'] = None

        if axs is None:
            axs = plt.gca()

        data_window_density = pd.DataFrame(columns = ['chr', 'start', 'end', 'npos'])
        for ef_bin in self.genome.iter_positions_in_windows(bed_pos_df, window_size = plt_options['window_size']):
            data_window_density = pd.concat([data_window_density, pd.DataFrame({"chr": ef_bin[0], "start": ef_bin[1][0], "end": ef_bin[1][1], "npos": len(ef_bin[2]) }, index=[0])  ], ignore_index=True)

        data_window_density = data_window_density[data_df['chrom'] == echr]
        if plt_options['nsmooth'] is not None:
            data_window_density['npos'] = smooth_sum(data_window_density['npos'].values, plt_options['nsmooth']) 
            

        sns.lineplot(self.genome.get_genomewide_inds( data_window_density.iloc[:,[0,1,2]] ), data_window_density['npos'].values, ax = axs, c = plt_options['color']  )
        if 'centro_mid' in self.genome.__dict__.keys():
            echr_centro = self.genome.centro_mid[self.genome.get_chr_ind( echr )]
            axs.plot( (echr_centro, echr_centro), (0,np.maximum(data_window_density['npos'])), '--', c = "#636363" )
        
        # self.chr_info.iloc[self.genome.get_chr_ind( echr ),:]["mid"]
        # axs.set_xticks( chr_mid_ix[np.where(np.array(self.genome.golden_chrlen) > 1000000)[0]] )
        # axs.set_xticklabels( np.array(self.genome.chrs)[np.where(np.array(self.genome.golden_chrlen) > 1000000)[0]] )

    def generate_heatmap_genomewide(self, score_df, plt_options = {}, axs = None, **kwargs):
        """
        Function to generate a genome wide heatmap
            plots chromosomes separately
            boxes for pericentromers

        """
        assert isinstance(score_df, pd.DataFrame), "provide a dataframe object"

        if "lims" not in plt_options.keys():
            plt_options['lims'] = (np.nanmin(score_df.values), np.nanmax(score_df.values))
        if "xlabel" not in plt_options.keys():
            plt_options['xlabel'] = ""
        if "ylabel" not in plt_options.keys():
            plt_options['ylabel'] = ""
        if "title" not in plt_options.keys():
            plt_options['title'] = ""
        if "cmap" not in plt_options.keys():
            plt_options['cmap'] = "Reds"
        if "col_centro_line" not in plt_options.keys():
            plt_options['col_centro_line'] = "#636363"
        if "col_chr_line" not in plt_options.keys():
            plt_options['col_chr_line'] = "#525252"

        if axs is None:
            axs = plt.gca()
        
        score_df_new = score_df.copy()
        if pd.api.types.is_string_dtype( score_df_new.columns ):
            score_df_new.columns = self.genome.get_genomewide_inds( pd.Series(score_df_new.columns) )
        else:
            score_df_new.columns = score_df_new.columns.astype(int)
        
        chr_end_ix = np.searchsorted( score_df_new.columns.values, self.genome.get_genomewide_inds( pd.DataFrame({"chr":self.genome.chrs, "pos": self.genome.golden_chrlen}) ), "right" )[0:-1]
        
        chr_mid_ix = np.searchsorted( score_df_new.columns.values, self.genome.get_genomewide_inds( pd.DataFrame({"chr":self.genome.chrs, "pos": (np.array(self.genome.golden_chrlen)/2).astype(int) }) ), "right" )
        # score_df_new.insert(loc = int(e_chr_end_ix), column = "break" + str(e_chr_end_ix), value = np.nan)
        
        sns.heatmap(
            score_df_new,
            vmin=plt_options['lims'][0], 
            vmax=plt_options['lims'][1], 
            cmap=plt_options['cmap'],
            ax = axs,
            xticklabels=False,
            yticklabels=False,
            **kwargs
            # cbar_kws=dict(use_gridspec=False,location="bottom")
        )
        axs.set_xlabel(plt_options['xlabel'])
        axs.set_ylabel(plt_options['ylabel'])
        axs.set_title(plt_options['title'])
        
        for e_chr_end_ix in chr_end_ix:
            axs.plot( (e_chr_end_ix, e_chr_end_ix), (0,score_df_new.shape[0]), '-', c = plt_options['col_chr_line'] )
            
        if 'centro_mid' in self.genome.__dict__.keys():
            centro_df = pd.DataFrame({"chr":self.genome.chrs, "pos": self.genome.centro_mid})
            centro_df = centro_df[centro_df['pos'] > 0]
            chr_centro_mid_ix = np.searchsorted( score_df_new.columns.values, self.genome.get_genomewide_inds( centro_df ), "right" )
            for e_chr_pos_ix in chr_centro_mid_ix:
                axs.plot( (e_chr_pos_ix, e_chr_pos_ix), (0,score_df_new.shape[0]), '--', c = plt_options['col_centro_line'] )


    
        axs.set_xticks( chr_mid_ix[np.where(np.array(self.genome.golden_chrlen) > 1000000)[0]] )
        axs.set_xticklabels( np.array(self.genome.chrs)[np.where(np.array(self.genome.golden_chrlen) > 1000000)[0]] )
        return(axs)

    def manhattan_plot(self, x_ind, y_ind, plt_options = {}, axs = None, **kwargs):
        """
        Plot a manhattan plot
        """
        chr_info = self.chr_info.copy()
        if "color" not in plt_options.keys():
            chr_info['color'] = pd.Series( ["#1f78b4", "#33a02c"] ).loc[np.arange(self.chr_info.shape[0]) % 2 ].values
        else:
            chr_info['color'] = plt_options['color']
        
        if "ylim" not in plt_options.keys():
            plt_options['ylim'] = (np.nanmin(y_ind), np.nanmax(y_ind))
        if "ylabel" not in plt_options.keys():
            plt_options['ylabel'] = "markers"
        if "gap" not in plt_options.keys():
            plt_options['gap'] = 10000000
        if "thres" not in plt_options.keys():
            plt_options['thres'] = None
        if "size" not in plt_options.keys():
            plt_options['size'] = 6
        
        if 'line' not in plt_options.keys():
            plt_options['line'] = False
        else:
            plt_options['line'] = True
        
        if axs is None:
            axs = plt.gca()
        
        log.warn("filtering out genome scaffolds less than 1Mb")
        chr_info = chr_info[chr_info['len'] > 1000000]
        for echr in chr_info.iterrows():
            t_chr_ix = np.where((x_ind <= echr[1]['chr_ind_end'] ) & ( x_ind > echr[1]['chr_ind_start'] ))[0]
            if plt_options['line']:
                axs.plot(x_ind[t_chr_ix] + (plt_options['gap'] * echr[0]), y_ind[t_chr_ix], '-', color = echr[1]['color'], **kwargs)
            else:
                axs.scatter(x_ind[t_chr_ix] + (plt_options['gap'] * echr[0]), y_ind[t_chr_ix], s = plt_options['size'], c = echr[1]['color'], **kwargs)
            
        axs.set_xticks( chr_info['chr_ind_start'].values + (np.arange(chr_info.shape[0]) * plt_options['gap']) + chr_info['mid'].values )
        axs.set_xticklabels( chr_info['chr'].values )
        if plt_options['thres'] is not None:
            axs.plot((0, chr_info['chr_ind_end'].iloc[-1] + (plt_options['gap'] * (chr_info.shape[0] - 1))  ), (plt_options['thres'], plt_options['thres']), "--", color = "gray")
        
        axs.set_xlabel( plt_options['ylabel'] )
        axs.set_ylim( plt_options['ylim'] )
        return(axs)





