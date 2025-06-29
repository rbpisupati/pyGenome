import sys
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from . import genome
import logging
log = logging.getLogger(__name__)

def scale_colors(minval, maxval, val, safe_colors = None):
    if safe_colors is None:
        ### Copied colors from cb.sequential.BuGn_7.colors
        safe_colors = [
            [237, 248, 251],
            [204, 236, 230],
            [153, 216, 201],
            [102, 194, 164],
            [65, 174, 118],
            [35, 139, 69],
            [0, 88, 36]
        ]
    EPSILON = sys.float_info.epsilon  # Smallest possible difference.
    i_f = float(val-minval) / float(maxval-minval) * (len(safe_colors)-1)
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    if f < EPSILON:
        ret_col = safe_colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = safe_colors[i], safe_colors[i+1]
        ret_col = int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))
    return('#%02x%02x%02x' % (ret_col[0], ret_col[1], ret_col[2]))

np_scale_colors = np.vectorize(scale_colors, excluded = ['safe_colors'])

def smooth_sum(arr, n_times= 100):
    arr = np.array(arr, dtype = float)
    for e_ind in np.arange(n_times):
        arr = np.insert(arr, 0, arr[0])
        arr = np.append(arr, arr[-1] )
        arr = sp.signal.convolve( arr, [0.25,0.5,0.25], mode = "same" )[1:-1]
#         arr = sp.signal.convolve( arr, [0.25,0.5,0.25] )[1:-1]
    return(arr)


def rand_jitter(arr, stdev = 0):
    """
    function add random jitter to an array of values
    """
    # stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


class PlottingGenomeWide(object):
    """
    Class function to different plotting options

    1. Density plot for given positions.
    2. Heatmap

    """

    def __init__(self, ref_genome_class, min_chr_length = 1000000):
        assert type(ref_genome_class) is genome.GenomeClass, "provide a genome class"
        self.genome = ref_genome_class

        self.chr_info = pd.DataFrame({"chr":self.genome.chrs})
        self.chr_info['len'] = self.genome.golden_chrlen
        self.chr_info['mid'] = (np.array(self.genome.golden_chrlen)/2).astype(int)
        self.chr_info['chr_ind_start'] = np.array(self.genome.chr_inds)[:-1]
        self.chr_info['chr_ind_end'] = np.array(self.genome.chr_inds)[1::]
        log.warn("filtering out genome scaffolds less than %s" % min_chr_length)
        self.chr_info = self.chr_info[self.chr_info['len'] > min_chr_length]

    def patches_from_df(self, df, y_min=0, y_max=1, axs = None, plt_options = None):
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
        
        Create rectangular patches in a matplotlib Axes from a DataFrame with 'start' and 'end' columns.

        Parameters:
        - axs : matplotlib.axes.Axes
            The axes to which patches will be added.
        - df : pandas.DataFrame
            A dataframe containing 'start' and 'end' columns for x-axis ranges.
        - y_min : float, default=0
            The bottom of the rectangle in y-axis units.
        - y_max : float, default=1
            The top of the rectangle in y-axis units.
        - color : str, default='lightgray'
            Color of the patches.
        - alpha : float, default=0.5
            Transparency of the patches.

        Returns:
        - patches : list of matplotlib.patches.Rectangle
            List of created patches.
        """
        import matplotlib.patches as mpatches
    
        if axs is None:
            axs = plt.gca()
        
        if plt_options is None:
            plt_options = {}
        if "color" not in plt_options.keys():
            plt_options['color'] = 'skyblue'
        if "alpha" not in plt_options.keys():
            plt_options['alpha'] = 0.5
        if "jitter_stdev" not in plt_options.keys():
            plt_options['jitter_stdev'] = 0
        
        if 'boxstyle' not in plt_options.keys():
            plt_options['boxstyle'] = "rarrow" ## ["rarrow","square"] for an arrow
        

        patches = []
        for _, row in df.iterrows():
            # rect = mpatches.Rectangle((row['start'], rand_jitter([y_min], plt_options['jitter_stdev'])[0]), row['end'] - row['start'], y_max - y_min, color=plt_options['color'], alpha=plt_options['alpha'])
            rect = mpatches.FancyBboxPatch(
                (row['start'], rand_jitter([y_min], plt_options['jitter_stdev'])[0]), row['end'] - row['start'], y_max - y_min,  # x, y, width, height
                boxstyle=plt_options['boxstyle'],  # arrow on right
                facecolor=plt_options['color'],
                edgecolor=plt_options['color'],
                alpha = plt_options['alpha']
            )

            axs.add_patch(rect)
            patches.append(rect)
        return patches
    
    def lines_from_df(self, df, y_min = 0, y_max = 1, axs = None, plt_options = None):
        """
        Create line collections from a DataFrame with 'start' and 'end' columns.

        Parameters:
        - df : pandas.DataFrame
            A dataframe containing 'start' and 'end' columns for x-axis ranges.
        - y_positions : dict
            A dictionary mapping chromosome names to y-axis positions.
        - axs : matplotlib.axes.Axes, optional
            The axes to which the lines will be added. If None, uses the current axes.
        - plt_options : dict, optional
            Additional options for line plotting (e.g., color, linewidth).

        Returns:
        - collections : list of matplotlib.collections.LineCollection
            List of created line collections.
        """
        from matplotlib.collections import LineCollection

        if axs is None:
            axs = plt.gca()
        
        if plt_options is None:
            plt_options = {}
        if "color" not in plt_options.keys():
            plt_options['color'] = 'black'
        if "linewidth" not in plt_options.keys():
            plt_options['linewidth'] = 1.0
        
        lines = [((row['start'], y_min), (row['start'], y_max)) for _, row in df.iterrows()]
        collection = LineCollection(lines, colors=plt_options['color'], linewidths=plt_options['linewidth'])
        axs.add_collection(collection)
        
        return collection

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

    def density_line_plot_positions(self, bed_pos_df, echr = "Chr1", axs = None, plt_options = {}):
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
        if "line" not in plt_options.keys():
            plt_options['line'] = True
        if "ylabel" not in plt_options.keys():
            plt_options['ylabel'] = "density"
        if "xlabel" not in plt_options.keys():
            plt_options['xlabel'] = "windows"

        if axs is None:
            axs = plt.gca()

        data_window_density = pd.DataFrame(columns = ['chr', 'start', 'end', 'npos'])
        for ef_bin in self.genome.iter_positions_in_windows(bed_pos_df, window_size = plt_options['window_size']):
            data_window_density = pd.concat([data_window_density, pd.DataFrame({"chr": ef_bin[0], "start": ef_bin[1][0], "end": ef_bin[1][1], "npos": len(ef_bin[2]) }, index=[0])  ], ignore_index=True)

        if plt_options['nsmooth'] is not None:
            for ef_density in data_window_density.groupby('chr'):
                data_window_density.loc[ef_density[1].index, 'npos'] = smooth_sum(ef_density[1]['npos'], plt_options['nsmooth']) 

        self.manhattan_plot( 
            self.genome.get_genomewide_inds( data_window_density.iloc[:,[0,1,2]] ), 
            data_window_density['npos'].values, 
            plt_options=plt_options,
            axs = axs
        )

    def generate_heatmap_genomewide(self, score_df, plt_options = {}, axs = None, **kwargs):
        """
        Function to generate a genome wide heatmap
            plots chromosomes separately
            boxes for pericentromers

        """
        chr_info = self.chr_info.copy()
        assert isinstance(score_df, pd.DataFrame), "provide a dataframe object"

        if "lims" not in plt_options.keys():
            plt_options['lims'] = (np.nanmin(score_df.values), np.nanmax(score_df.values))
        if "xlabel" not in plt_options.keys():
            plt_options['xlabel'] = ""
        if "ylabel" not in plt_options.keys():
            plt_options['ylabel'] = ""
        if "yticklabels" not in plt_options.keys():
            plt_options['yticklabels'] = False
        if "title" not in plt_options.keys():
            plt_options['title'] = ""
        if "cmap" not in plt_options.keys():
            plt_options['cmap'] = "Reds"
        if "chr_line_vertical" not in plt_options.keys():
            plt_options['chr_line_vertical'] = True
        if "chr_line_horizontal" not in plt_options.keys():
            plt_options['chr_line_horizontal'] = False
        if "chr_line_centro" not in plt_options.keys():
            plt_options['chr_line_centro'] = True
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
        
        sns.heatmap(
            score_df_new,
            vmin=plt_options['lims'][0], 
            vmax=plt_options['lims'][1], 
            cmap=plt_options['cmap'],
            ax = axs,
            xticklabels=False,
            yticklabels=plt_options['yticklabels'],
            **kwargs
            # cbar_kws=dict(use_gridspec=False,location="bottom")
        )
        axs.set_xlabel(plt_options['xlabel'])
        axs.set_ylabel(plt_options['ylabel'])
        axs.set_title(plt_options['title'])
        
        if plt_options['chr_line_vertical']:
            chr_end_ix = np.searchsorted( score_df_new.columns.values, self.genome.get_genomewide_inds(chr_info.loc[:,['chr','len']]), "right" )[0:-1]
            for e_chr_end_ix in chr_end_ix:
                axs.plot( (e_chr_end_ix, e_chr_end_ix), (0,score_df_new.shape[0]), '-', c = plt_options['col_chr_line'] )

        if plt_options['chr_line_horizontal']:
            chr_end_ix = np.searchsorted( score_df_new.index.values, self.genome.get_genomewide_inds(chr_info.loc[:,['chr','len']]), "right" )[0:-1]
            for e_chr_end_ix in chr_end_ix:
                axs.plot( (0,score_df_new.shape[0]), (e_chr_end_ix, e_chr_end_ix), '-', c = plt_options['col_chr_line'] )    
        
        if 'centro_mid' in self.genome.__dict__.keys() and plt_options['chr_line_centro']:
            centro_df = pd.DataFrame({"chr":self.genome.chrs, "pos": self.genome.centro_mid})
            centro_df = centro_df[centro_df['pos'] > 0]
            chr_centro_mid_ix = np.searchsorted( score_df_new.columns.values, self.genome.get_genomewide_inds( centro_df ), "right" )
            for e_chr_pos_ix in chr_centro_mid_ix:
                axs.plot( (e_chr_pos_ix, e_chr_pos_ix), (0,score_df_new.shape[0]), '--', c = plt_options['col_centro_line'] )


        chr_mid_ix = np.searchsorted( score_df_new.columns.values, self.genome.get_genomewide_inds(chr_info.loc[:,['chr','mid']]), "right" )
        axs.set_xticks( chr_mid_ix )
        axs.set_xticklabels( chr_info['chr'] )
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
        if "gap" not in plt_options.keys():
            plt_options['gap'] = 10000000
        
        if "ylim" not in plt_options.keys():
            plt_options['ylim'] = (np.nanmin(y_ind), np.nanmax(y_ind))
        if "xlim" not in plt_options.keys():
            plt_options['xlim'] = (0, chr_info['chr_ind_end'].iloc[-1] + (plt_options['gap'] * (chr_info.shape[0] - 1))  )

        if "ylabel" not in plt_options.keys():
            plt_options['ylabel'] = ""
        if "thres" not in plt_options.keys():
            plt_options['thres'] = None
        if "size" not in plt_options.keys():
            plt_options['size'] = 6
        
        if "plt_xindex" not in plt_options.keys():
            plt_options['plt_xindex'] = False
        
        if "nsmooth" not in plt_options.keys():
            plt_options['nsmooth'] = 0
        
        if 'line' not in plt_options.keys():
            plt_options['line'] = False
        
        if axs is None:
            axs = plt.gca()
        
        
        for echr in chr_info.iterrows():
            t_chr_ix = np.where((x_ind <= echr[1]['chr_ind_end'] ) & ( x_ind > echr[1]['chr_ind_start'] ))[0]
            chr_info.loc[echr[0], 'rel_ind_start'] = t_chr_ix[0] + (plt_options['gap'] * echr[0])
            chr_info.loc[echr[0], 'rel_ind_mid'] = int(t_chr_ix.shape[0] / 2)
            if plt_options['plt_xindex']:
                t_x_ind = chr_info.loc[echr[0], 'rel_ind_start'] + np.arange(t_chr_ix.shape[0])
            else:
                t_x_ind = x_ind[t_chr_ix] + (plt_options['gap'] * echr[0])

            if plt_options['line']:
                axs.plot(t_x_ind, smooth_sum(y_ind[t_chr_ix], plt_options['nsmooth']), '-', color = echr[1]['color'], **kwargs)
            else:
                axs.scatter(t_x_ind, y_ind[t_chr_ix], s = plt_options['size'], c = echr[1]['color'], **kwargs) 
        
        if plt_options['plt_xindex']:
            axs.set_xticks( chr_info['rel_ind_mid'].values + chr_info['rel_ind_start'].values )
            plt_options['xlim'] = (0, x_ind.shape[0] + (plt_options['gap'] * (chr_info.shape[0] - 1))  )
        else:
            axs.set_xticks( chr_info['chr_ind_start'].values + (np.arange(chr_info.shape[0]) * plt_options['gap']) + chr_info['mid'].values )
        axs.set_xticklabels( chr_info['chr'].values )
        if plt_options['thres'] is not None:
            axs.plot(plt_options['xlim'], (plt_options['thres'], plt_options['thres']), "--", color = "gray")
        
        axs.set_xlabel( plt_options['ylabel'] )
        axs.set_ylim( plt_options['ylim'] )
        axs.set_xlim( plt_options['xlim'] )
        return(axs)

    def gwas_peaks_matrix(self, x_ind, y_ind, peak_heights = None, plt_options = {}, legend_scale = None, **kwargs):
        """
        given indices for x axis and y axis. this function plots our favorite matrix plot
        
        """
        chr_info = self.chr_info.copy()

        if "ylabel" not in plt_options.keys():
            plt_options['ylabel'] = ""
        if "xlabel" not in plt_options.keys():
            plt_options['xlabel'] = ""
        if "color" not in plt_options.keys():
            plt_options['color'] = "#d8b365"
        if "cmap" not in plt_options.keys():
            plt_options['cmap'] = (None, "BuGn") ### Chooses palettable.colorbrewer.sequential.BuGn_7.colors
        if "hist_color" not in plt_options.keys():
            plt_options['hist_color'] = "#8c510a"
        if "alpha" not in plt_options.keys():
            plt_options['alpha'] = 0.1
        if "height" not in plt_options.keys():
            plt_options['height'] = 12
        # if "ratio" not in plt_options.keys():
        plt_options['ratio'] = 1

        if peak_heights is not None:
            if legend_scale is None: 
                legend_scale = [np.min(peak_heights), np.max(peak_heights)]
            plt_options['color'] = np_scale_colors(legend_scale[0], legend_scale[1], peak_heights, safe_colors =  plt_options['cmap'][0])

        plt_limits = (1, chr_info['chr_ind_end'].values[-1])

        if type(plt_options['color']) is np.ndarray:
            p = sns.jointplot(
                x = x_ind, 
                y = y_ind, 
                marginal_kws={"bins": np.linspace(1, plt_limits[1], 250),"color": plt_options['hist_color'] }, 
                xlim = plt_limits, 
                ylim = plt_limits, 
                kind = "scatter", 
                alpha = plt_options['alpha'], 
                joint_kws={"s": 8}, 
                height = plt_options['height'], 
                ratio = plt_options['ratio'], 
                **kwargs
            )
            p.ax_joint.cla()
            #for i in range(len(x_ind)):
            p.ax_joint.scatter(x = x_ind, y = y_ind, c=plt_options['color'], s = 8)
        else:
            p = sns.jointplot(
                x = x_ind, 
                y = y_ind, 
                marginal_kws={"bins": np.linspace(1, plt_limits[1], 250),"color": plt_options['hist_color'] }, 
                xlim = plt_limits, 
                ylim = plt_limits, 
                kind = "scatter", 
                color = plt_options['color'], 
                alpha = plt_options['alpha'], 
                joint_kws={"s": 8}, 
                height = plt_options['height'], 
                ratio = plt_options['ratio'], 
                **kwargs
            )
        p.ax_marg_y.remove()
        p.ax_joint.plot( (0, 0) , plt_limits, '-', color = "gray")
        p.ax_joint.plot( plt_limits, (0, 0), '-', color = "gray")

        for echr in chr_info.iterrows():
            p.ax_joint.plot( (echr[1]['chr_ind_end'] , echr[1]['chr_ind_end']) , plt_limits, '-', color = "gray")
            p.ax_joint.plot( plt_limits, (echr[1]['chr_ind_end'] , echr[1]['chr_ind_end']), '-', color = "gray")

        if 'centro_mid' in self.genome.__dict__.keys():
            chr_info = pd.merge(chr_info, pd.DataFrame({"chr": self.genome.chrs, "centro_mid": self.genome.centro_mid}), left_on="chr", right_on="chr" )
            for echr in chr_info.iterrows():
                p.ax_joint.plot( (echr[1]['chr_ind_start'] + echr[1]['centro_mid'] , echr[1]['chr_ind_start'] + echr[1]['centro_mid']), plt_limits, ':k', color = "gray")
                p.ax_joint.plot( plt_limits, (echr[1]['chr_ind_start'] + echr[1]['centro_mid'] , echr[1]['chr_ind_start'] + echr[1]['centro_mid']), ':k', color = "gray")
        
        p.ax_joint.set_xticks( chr_info['chr_ind_start'] + chr_info['mid'] )
        p.ax_joint.set_xticklabels( chr_info['chr'] )
        p.ax_joint.set_yticks( chr_info['chr_ind_start'] + chr_info['mid'] )
        p.ax_joint.set_yticklabels( chr_info['chr'] )
        
        if legend_scale is not None:
            p.ax_marg_x.remove()
            norm = plt.Normalize(legend_scale[0], legend_scale[1])
            sm = plt.cm.ScalarMappable(cmap=plt_options['cmap'][1], norm=norm)
            sm.set_array([])
            p.ax_joint.figure.colorbar(sm, shrink = 0.5)
        
        p.ax_joint.set_xlabel( plt_options['xlabel'] )
        p.ax_joint.set_ylabel( plt_options['ylabel'] )
        return(p)





