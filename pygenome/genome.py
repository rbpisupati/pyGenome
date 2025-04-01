# Loading genome data
import logging
import numpy as np
import pandas as pd
import string
from pyfaidx import Faidx, Fasta
import sys, os.path

log = logging.getLogger(__name__)

def die(msg):
  sys.stderr.write('Error: ' + msg + '\n')
  sys.exit(1)


def iter_chr_positions_windows(reference_bed, query_positions, window_size):
    assert isinstance(query_positions, np.ndarray), "please provide a numpy array"
    assert isinstance(reference_bed, list), "please provide a list for position start and end"
    assert len(reference_bed) == 2, "just start and end please, [start, end]"
    filter_pos_ix = np.arange( np.searchsorted(query_positions, reference_bed[0], 'left'  ), np.searchsorted(query_positions, reference_bed[1], 'right') )
    ind = 0
    # if len(filter_pos_ix) == 0:
    #     return( (reference_bed, filter_pos_ix) )
    for t in range(reference_bed[0], reference_bed[1], window_size):
        skipped = True
        result = []
        bin_bed = [int(t), min(reference_bed[1], int(t) + window_size - 1)]
        for epos in query_positions[ind:]:
            if epos >= bin_bed[0]:
                if epos <= bin_bed[1]:
                    result.append(filter_pos_ix[ind])
                elif epos > bin_bed[1]:
                    skipped = False
                    yield((bin_bed, result))
                    break
                ind = ind + 1
        if skipped:
            yield((bin_bed, result))


class GenomeClass(object):
    ## coordinates for ArabidopsisGenome using TAIR 10

    def __init__(self, ref_genome = "at_tair10"):
        self.genome_str = ref_genome
        if ref_genome == "at_tair10":
            self.chrs = ['Chr1','Chr2','Chr3','Chr4','Chr5']
            self.def_color = ["#1f78b4", "#33a02c", "#1f78b4", "#33a02c", "#1f78b4"]
            self.real_chrlen = [34964571, 22037565, 25499034, 20862711, 31270811]
            self.golden_chrlen = [30427671, 19698289, 23459830, 18585056, 26975502]
            self.centro_start = [14364752, 3602775, 12674550, 2919690, 11668616]
            self.centro_end   = [15750321, 3735247, 13674767, 4011692, 12082583]
            self.centro_mid = np.add(self.centro_start, self.centro_end)/2
        elif os.path.exists(ref_genome):
            ## Provide a fasta file to check for genome lengths etc
            self.fasta = Fasta(ref_genome)
            self.fasta_file = ref_genome
            genome = Faidx(ref_genome).index
            self.chrs = []
            self.real_chrlen = []
            for key,value in genome.items():
                self.chrs.append(key)
                self.real_chrlen.append(genome[key].rlen)
            # self.chrs = np.array(list(genome.keys())).astype('U13')
            # self.real_chrlen = [ genome[ef].rlen for ef in self.chrs]
            self.golden_chrlen = self.real_chrlen
        self.chr_inds = np.append(0, np.cumsum(self.golden_chrlen))        

    def load_bed_ids_str(self, **kwargs):
        for req_name in kwargs:
            req_bed_df = pd.read_csv( kwargs[req_name], header=None, sep = "\t", dtype = {0: str, 1: int, 2: int} )
            setattr(self, req_name, req_bed_df)
            setattr(self, req_name + "_str", np.array(req_bed_df.iloc[:,0] + ',' + req_bed_df.iloc[:,1].map(str) + ',' + req_bed_df.iloc[:,2].map(str), dtype="str") )

    def determine_bed_from_araportids(self, name, araportids, return_bed=False):
        assert type(araportids) is pd.core.series.Series, "please provide a pandas series object"
        assert hasattr(self, name), "please load required bed file using 'load_bed_ids_str' function. ex., ARAPORT11/Araport11_GFF3_genes_201606.bed"
        bed_str = np.zeros(0, dtype="float")
        bed_ix = np.zeros(0, dtype="int")
        for ei in araportids:
            t_ind = np.where( self.__getattribute__( name ).iloc[:,3] == ei )[0]
            if len(t_ind) == 0:
                bed_str = np.append(bed_str, '')
            else:
                bed_str = np.append( bed_str, self.__getattribute__( name + "_str" )[t_ind[0]] )
                bed_ix = np.append(bed_ix, t_ind[0])
        if return_bed:
            bed_str = self.__getattribute__( name ).iloc[bed_ix,]
            bed_str = bed_str.rename(columns={0:"chrom", 1:"start", 2:"end", 3:'id', 4:'score',5:'strand'})
        return( bed_str )

    def get_chr_ind(self, echr):
        real_chrs = np.array( [ ec.replace("Chr", "").replace("chr", "") for ec in self.chrs ] )
        if type(echr) is str or type(echr) is np.str_:
            echr_num = str(echr).replace("Chr", "").replace("chr", "")
            if len(np.where(real_chrs == echr_num)[0]) == 1:
                return(np.where(real_chrs == echr_num)[0][0])
            else:
                return(None)
        echr_num = np.unique( np.array( echr ) )
        ret_echr_ix = np.zeros( len(echr), dtype="int8" )
        for ec in echr_num:
            t_ix = np.where(real_chrs ==  str(ec).replace("Chr", "").replace("chr", "") )[0]
            ret_echr_ix[ np.where(np.array( echr ) == ec)[0] ] = t_ix[0]
        return(ret_echr_ix)

    def get_genomewide_inds(self, df_str, gap = 0, str_split = ","):
        """
        This is the function to give the indices of the genome when you give a bed file.

        Inputs:
            Either a pandas series, if so provide a string to split the series object to chromsome ID and position
            Or a pandas dataframe with either two or three columns (chromosome, start and end)
        
        Output: 
            Numpy array for the indices
        """
        ### 
        assert type(df_str) is pd.core.series.Series or type(df_str) is pd.core.frame.DataFrame, "please input pandas dataframe or series object"
        if type(df_str) is pd.core.series.Series:
            df_str = df_str.copy()
            df_str = df_str.str.split( str_split, expand = True )
        ## here first column is chr and second is position
        if df_str.shape[1] == 2:
            f_df_str = pd.DataFrame({
                "chr": df_str.iloc[:,0],
                "start": df_str.iloc[:,1].astype(int),
                "end": df_str.iloc[:,1].astype(int)
            }).loc[:,['chr', 'start', 'end']]
        elif df_str.shape[1] >= 3:
            f_df_str = df_str.iloc[:,[0,1,2]].copy()
            f_df_str.iloc[:,1] = f_df_str.iloc[:,1].astype(int)
            f_df_str.iloc[:,2] = f_df_str.iloc[:,2].astype(int)
        df_chr_inds = self.get_chr_ind(f_df_str.iloc[:,0])
        pos_ix = self.chr_inds[ df_chr_inds ]   ### Start of chromosome
        pos_ix += gap * df_chr_inds             ## Adding required gap specific for each chromsome
        pos_ix += f_df_str.iloc[:,[1,2]].mean(1).astype(int).values ### adding in the positions now
        return( pos_ix )

    def get_genomic_position_from_ind(self, ind):
        ## This function is just opposite to the one before.
        # Given an index, we should get the chromosomes and position
        chr_idx = np.searchsorted( self.chr_inds, ind ) - 1
        ind_pos = ind - self.chr_inds[chr_idx]
        ind_chr = np.array(self.chrs)[chr_idx]
        return( pd.DataFrame( np.column_stack((ind_chr, ind_pos)), columns = ["chr", "pos"]  ) )

    def iter_windows_echr(self, echr, window_size, overlap=0):
        chr_ind = self.get_chr_ind(echr)
        if overlap >= window_size:
            raise(NotImplementedError)
        if overlap > 0:
            for x in range(1, self.golden_chrlen[chr_ind], overlap):
                yield([x, x + window_size - 1])
        else:
            for x in range(1, self.golden_chrlen[chr_ind], window_size):
                yield([x, x + window_size - 1])

    def iter_windows(self, window_size, overlap=0):
        for echr, echrlen in zip(self.chrs, self.golden_chrlen):
            echr_windows = self.iter_windows_echr(echr, window_size, overlap)
            for ewin in echr_windows:
                yield([echr, ewin[0], ewin[1]])

    def iter_positions_in_windows(self, query_bed_df, window_size):
        assert isinstance(query_bed_df, pd.DataFrame), "provide a dataframe object" 
        for echr, echrlen in zip(self.chrs, self.golden_chrlen):
            echr_query_start = np.searchsorted(query_bed_df.iloc[:,0], echr, 'left'  )
            echr_query_ix = np.arange(echr_query_start, np.searchsorted(query_bed_df.iloc[:,0], echr, 'right') )
            echr_windows = iter_chr_positions_windows([1, echrlen], query_bed_df.iloc[echr_query_ix,1].values, window_size)
            for ewin in echr_windows:
                yield([echr, ewin[0], ewin[1] + echr_query_start])


    def estimated_cM_distance(self, snp_position):
        ## snp_position = "Chr1,150000" or "Chr1,1,300000"
        # Data based on
        #Salome, P. A., Bomblies, K., Fitz, J., Laitinen, R. A., Warthmann, N., Yant, L., & Weigel, D. (2011)
        #The recombination landscape in Arabidopsis thaliana F2 populations. Heredity, 108(4), 447-55.
        assert isinstance(snp_position, basestring)
        assert len(snp_position.split(",")) >= 2
        if len(snp_position.split(",")) == 2:
            snp_position = [snp_position.split(",")[0], int(snp_position.split(",")[1])]
        elif len(snp_position.split(",")) == 3:
            snp_position = [snp_position.split(",")[0], (int(snp_position.split(",")[1]) + int(snp_position.split(",")[2])) / 2 ]
        mean_recomb_rates = [3.4, 3.6, 3.5, 3.8, 3.6]  ## cM/Mb
        chr_ix = self.get_chr_ind( snp_position[0] )
        return( mean_recomb_rates[chr_ix] * snp_position[1] / 1000000 )

    def get_mc_context(self, cid, pos):
        dnastring_pos = self.fasta[cid][pos:pos+3].seq.encode('ascii').upper().decode("utf-8")
        ## make sure you can have to identify strand here
        dnastring_neg = self.fasta[cid][pos-2:pos+1].seq.encode('ascii').upper().decode("utf-8")  ## Changed the position, different from forward
        dnastring_neg = get_reverse_complement(dnastring_neg)
        if dnastring_pos[0].upper() == 'C' and dnastring_neg[0].upper() != 'C':
            strand = '0'
            dnastring = dnastring_pos
        elif dnastring_pos[0].upper() != 'C' and dnastring_neg[0].upper() == 'C':
            strand = '1'
            dnastring = dnastring_neg
        else:
            return((dnastring_pos, dnastring_neg, None))
        if dnastring[1].upper() == 'G':
            dna_context = ["CG",0]
        elif dnastring[2].upper() == 'G':
            dna_context = ["CHG",1]
        elif dnastring:
            dna_context = ["CHH",2]
        return((dnastring, dna_context, strand))

    def get_inds_overlap_region(self, region_bed_df, name="genes", request_ind = None, g = None):
        import pybedtools as pybed
        assert hasattr(self, name), "please load required bed file using 'load_bed_ids_str' function. ex., ARAPORT11/Araport11_GFF3_genes_201606.bed"
        assert type(region_bed_df) is pd.core.frame.DataFrame, "please provide a pd.DataFrame object"
        if request_ind is None:
            gene_bed = pybed.BedTool.from_dataframe( self.__getattribute__(name).iloc[:,[0,1,2]] )
        else:
            gene_bed = pybed.BedTool.from_dataframe( self.__getattribute__(name).iloc[:,[0,1,2]].iloc[[request_ind]]  )
        region_bed = pybed.BedTool.from_dataframe(region_bed_df.iloc[:,[0,1,2]])
        region_bed_str = np.array(region_bed_df.iloc[:,0].map(str) + "," + region_bed_df.iloc[:,1].map(str) + "," +  region_bed_df.iloc[:,2].map(str), dtype="str")
        ## Just taking first three columns for bedtools
        inter_region_bed = region_bed.intersect(gene_bed, wa=True)
        inter_gene_bed = gene_bed.intersect(region_bed, wa=True)
        if inter_region_bed.count() == 0:   ## Return if there are no matching lines.
            return(None)
        inter_region_bed = inter_region_bed.to_dataframe() ## wa is to return the entire bed.
        inter_region_bed_str = np.array(inter_region_bed.iloc[:,0].map(str) + "," + inter_region_bed.iloc[:,1].map(str) + "," +  inter_region_bed.iloc[:,2].map(str), dtype="str")
        inter_gene_bed = inter_gene_bed.to_dataframe()
        inter_gene_bed_str = np.array(inter_gene_bed.iloc[:,0].map(str) + "," + inter_gene_bed.iloc[:,1].map(str) + "," +  inter_gene_bed.iloc[:,2].map(str), dtype="str")
        out_dict = { "region_ix": np.where( np.in1d(region_bed_str, inter_region_bed_str ) )[0] }
        out_dict['ref_ix'] = np.where( np.in1d(self.__getattribute__(name + '_str'), inter_gene_bed_str ) )[0]
        return(out_dict)

    def search_sort_centro_indices(self, required_region, highlighted_bed_df="centro"):
        if highlighted_bed_df == "centro":
            highlighted_bed_df = pd.DataFrame( {"chr": self.chrs, "pos": self.centro_mid} )
        elif highlighted_bed_df == "ends":
            highlighted_bed_df = pd.DataFrame( {"chr": self.chrs, "pos": self.golden_chrlen} )
        assert type(highlighted_bed_df) is pd.core.frame.DataFrame, "please provide a pd.DataFrame object"
        region_bed_ix = self.get_genomewide_inds( required_region )
        return( np.searchsorted(region_bed_ix, self.get_genomewide_inds( highlighted_bed_df ) ) )

    def add_upstream_downstream_binbed(self, binbed, updown = 50000):
        """
        A function to add upstream and downsteam 
        input:
            1. binbed: "Chr1,1,1000"
            2. updown: 50kb or so
        output:
            A pandas series adding in given updown
        """
        binbed_str = pd.Series( binbed ).str.split( ",", expand = True )
        t_chr_maxlen = pd.Series(self.golden_chrlen)[binbed_str.iloc[:,0].apply(self.get_chr_ind)]
        binbed_str.iloc[:,1] = pd.DataFrame( {'a': binbed_str.iloc[:,1].astype(int) - updown, 'b': 0} ).max(axis =1)
        binbed_str.iloc[:,2] = pd.DataFrame( np.column_stack((binbed_str.iloc[:,2].astype(int) + updown, t_chr_maxlen)) ).min(axis =1) 
        return( binbed_str.iloc[:,0] + "," + binbed_str.iloc[:,1].astype(str) + "," + binbed_str.iloc[:,2].astype(str) )

    def identify_all_cytosines_gene(self, gene_id):
        bed_str = bed_str.split(",")
        bed_str = [bed_str[0], int(bed_str[1]), int(bed_str[2])]
        bed_seq = self.fasta[bed_str[0]][bed_str[1]:bed_str[2]].seq
        return(bed_seq)


    def getGC(self, bed_str):
        from collections import defaultdict
        bed_str = bed_str.split(",")
        bed_str = [bed_str[0], int(bed_str[1]), int(bed_str[2])]
        bed_seq = self.fasta[bed_str[0]][bed_str[1]:bed_str[2]].seq.upper()
        char_count = defaultdict(int)
        for char in bed_seq:
            char_count[char] += 1
        gc_sum = char_count['C'] + char_count['G']
        at_sum = char_count['A'] + char_count['T']
        return( (100 * gc_sum) / (gc_sum + at_sum)  )

    def output_stats(self, output_file = None):
        """
        """
        contig_lens, scaffold_lens, gc_cont = self.read_genome()
        contig_stats = self._calculate_stats(contig_lens, gc_cont)
        # scaffold_stats = self._calculate_stats(scaffold_lens, gc_cont)
        # stat_output = {'Contig Stats': contig_stats,
                        # 'Scaffold Stats': scaffold_stats}
        return(contig_stats)
    
    def _fasta_iter(self, fasta_file=None):
        """Takes a FASTA file, and produces a generator of Header and Sequences.
        This is a memory-efficient way of analyzing a FASTA files -- without
        reading the entire file into memory.

        Parameters
        ----------
        fasta_file : str
            The file location of the FASTA file

        Returns
        -------
        header: str
            The string contained in the header portion of the sequence record
            (everything after the '>')
        seq: str
            The sequence portion of the sequence record
        """
        from itertools import groupby
        if fasta_file is None:
            fasta_file = self.fasta_file
        fh = open(fasta_file)
        fa_iter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in fa_iter:
            # drop the ">"
            header = next(header)[1:].strip()
            # join all sequence lines to one.
            seq = "".join(s.upper().strip() for s in next(fa_iter))
            yield header, seq

    def read_genome(self, fasta_file=None):
        """Takes a FASTA file, and produces 2 lists of sequence lengths. It also
        calculates the GC Content, since this is the only statistic that is not
        calculated based on sequence lengths.

        Parameters
        ----------
        fasta_file : str
            The file location of the FASTA file

        Returns
        -------
        contig_lens: list
            A list of lengths of all contigs in the genome.
        scaffold_lens: list
            A list of lengths of all scaffolds in the genome.
        gc_cont: float
            The percentage of total basepairs in the genome that are either G or C.
        """
        if fasta_file is None:
            fasta_file = self.fasta_file
        gc = 0
        total_len = 0
        contig_lens = []
        scaffold_lens = []
        for _, seq in self._fasta_iter(fasta_file):
            scaffold_lens.append(len(seq))
            if "NN" in seq:
                contig_list = seq.split("NN")
            else:
                contig_list = [seq]
            for contig in contig_list:
                if len(contig):
                    gc += contig.count('G') + contig.count('C')
                    total_len += len(contig)
                    contig_lens.append(len(contig))
        gc_cont = (gc / total_len) * 100
        return contig_lens, scaffold_lens, gc_cont

    @staticmethod
    def _calculate_stats(seq_lens, gc_cont):
        """
        calculate_stats takes a list of sequence lengths and the GC content
        and returns a dictionary of statistics.
        """
        stats = {}
        seq_array = np.array(seq_lens)
        stats['sequence_count'] = seq_array.size
        stats['gc_content'] = gc_cont
        sorted_lens = seq_array[np.argsort(-seq_array)]
        stats['longest'] = int(sorted_lens[0])
        stats['shortest'] = int(sorted_lens[-1])
        stats['median'] = np.median(sorted_lens)
        stats['mean'] = np.mean(sorted_lens)
        stats['total_bps'] = int(np.sum(sorted_lens))
        csum = np.cumsum(sorted_lens)
        for level in [10, 20, 30, 40, 50]:
            nx = int(stats['total_bps'] * (level / 100))
            csumn = min(csum[csum >= nx])
            l_level = int(np.where(csum == csumn)[0])
            n_level = int(sorted_lens[l_level])

            stats['L' + str(level)] = l_level
            stats['N' + str(level)] = n_level
        return stats



def scale_colors(minval, maxval, val, safe_colors = None):
    import palettable
    if safe_colors is None:
        safe_colors = palettable.colorbrewer.sequential.BuGn_7.colors
    EPSILON = sys.float_info.epsilon  # Smallest possible difference.
    i_f = float(val-minval) / float(maxval-minval) * (len(safe_colors)-1)
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    if f < EPSILON:
        ret_col = safe_colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = safe_colors[i], safe_colors[i+1]
        ret_col = int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))
    return('#%02x%02x%02x' % (ret_col[0], ret_col[1], ret_col[2]))

np_scale_colors = np.vectorize(scale_colors, excluded = ["safe_colors"])


def get_reverse_complement(seq):
    old_chars = "ACGT"
    replace_chars = "TGCA"
    tab = str.maketrans(old_chars,replace_chars)
    return(seq.translate(tab)[::-1])
