import numpy as np
from itertools import groupby
import json
import sys

from . import genome


### Adapted from https://github.com/MikeTrizna/assembly_stats
class GenomeStats(genome.GenomeClass):
    """
    Class for calculating genome statistics.
    """

    def __init__(self, fasta_file):
        super().__init__(fasta_file)
        self.fasta_file = fasta_file

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
