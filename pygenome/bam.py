import numpy as np
import pandas as pd
import pysam
from . import genome

class BamFile(object):
    """Class for handling BAM files."""
    def __init__(self, bam_file, ref_fasta):
        """
        Initialize BamFile object.
        
        Args:
            bam_file (str): Path to BAM file.
            ref_fasta (str): Path to reference FASTA file.
        """
        self.bam_file = bam_file
        self.bam = pysam.AlignmentFile(bam_file, "rb")
        self.ref_fasta = genome.GenomeClass(ref_fasta)

    def bam_counts_across_windows(self, chr, window_size):
        """
        Plot coverage from BAM file.
        """
        # Get chromosome names and lengths
        # chr_ix = self.ref_fasta.get_chr_ind(chr)
        
        # Create a DataFrame to hold coverage data
        coverage_data = pd.DataFrame(columns=['chr', 'start', 'end', 'coverage'])

        for ef_bin in enumerate(self.ref_fasta.iter_windows_echr( chr, window_size = window_size )):
            coverage_data.loc[ef_bin[0],'chr'] = chr
            coverage_data.loc[ef_bin[0],'start'] = ef_bin[1][0]
            coverage_data.loc[ef_bin[0],'end'] = ef_bin[1][1]
            read_counts = 0

            # Iterate over each read in the BAM file for the given chromosome
            for read in self.bam.fetch(chr, ef_bin[1][0], ef_bin[1][1]):
                # Skip unmapped reads
                if read.is_unmapped:
                    continue
                # Increment the count for this window
                read_counts += 1
            
            coverage_data.loc[ef_bin[0],'coverage'] = read_counts
            # Uncomment the following line if you want to count reads using bam.count
            # Note: This is may not be an efficient way to count reads in a region
            # coverage_data.loc[ef_bin[0],'coverage'] = self.bam.count(chr, ef_bin[1][0], ef_bin[1][1])

        # For now, just return the DataFrame
        return coverage_data

    def to_fastq(self, chr, start, end, out_file, req_tag = None):
        """
        Convert BAM file to FASTQ format.
        
        Args:
            chr (str): Chromosome name.
            start (int): Start position.
            end (int): End position.
            out_file (str): Path to output FASTQ file.
        """
        if req_tag is None:
            check = False
        else:
            check = True
        
        fastqfile = open(out_file, "w")
        count = 0
        for ef in self.bam.fetch(chr, start, end):
            # Skip reads with no tags
            if check:
                if not ef.has_tag(req_tag):
                    continue 

            ef_info = ef.to_dict()
            fastq_entry = f"@{ef_info['name']}\n{ef_info['seq']}\n+\n{ef_info['qual']}\n"
            count += 1
            fastqfile.write(fastq_entry)
        
        print(f"Printed {count} reads to FASTQ file")
        fastqfile.close()
        

