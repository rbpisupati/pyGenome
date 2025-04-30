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
            # randint = str(np.random.choice(1000,1)[0])
            fastq_entry = f"@{ef_info['name']}\n{ef_info['seq']}\n+\n{ef_info['qual']}\n"
            count += 1
            fastqfile.write(fastq_entry)
        
        print(f"Printed {count} reads to FASTQ file")
        fastqfile.close()
        
    def fetch_read_information(self, chr, start, end):
        """
        Get read information from BAM file.
        
        Args:
            chr (str): Chromosome name.
            start (int): Start position.
            end (int): End position.
        
        Returns:
            list: List of read information dictionaries.
        """
        reads = []
        read_seq = []
        for ef in self.bam.fetch(chr, start, end):
            # Skip unmapped reads
            if ef.is_unmapped:
                continue
            # Skip reads with no tags
            if ef.get_tag("tp") != 'P':
                print('care, there are non primary aligned reads in the bam file')
                continue
            reads.append(ef)
            read_seq.append( _get_read_seq_from_position(ef, start) )
            

        read_info = pd.DataFrame([ef.to_dict() for ef in reads])
        read_info.loc[:,'qry_length'] = [ef.infer_query_length() for ef in reads]
        read_info.loc[:,'qry_start'] = [ef.query_alignment_start for ef in reads]
        read_info.loc[:,'qry_end'] = [ef.query_alignment_end for ef in reads]
        read_info.loc[:,'ref_length'] = [ef.reference_length for ef in reads]
        read_info.loc[:,'ref_start'] = [ef.reference_start for ef in reads]
        read_info.loc[:,'ref_end'] = [ef.reference_end for ef in reads]
        read_info.loc[:,'qry_seq_from_start'] = read_seq

        return({'df':read_info, 'reads': reads})


def _get_read_seq_from_position(alignedread, ref_position):
    """
    Function to get the sequence from a given position in the reference genome
    """
    read_sequence = ''
    ## first check for the forward reads
    if alignedread.is_reverse:
        # Output reverse complement if the read is mapped in reverse
        # it does not matter if the read is reverse or not
        # the reference positions and query positions are the same
        qry_sequence = alignedread.query_sequence
    else:
        qry_sequence = alignedread.query_sequence
    
    ## read should be aligned until the break point or 5 base pairs before.
    overlap = alignedread.get_overlap(ref_position - 5, ref_position)
    # print(overlap)
    if overlap > 0:
        ## reads that overlap the breakpoint
        # now you have two scenarios
        # 1. reads that pass through the break point
        # 2. reads that are only mapped till the breakpoint or before 
        aligned_pairs = alignedread.get_aligned_pairs(matches_only = True)
        # print((aligned_pairs[0], aligned_pairs[-1]))

        # reads that pass through the break point have last match position higher than the position
        if aligned_pairs[-1][1] > ref_position:
            qry_break_ix = [t for t in aligned_pairs if t[1] == ref_position - 1]
            assert len(qry_break_ix) > 0, 'there must be an aligned position at the break point'
            # print(qry_break_ix)
            read_sequence = qry_sequence[qry_break_ix[0][0]:]
            
        else: 
            ## Here the read is aligned either until the breakpoint or before
            # we need to check if the read is long enough 
            # if alignedread.infer_query_length() - aligned_pairs[-1][0] > minimum_read_extend_length:
            ## the read is long enough to extend the breakpoint
            read_sequence = qry_sequence[aligned_pairs[-1][0]:]
        
    return(read_sequence)