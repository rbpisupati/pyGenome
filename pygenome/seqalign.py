import pymummer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

from . import genome

# Child class inheriting Nucmer runner
class Nucmer(pymummer.nucmer.Runner):
    
    def __init__(self, ref, query, outfile, min_length=2000, gap = 0):
        super().__init__(ref=ref, query = query, outfile=outfile)  # Calls Person's __init__()
        self.qry_fasta = genome.GenomeClass(query)
        self.ref_fasta = genome.GenomeClass(ref)
        self.outfile = outfile
        if not os.path.exists(outfile):
            self.run()
            self._log = subprocess.Popen("nucmer -p %s.nucmer %s %s " % (self.outfile, ref, query),  shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        self.result = self._get_nucmer_output(min_length=min_length, gap = gap)

    def _get_nucmer_output(self, min_length = 100, gap = 0):
        """
        get the output of the nucmer run in a dataframe
        """
        file_reader = pymummer.coords_file.reader(self.outfile)
        output_coords = pd.DataFrame(columns=['ref_name','ref_start','ref_end','qry_name','qry_start','qry_end','similarity','strand'])
        for ix, coord in enumerate(file_reader):
            ## remove self hits
            if coord.is_self_hit():
                continue
            if coord.hit_length_qry < min_length:
                continue
            if coord.hit_length_ref < min_length:
                continue
            
            output_coords.loc[ix,'hit_length_ref'] = coord.hit_length_ref
            output_coords.loc[ix,'ref_name'] = coord.ref_name
            output_coords.loc[ix,'qry_name'] = coord.qry_name
            output_coords.loc[ix,'similarity'] = coord.percent_identity
            output_coords.loc[ix,'strand'] = coord.on_same_strand()
            output_coords.loc[ix,['ref_start','ref_end']] =  [coord.ref_start, coord.ref_end]
            output_coords.loc[ix,['qry_start','qry_end']] = [coord.qry_start, coord.qry_end]
        ## get genome wide coordinates 
        output_coords['ref_start_ix'] = self.ref_fasta.get_genomewide_inds(output_coords.loc[:,['ref_name','ref_start']], gap = gap)
        output_coords['ref_end_ix'] = self.ref_fasta.get_genomewide_inds(output_coords.loc[:,['ref_name','ref_end']], gap = gap)
        output_coords['qry_start_ix'] = self.qry_fasta.get_genomewide_inds(output_coords.loc[:,['qry_name','qry_start']], gap = gap)
        output_coords['qry_end_ix'] = self.qry_fasta.get_genomewide_inds(output_coords.loc[:,['qry_name','qry_end']], gap = gap)

        return(output_coords)

    def plotly_output(self, output_df = None):
        """
        create a plotly figure 
        
        """
        import plotly.graph_objects as go

        if output_df is None:
            output_df = self.result

        lines = []
        for ef in output_df.iterrows():
            if ef[1].strand:
                col = '#2166ac'
            else:
                col = '#b2182b'
            lines.append(go.Scatter(
                x = [ef[1].ref_start_ix, ef[1].ref_end_ix],
                y = [ef[1].qry_start_ix, ef[1].qry_end_ix],
                name = ef[0],mode='lines',  # Draw lines
                line=dict(color=col),
                hovertext='\n'.join([f"{idx}: {val}" for idx, val in ef[1].iloc[[1,2,4,5,7,8]].items()]) ,
                showlegend=False
            ))
        
        ## adding horiontal lines for the query genome
        for ef_chr in self.qry_fasta.chr_inds:
            lines.append(go.Scatter(
                x=[0,self.ref_fasta.chr_inds[-1]],  # x-coordinates (start and end points for reference)
                y=[ef_chr,ef_chr],  # y-coordinates (constant for horizontal line)
                mode='lines',  # Connect the points with a line
                showlegend=False,
                line=dict(color='#525252', width=0.5)  # Customize line color and width
            ))
        ## adding vertical lines for the reference genome
        for ef_chr in self.ref_fasta.chr_inds:
            lines.append(go.Scatter(
                x=[ef_chr,ef_chr],  # x-coordinates (constant for vertical line)
                y=[0,self.qry_fasta.chr_inds[-1]],  # y-coordinates (start and end points for query)
                mode='lines',  # Connect the points with a line
                showlegend=False,
                line=dict(color='#525252', width=0.5)  # Customize line color and width
            ))
        
        # fig = go.Figure(data  = lines)
        return(lines)


        
    