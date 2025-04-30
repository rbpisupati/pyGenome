import subprocess
import os
import pandas as pd
import io

class SeqKitRunner:
    def __init__(self, seqkit_path='seqkit'):
        self.seqkit_path = seqkit_path

    def _run_command(self, command, shell = True):
        """Run a shell command and return the output or raise an error."""
        try:
            result = subprocess.run(command, shell = shell, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SeqKit error: {e.stderr.strip()}")

    def view(self, input_file):
        """Run 'seqkit view' to print sequence info."""
        cmd = [self.seqkit_path, 'view', input_file]
        return self._run_command(cmd)

    def grep(self, input_file, pattern, output_file=None, ignore_case=True):
        """Filter sequences by ID or name using 'seqkit grep'."""
        cmd = [self.seqkit_path, 'grep', '-p', pattern, input_file]
        if ignore_case:
            cmd.insert(2, '-i')
        if output_file:
            cmd += ['-o', output_file]
        return self._run_command(cmd)

    def subseq(self, input_file, region, output_file=None):
        """Extract subsequences using 'seqkit subseq'."""
        cmd = [self.seqkit_path, 'subseq', '-r', region, input_file]
        if output_file:
            cmd += ['-o', output_file]
        return self._run_command(cmd)

    def stats(self, input_file):
        """Return sequence statistics using 'seqkit stats'."""
        cmd = [self.seqkit_path, 'stats', input_file]
        # return  self._run_command(cmd, shell = False)
        return pd.read_csv( io.StringIO(  self._run_command(cmd, shell = False) ), delimiter=r"\s+" )

    def locate(self, input_file, seq):
        """Locate a sequence in a FASTA/Q file using 'seqkit locate'."""
        if os.path.isfile( input_file ):
            if os.path.splitext(input_file)[-1] == '.gz':
                # If the input file is gzipped, use zcat to decompress it
                cmd = 'zcat %s | %s locate -p %s ' % (input_file, self.seqkit_path, seq)
            else:
                cmd = "cat %s | %s locate -p %s " % (input_file, self.seqkit_path, seq)
        else:
            cmd = "echo '>input\n%s\n' | %s locate -p %s " % (input_file, self.seqkit_path, seq) 
            # ['echo', input_file, ' | ', self.seqkit_path, 'locate', '-p', seq]
        # return self._run_command(cmd)
        return pd.read_csv( io.StringIO( self._run_command(cmd) ), sep = r"\s+" )
