"""Run AUCpreD on individual sequences in trimmed alignments."""

import multiprocessing as mp
import os
import re
import subprocess

from src.utils import read_fasta


def run_cmd(OGid):
    msa = read_fasta(f'../../../data/alignments/fastas/{OGid}.afa')
    prefix = f'out/{OGid}/'

    if not os.path.exists(prefix):
        os.mkdir(prefix)

    for header, seq in msa:
        ppid = re.search(ppid_regex, header).group(1)
        seq = seq.translate({ord('-'): None, ord('.'): None})
        if len(seq) < 10000:  # AUCpreD uses PSIPRED which has a length limit of 10000
            with open(f'{prefix}/{ppid}.fasta', 'w') as file:
                seqstring = '\n'.join([seq[i:i+80] for i in range(0, len(seq), 80)])
                file.write(f'{header}\n{seqstring}\n')
            subprocess.run(f'../../../bin/Predict_Property/AUCpreD.sh -i {prefix}/{ppid}.fasta -o {prefix}',
                           check=True, shell=True)
            os.remove(f'{prefix}/{ppid}.fasta')


num_processes = int(os.environ.get('SLURM_CPUS_ON_NODE', 1))
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'

if __name__ == '__main__':
    if not os.path.exists('out/'):
        os.makedirs('out/')

    with mp.Pool(processes=num_processes) as pool:
        OGids = [path.removesuffix('.afa') for path in os.listdir('../../../data/alignments/fastas/') if path.endswith('.afa')]
        pool.map(run_cmd, OGids)
