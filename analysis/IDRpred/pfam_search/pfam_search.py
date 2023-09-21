"""Search alignment sequences against Pfam models."""

import os
import re
from subprocess import run

from src.utils import read_fasta

spid_regex = r'spid=([a-z]+)'

if not os.path.exists('out/'):
    os.mkdir('out/')

OGids = sorted([path.removesuffix('.afa') for path in os.listdir('../../../data/alignments/fastas/') if path.endswith('.afa')])
for OGid in OGids:
    record = None
    for header, seq in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        spid = re.search(spid_regex, header).group(1)
        if spid == 'dmel':
            record = (header, seq)
    if record is None:
        raise RuntimeError(f'Alignment {OGid} does not have a sequence from dmel.')

    with open(f'out/{OGid}_dmel_temp.fa', 'w') as file:
        header, seq = record
        seq = seq.replace('-', '')  # Remove gaps
        seqstring = '\n'.join([seq[i:i+80] for i in range(0, len(seq), 80)])
        file.write(f'{header}\n{seqstring}')

    cmd = ['../../../bin/hmmscan',
           '--domE', '1E-10',  # Domain reporting threshold
           '-o', '/dev/null',  # Discard output from STDIN
           '--domtblout', f'out/{OGid}.txt',
           '../../../data/Pfam/Pfam-A_36_0.hmm',
           f'out/{OGid}_dmel_temp.fa']
    run(cmd, check=True)

    os.remove(f'out/{OGid}_dmel_temp.fa')

"""
NOTES
When planning this analysis, I originally considered searching every sequence in the alignment against the Pfam database
and attempting to merge the results somehow. I quickly realized this would greatly increase the complexity of the post-
processing of the output with little increase or even a decrease in the significant and interpretability of the results.
The main issue is merging the domain hits, which needs to be done a per-domain basis to ensure the same domains are
combined rather than similar or even different domains that hit to an overlapping interval. Additionally, my initial
plan for merging the hits was to create a vector with a confidence of that hit at each position in the alignment.
However, this muddies the interpretations of overlaps with IDRs because instead of a fraction overlap, the analogous
metric would be an average confidence over all positions, which is actually less helpful. One way of fixing this is to
threshold the confidences to make them Boolean variables, but that basically brings us back to the single sequence
comparison. I think because the Pfam models should already be highly sensitive, there's not much gained by trying to
merge the results from searches against each sequence in the alignment.
"""