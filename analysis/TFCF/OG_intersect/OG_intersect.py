"""Intersect TF and CF lists with curated OGs."""

import os
import re

import pandas as pd
from src.utils import read_fasta

ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
gnid_regex = r'gnid=([A-Za-z0-9_.]+)'

# Load OGs
rows = []
OGids = sorted([path.removesuffix('.afa') for path in os.listdir('../../../data/alignments/fastas/') if path.endswith('.afa')])
for OGid in OGids:
    msa = read_fasta(f'../../../data/alignments/fastas/{OGid}.afa')
    for header, seq in msa:
        ppid = re.search(ppid_regex, header).group(1)
        gnid = re.search(gnid_regex, header).group(1)
        rows.append({'OGid': OGid, 'ppid': ppid, 'gnid': gnid})
OGs = pd.DataFrame(rows)

# Load TFs and CFs and merge with OGs
TFs = pd.read_table('../update_ids/out/TFs.txt')
CFs = pd.read_table('../update_ids/out/CFs.txt')

OGs_TF = OGs.merge(TFs, how='inner')
OGs_CF = OGs.merge(CFs, how='inner')

# Write stats and IDs to file
if not os.path.exists('out/'):
    os.mkdir('out/')

output = f"""\
Number of TF OGs: {len(OGs_TF)}
Number of unique TF genes: {OGs_TF['gnid'].nunique()}
Number of CF OGs: {len(OGs_CF)}
Number of unique CF genes: {OGs_CF['gnid'].nunique()}
"""
with open('out/output.txt', 'w') as file:
    file.write(output)

OGs_TF.to_csv('out/TFs.tsv', sep='\t', index=False)
OGs_CF.to_csv('out/CFs.tsv', sep='\t', index=False)

"""
NOTES
The number of unique genes for each set of OGs is the same as the total. This means the genes are uniquely mapped to
OGs (as opposed to one gene potentially being in multiple OGs which is quite possible). The filtered list of OGs already
removed OGs with multiple representatives from each species, so the OGs also uniquely map to genes.
"""