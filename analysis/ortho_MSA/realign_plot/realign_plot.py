"""Plot re-aligned alignments."""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import skbio
from src.draw import draw_msa
from src.utils import read_fasta

tree = skbio.read('../../ortho_tree/ctree_WAG/out/100red_ni.txt', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree.tips())}
spids = set([tip.name for tip in tree.tips() if tip.name != 'sleb'])

OG_filter = pd.read_table('../OG_filter/out/OG_filter.tsv')
df = pd.read_table('../gap_contrasts/out/total_sums.tsv').merge(OG_filter[['OGid', 'sqidnum']], on='OGid', how='left')  # total_sums.tsv has gnidnum already
df['norm1'] = df['total'] / df['gnidnum']
df['norm2'] = df['total'] / (df['gnidnum'] * df['len2'])

for label in ['norm1', 'norm2']:
    if not os.path.exists(f'out/{label}/'):
        os.makedirs(f'out/{label}/')

    head = df.sort_values(by=label, ascending=False).head(150)
    for i, row in enumerate(head.itertuples()):
        msa = read_fasta(f'../realign_hmmer2/out/{row.OGid}.mfa')
        msa = [(re.search(r'spid=([a-z]+)', header).group(1), seq) for header, seq in msa]

        msa = [seq.upper() for _, seq in sorted(msa, key=lambda x: tip_order[x[0]])]  # Re-order sequences and extract seq only
        im = draw_msa(msa)
        plt.imsave(f'out/{label}/{i}_{row.OGid}.png', im)

"""
DEPENDENCIES
../../ortho_tree/ctree_WAG/ctree_WAG.py
    ../../ortho_tree/ctree_WAG/out/100red_ni.txt
../gap_contrasts/gap_contrasts_calc.py
    ../gap_contrasts/out/total_sums.tsv
../OG_filter/OG_filter.py
    ../OG_filter/out/OG_filter.tsv
../realign_hmmer2/realign_hmmer2.py
    ../realign_hmmer2/out/*.mfa
"""