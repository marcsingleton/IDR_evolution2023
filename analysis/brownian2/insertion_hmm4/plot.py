"""Plot example alignments segmented via posterior decoding."""

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import skbio
import src.hmm as hmm
import src.draw as draw
import utils
from src.brownian2.trim import trim_terminals
from src.utils import read_fasta

# Load model parameters
with open('out/model.json') as file:
    params = json.load(file)


# Load OGids
OGids = set()
with open('../config/segments.tsv') as file:
    file.readline()  # Skip header
    for line in file:
        OGid = line.split()[0]
        OGids.add(OGid)

# Load tree
tree_template = skbio.read('../../ortho_tree/consensus_LG/out/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_template.tips())}

# Plot alignments
for OGid in OGids:
    # Load msa and trim terminal insertions
    msa = trim_terminals(read_fasta(f'../../ortho_MSA/realign_hmmer1/out/{OGid}.mfa'))
    msa = [(re.search(r'spid=([a-z]+)', header).group(1), seq) for header, seq in msa]

    # Create emission sequence
    col0 = []
    emits = []
    for j in range(len(msa[0][1])):
        col = [1 if msa[i][1][j] in ['-', '.'] else 0 for i in range(len(msa))]
        emit0 = all([c0 == c for c0, c in zip(col0, col)])
        emit1 = sum(col)
        emits.append((emit0, j))  # The tree probabilities are pre-calculated, so emission value is its index
        col0 = col

    # Load tree and convert to vectors at tips
    tree = tree_template.deepcopy().shear([spid for spid, _ in msa])
    tips = {tip.name: tip for tip in tree.tips()}
    for spid, seq in msa:
        tip = tips[spid]
        conditional = np.zeros((2, len(seq)))
        for j, sym in enumerate(seq):
            if sym in ['-', '.']:
                conditional[0, j] = 1
            else:
                conditional[1, j] = 1
        tip.conditional = conditional

    # Instantiate model
    e_dists_rv = {}
    for state, (p, pi, q0, q1) in params['e_dists'].items():
        array = utils.get_tree_probability(tree, pi, q0, q1)
        e_dists_rv[state] = utils.BinomialArrayRV(p, array)
    model = hmm.HMM(params['t_dists'], e_dists_rv, params['start_dist'])

    # Decode states and plot
    fbs = model.forward_backward(emits)
    msa = [seq.upper() for _, seq in sorted(msa, key=lambda x: tip_order[x[0]])]  # Re-order sequences and extract seq only

    draw.plot_msa_lines(msa, [fbs['1A'], fbs['2'], fbs['3'], fbs['1B']], figsize=(15, 6))
    plt.savefig(f'out/{OGid}_wide.png', bbox_inches='tight')
    plt.close()

    draw.plot_msa_lines(msa, [fbs['1A'], fbs['2'], fbs['3'], fbs['1B']], figsize=(8, 8))
    plt.savefig(f'out/{OGid}_tall.png', bbox_inches='tight')
    plt.close()

"""
DEPENDENCIES
../../ortho_MSA/realign_hmmer1/realign_hmmer1.py
    ../../ortho_MSA/realign_hmmer1/out/*.mfa
../../ortho_tree/consensus_LG/consensus_LG.py
    ../../ortho_tree/consensus_LG/out/100R_NI.nwk
../config/segments.tsv
./fit.py
    ./out/model.json
"""