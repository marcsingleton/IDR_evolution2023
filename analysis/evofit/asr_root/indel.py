"""Calculate ancestral sequence reconstruction at root for indel process."""

import json
import os
import re

import numpy as np
import skbio
from scipy.special import gammainc
from scipy.stats import gamma
from src.evosim.asr import get_conditional
from src.utils import read_fasta

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)

if not os.path.exists('out/'):
    os.mkdir('out/')

OGids = [path.removesuffix('.iqtree') for path in os.listdir('../asr_indel/out/') if path.endswith('.iqtree')]
for OGid in OGids:
    # Load tree
    indel_tree = skbio.read(f'../asr_indel/out/{OGid}.treefile', 'newick', skbio.TreeNode)
    tree = tree_template.shear([tip.name for tip in indel_tree.tips()])
    indel_length = indel_tree.descending_branch_length()
    length = tree.descending_branch_length()
    speed = indel_length / length

    with open(f'../asr_indel/out/{OGid}.iqtree') as file:
        line = file.readline()

        # Load data statistics
        while not line.startswith('Input data:'):
            line = file.readline()
        match = re.search('([0-9]+) sequences with ([0-9]+) binary sites', line)
        num_seqs = int(match.group(1))
        num_columns = int(match.group(2))

        # Load substitution model
        while line != 'State frequencies: (estimated with maximum likelihood)\n':
            line = file.readline()

        # Load equilibrium frequencies
        freqs = np.zeros(2)
        for _ in range(2):
            line = file.readline()
        for i in range(2):
            freq = float(line.rstrip('\n').split(' = ')[1])
            freqs[i] = freq
            line = file.readline()

        # Load rate matrix
        matrix = np.zeros((2, 2))
        for _ in range(3):
            line = file.readline()
        for i in range(2):
            rates = line.split()
            matrix[i] = [float(rate) for rate in rates[1:]]
            line = file.readline()

        # Load rate categories
        # In IQ-TREE, only the shape parameter is fit and the rate parameter beta is set to alpha so the mean of gamma distribution is 1
        # The calculations here directly correspond to equation 10 in Yang. J Mol Evol (1994) 39:306-314.
        # Note the equation has a small typo where the difference in gamma function evaluations should be divided by the probability
        # of that category since technically it is the rate given that category
        while not line.startswith('Model of rate heterogeneity:'):
            line = file.readline()
        if 'Uniform' in line:
            num_categories = 1
            alpha = 'NA'
            rates = [(1, 1)]
        elif 'Gamma' in line:
            num_categories = int(line.rstrip('\n').split(' Gamma with ')[1].removesuffix(' categories'))
            alpha = float(file.readline().rstrip('\n').split(': ')[1])
            igfs = []  # Incomplete gamma function evaluations
            for i in range(num_categories+1):
                x = gamma.ppf(i/num_categories, a=alpha, scale=1/alpha)
                igfs.append(gammainc(alpha+1, alpha*x))
            rates = []
            for i in range(num_categories):
                rate = num_categories * (igfs[i+1] - igfs[i])
                rates.append((rate, 1/num_categories))
        else:
            raise RuntimeError('Unknown rate model detected.')

    # Load sequence and convert to vectors at tips of tree
    msa = read_fasta(f'../asr_indel/out/{OGid}.afa')
    tips = {tip.name: tip for tip in tree.tips()}
    for header, seq in msa:
        spid = header.split()[0][1:]  # Split on white space, first field, trim >
        tip = tips[spid]
        value = np.zeros((2, len(seq)))
        for j, sym in enumerate(seq):
            value[int(sym), j] = 1
        tip.value = value

    # Get likelihoods for rate categories
    likelihoods = []
    for rate, prior in rates:
        s, conditional = get_conditional(tree, speed * rate * matrix)
        likelihood = np.expand_dims(freqs, -1) * conditional
        likelihoods.append(np.exp(s) * likelihood * prior)

    likelihoods = np.stack(likelihoods)
    likelihoods = likelihoods / likelihoods.sum(axis=(0, 1))
    np.save(f'out/{OGid}_indel.npy', likelihoods)

    # Save model information as JSON
    partition = {'num_seqs': num_seqs, 'num_columns': num_columns,
                 'num_categories': num_categories, 'alpha': alpha, 'speed': speed, 'rates': rates}
    with open(f'out/{OGid}_indel_model.json', 'w') as file:
        json.dump(partition, file)

"""
NOTES
See notes in aa.py for reasoning for re-calculating rates from alpha.

The script will likely raise some RuntimeWarnings caused by overflow during matrix exponentiation. In these cases, the
matrix has large rates which cause overflow during matrix exponentiation. Fortunately, they can safely be ignored
because the function still returns the correct limiting distribution.
"""