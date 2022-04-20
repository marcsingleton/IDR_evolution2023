"""Detect alignments with misaligned regions."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import skbio
from src.draw import plot_msa_data
from src.utils import read_fasta

# LG background frequencies
prior = {'A': 0.079066, 'R': 0.055941, 'N': 0.041977, 'D': 0.053052, 'C': 0.012937,
         'Q': 0.040767, 'E': 0.071586, 'G': 0.057337, 'H': 0.022355, 'I': 0.062157,
         'L': 0.099081, 'K': 0.064600, 'M': 0.022951, 'F': 0.042302, 'P': 0.044040,
         'S': 0.061197, 'T': 0.053287, 'W': 0.012066, 'Y': 0.034155, 'V': 0.069147}
a = 1E-3  # Coefficient of outlier curve
spid_regex = r'spid=([a-z]+)'
tree = skbio.read('../../ortho_tree/consensus_LG/out/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree.tips())}

records = []
for OGid in [path.removesuffix('.afa') for path in os.listdir('../realign_hmmer/out/') if path.endswith('.afa')]:
    msa = [(re.search(spid_regex, header).group(1), seq.upper()) for header, seq in read_fasta(f'../realign_hmmer/out/{OGid}.afa')]

    idx = 0
    for j in range(len(msa[0][1])):
        for i in range(len(msa)):
            sym = msa[i][1][j]
            if sym == '.' or sym.islower():
                break
        else:
            idx = j
            break  # if no break exit
    msa = [(header, seq[idx:]) for header, seq in msa]

    idx = len(msa[0][1])
    for j in range(len(msa[0][1]), 0, -1):
        for i in range(len(msa)):
            sym = msa[i][1][j - 1]
            if sym == '.' or sym.islower():
                break
        else:
            idx = j
            break  # if no break exit
    msa = [(header, seq[:idx]) for header, seq in msa]

    # Count gaps
    gaps = []
    for j in range(len(msa[0][1])):
        gap = sum([1 if msa[i][1][j] in ['-', '.'] else 0 for i in range(len(msa))])
        gaps.append(gap)

    # Threshold, merge, and size filter to get regions
    binary = ndimage.binary_closing(np.array(gaps) < 1, structure=[1, 1, 1])
    regions = [region for region, in ndimage.find_objects(ndimage.label(binary)[0]) if (region.stop-region.start) >= 30]

    # Calculate total scores for each sequence over all regions
    scores = {header: 0 for header, _ in msa}
    for region in regions:
        for i in range(region.start, region.stop):
            # Build model
            model = {aa: 2*count for aa, count in prior.items()}  # Start with weighted prior "counts"
            for _, seq in msa:
                sym = '-' if seq[i] == '.' else seq[i]  # Convert . to - for counting gaps
                model[sym] = model.get(sym, 0) + 1  # Provide default of 0 for non-standard symbols and gaps
            total = sum(model.values())
            model = {aa: np.log(count/total) for aa, count in model.items()}  # Re-normalize and convert to log space

            # Apply model
            for header, seq in msa:
                sym = '-' if seq[i] == '.' else seq[i]  # Convert . to - for counting gaps
                scores[header] += model[sym]

    # Record statistics for each sequence
    values = list(scores.values())
    mean = np.mean(values)
    std = np.std(values)
    iqr = np.quantile(values, 0.75) - np.quantile(values, 0.25)

    for header, score in scores.items():
        x = score - mean
        records.append((x, std, iqr, OGid, msa, regions))

# Plot outputs
if not os.path.exists('out/'):
    os.mkdir('out/')

plt.scatter([record[0] for record in records], [record[1] for record in records], s=10, alpha=0.25, edgecolors='none')
plt.xlabel('(score - alignment mean) of sequence')
plt.ylabel('Standard deviation of scores in alignment')
plt.savefig('out/scatter_std-score.png')
plt.close()

plt.scatter([record[0] for record in records], [record[2] for record in records], s=10, alpha=0.25, edgecolors='none')
plt.xlabel('(score - alignment mean) of sequence')
plt.ylabel('IQR of scores in alignment')
plt.savefig('out/scatter_iqr-score1.png')

ymax = max([record[2] for record in records])
xmin = -(ymax/a)**0.5
xs = np.linspace(xmin, 0, 100)
ys = a*xs**2
plt.plot(xs, ys, color='C1')
plt.savefig('out/scatter_iqr-score2.png')
plt.close()

# Plot unique MSAs with largest deviations
OGids = set()
outliers = sorted([record for record in records if record[0] < -1 and record[2] < a*record[0]**2])  # Use -1 to exclude near-zero floating point rounding errors
for record in outliers:
    # Unpack variables
    OGid, msa, regions = record[3], record[4], record[5]
    if OGid in OGids:
        continue
    OGids.add(OGid)

    # Plot MSA with regions
    msa = [seq for _, seq in sorted(msa, key=lambda x: tip_order[x[0]])]  # Re-order sequences and extract seq only
    line = np.zeros(len(msa[0]))
    for region in regions:
        line[region] = 1
    plot_msa_data(msa, line, figsize=(16, 6))
    plt.savefig(f'out/{len(OGids)-1}_{OGid}.png', bbox_inches='tight', dpi=400)
    plt.close()

"""
DEPENDENCIES
../../ortho_tree/consensus_LG/consensus_LG.py
    ../../ortho_tree/consensus_LG/out/100R_NI.nwk
../realign_hmmer/realign_hmmer.py
    ../realign_hmmer/out/*.afa
"""