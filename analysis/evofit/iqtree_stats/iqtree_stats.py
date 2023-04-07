"""Plot results from fitting evolutionary parameters."""

import os
import re
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import skbio
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from src.utils import read_iqtree, read_paml

paml_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
labels = ['0R_disorder', '50R_disorder', '100R_disorder', '0R_order', '50R_order', '100R_order']
labels_suffix = r'_[0-9]+'
Record = namedtuple('Record', ['label', 'ematrix', 'rmatrix', 'freqs', 'length'])

# Load LG model
ematrix, freqs = read_paml('../../../data/matrices/LG.paml', norm=True)
rmatrix = freqs * ematrix
LG_record = Record('LG', np.stack([ematrix]), np.stack([rmatrix]), np.stack([freqs]), np.stack([np.nan]))

# Load IQ-TREE matrices
records = {}
for label in labels:
    file_labels = []
    for path in os.listdir('../iqtree_fit/out/'):
        match = re.match(f'({label}{labels_suffix})\.iqtree', path)
        if match:
            file_labels.append(match.group(1))

    ematrix_stack = []
    rmatrix_stack = []
    freqs_stack = []
    length_stack = []
    for file_label in sorted(file_labels):
        record = read_iqtree(f'../iqtree_fit/out/{file_label}.iqtree', norm=True)
        ematrix, freqs = record['ematrix'], record['freqs']
        rmatrix = freqs * ematrix

        tree = skbio.read(f'../iqtree_fit/out/{file_label}.treefile', 'newick', skbio.TreeNode)
        length = tree.descending_branch_length()

        ematrix_stack.append(ematrix)
        rmatrix_stack.append(rmatrix)
        freqs_stack.append(freqs)
        length_stack.append(length)
    records[label] = Record(label,
                            np.stack(ematrix_stack),
                            np.stack(rmatrix_stack),
                            np.stack(freqs_stack),
                            np.stack(length_stack))

# Make plots
if not os.path.exists('out/'):
    os.mkdir('out/')

# 0 RE-ORDER SYMBOLS BY DISORDER RATIO
freqs1 = records['50R_disorder'].freqs.mean(axis=0)
freqs2 = records['50R_order'].freqs.mean(axis=0)
sym2ratio = {sym: ratio for sym, ratio in zip(paml_order, freqs1 / freqs2)}
sym2idx = {sym: idx for idx, sym in enumerate(paml_order)}
alphabet = sorted(paml_order, key=lambda x: sym2ratio[x])
ix = [sym2idx[sym] for sym in alphabet]
ixgrid = np.ix_(ix, ix)
LG_record = Record(LG_record.label,
                   LG_record.ematrix[:, ixgrid[0], ixgrid[1]],
                   LG_record.rmatrix[:, ixgrid[0], ixgrid[1]],
                   LG_record.freqs[:, ix],
                   LG_record.length)
for label, record in records.items():
    records[label] = Record(label,
                            record.ematrix[:, ixgrid[0], ixgrid[1]],
                            record.rmatrix[:, ixgrid[0], ixgrid[1]],
                            record.freqs[:, ix],
                            record.length)

# 1 BUBBLE PLOT
plots = [('ematrix', 'exchangeability', 0.125),
         ('rmatrix', 'rate', 0.55)]
for data_label, title_label, scale in plots:
    fig, axs = plt.subplots(2, 3, figsize=(8, 6), layout='constrained')
    for ax, record in zip(axs.ravel(), records.values()):
        ax.set_xlim(0, len(alphabet)+1)
        ax.set_xticks(range(1, len(alphabet)+1), alphabet, fontsize=7)
        ax.set_ylim(0, len(alphabet)+1)
        ax.set_yticks(range(1, len(alphabet)+1), alphabet[::-1], fontsize=7, ha='center')
        ax.set_title(record.label)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_aspect(1)

        data = getattr(record, data_label)
        for i, row in enumerate(data.mean(axis=0)):
            for j, value in enumerate(row):
                c = Circle((j+1, len(alphabet)-i), scale*value**0.5, color='black')
                ax.add_patch(c)
    fig.suptitle(f'{title_label.capitalize()} matrices')
    fig.savefig(f'out/bubble_{data_label}.png')
    plt.close()

# 2 HEATMAP
plots = [('ematrix', 'exchangeability'),
         ('rmatrix', 'rate')]
for data_label, title_label in plots:
    vmax = max([getattr(record, data_label).mean(axis=0).max() for record in records.values()])
    fig, axs = plt.subplots(2, 3, figsize=(8, 6), layout='constrained')
    for ax, record in zip(axs.ravel(), records.values()):
        data = getattr(record, data_label)
        ax.imshow(data.mean(axis=0), vmin=0, vmax=vmax, cmap='Greys')
        ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
        ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7, ha='center')
        ax.set_title(record.label)
    fig.suptitle(f'{title_label.capitalize()} matrices')
    fig.colorbar(ScalarMappable(Normalize(0, vmax), cmap='Greys'), ax=axs, fraction=0.025)
    fig.savefig(f'out/heatmap_{data_label}.png')
    plt.close()

# 3 CORRELATION GRID
plots = [('ematrix', 'exchangeability'),
         ('rmatrix', 'rate')]
for data_label, title_label in plots:
    corr = np.zeros((len(labels), len(labels)))
    for i, label1 in enumerate(labels):
        matrix1 = getattr(records[label1], data_label).mean(axis=0)
        for j, label2 in enumerate(labels[:i+1]):
            matrix2 = getattr(records[label2], data_label).mean(axis=0)
            r = np.corrcoef(matrix1.ravel(), matrix2.ravel())
            corr[i, j] = r[0, 1]
            corr[j, i] = r[0, 1]

    fig, ax = plt.subplots()
    im = ax.imshow(corr)
    ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=30, rotation_mode='anchor',
                  horizontalalignment='right', verticalalignment='center')
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(f'Meta-alignment correlations: {title_label} matrix')
    fig.colorbar(im)
    fig.savefig(f'out/heatmap_corr_{data_label}.png')
    plt.close()

# 4 VARIATION
plots = [(records['50R_disorder'], 'ematrix', 'exchangeability'),
         (records['50R_order'], 'ematrix', 'exchangeability'),
         (records['50R_disorder'], 'rmatrix', 'rate'),
         (records['50R_order'], 'rmatrix', 'rate')]
for record, data_label, title_label in plots:
    fig = plt.figure()
    gs = plt.GridSpec(2, 2, left=0.075, right=0.9, top=0.9, bottom=0.1, height_ratios=[2, 1], wspace=0.45)

    data = getattr(record, data_label)
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mean, cmap='Greys')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7, ha='center')
    ax.set_title('Mean')
    fig.colorbar(im, cax=ax.inset_axes((1.05, 0, 0.05, 1)))

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(std / mean, cmap='Greys')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7, ha='center')
    ax.set_title('Coefficient of variation')
    fig.colorbar(im, cax=ax.inset_axes((1.05, 0, 0.05, 1)))

    ax = fig.add_subplot(gs[1, :])
    ax.scatter(mean, std / mean, s=10, alpha=0.5, edgecolor='none')
    ax.set_xlabel('Mean')
    ax.set_ylabel('Coefficient of variation')

    fig.suptitle(f'{record.label}: {title_label} matrix')
    fig.savefig(f'out/CV_{record.label}_{data_label}.png')
    plt.close()

pairs = [(records['50R_disorder'], LG_record),
         (records['50R_order'], LG_record),
         (records['50R_disorder'], records['50R_order'])]
plots = [(pairs, 'ematrix', 'exchangeability'),
         (pairs, 'rmatrix', 'rate')]

# 5 RATIO HEATMAPS
for pairs, data_label, title_label in plots:
    for record1, record2 in pairs:
        matrix1 = getattr(record1, data_label).mean(axis=0)
        matrix2 = getattr(record2, data_label).mean(axis=0)
        ratio = np.log10(matrix1 / matrix2)
        vext = np.nanmax(np.abs(ratio))

        fig, ax = plt.subplots()
        im = ax.imshow(ratio, vmin=-vext, vmax=vext, cmap='RdBu')
        ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
        ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7, ha='center')
        ax.set_title(f'log10 ratio of {record1.label} to {record2.label}:\n{title_label} matrix')
        fig.colorbar(im)
        fig.savefig(f'out/heatmap_ratio_{record1.label}-{record2.label}_{data_label}.png')
        plt.close()

for pairs, data_label, title_label in plots:
    vext = -np.inf
    for record1, record2 in pairs:
        matrix1 = getattr(record1, data_label).mean(axis=0)
        matrix2 = getattr(record2, data_label).mean(axis=0)
        ratio = np.log10(matrix1 / matrix2)
        v = np.nanmax(np.abs(ratio))
        if v > vext:
            vext = v

    fig, axs = plt.subplots(1, len(pairs), figsize=(9.6, 3.2), layout='constrained')
    for ax, pair in zip(axs.ravel(), pairs):
        record1, record2 = pair
        matrix1 = getattr(record1, data_label).mean(axis=0)
        matrix2 = getattr(record2, data_label).mean(axis=0)

        ratio = np.log10(matrix1 / matrix2)
        im = ax.imshow(ratio, vmin=-vext, vmax=vext, cmap='RdBu')
        ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
        ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7, ha='center')
        ax.set_title(f'{record1.label} to {record2.label}')
    fig.suptitle(f'log10 ratios of {title_label} matrix pairs')
    fig.colorbar(ScalarMappable(Normalize(-vext, vext), cmap='RdBu'), ax=axs[-1])
    fig.savefig(f'out/heatmap_ratio_{data_label}.png')
    plt.close()

# 6 DIFFERENCE HEATMAPS
for pairs, data_label, title_label in plots:
    for record1, record2 in pairs:
        matrix1 = getattr(record1, data_label).mean(axis=0)
        matrix2 = getattr(record2, data_label).mean(axis=0)
        diff = matrix1 - matrix2
        vext = np.nanmax(np.abs(diff))

        fig, ax = plt.subplots()
        im = ax.imshow(diff, vmin=-vext, vmax=vext, cmap='RdBu')
        ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
        ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7, ha='center')
        ax.set_title(f'Difference of {record1.label} to {record2.label}:\n{title_label} matrix')
        fig.colorbar(im)
        fig.savefig(f'out/heatmap_diff_{record1.label}-{record2.label}_{data_label}.png')
        plt.close()

for pairs, data_label, title_label in plots:
    vext = -np.inf
    for record1, record2 in pairs:
        matrix1 = getattr(record1, data_label).mean(axis=0)
        matrix2 = getattr(record2, data_label).mean(axis=0)
        diff = matrix1 - matrix2
        v = np.nanmax(np.abs(diff))
        if v > vext:
            vext = v

    fig, axs = plt.subplots(1, len(pairs), figsize=(9.6, 3.2), layout='constrained')
    for ax, pair in zip(axs.ravel(), pairs):
        record1, record2 = pair
        matrix1 = getattr(record1, data_label).mean(axis=0)
        matrix2 = getattr(record2, data_label).mean(axis=0)

        diff = matrix1 - matrix2
        im = ax.imshow(diff, vmin=-vext, vmax=vext, cmap='RdBu')
        ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
        ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7, ha='center')
        ax.set_title(f'{record1.label} to {record2.label}')
    fig.suptitle(f'Differences of {title_label} matrix pairs')
    fig.colorbar(ScalarMappable(Normalize(-vext, vext), cmap='RdBu'), ax=axs[-1])
    fig.savefig(f'out/heatmap_diff_{data_label}.png')
    plt.close()

# 7 FREQUENCIES
width = 0.2
bars = [records['50R_disorder'], records['50R_order'], LG_record]
fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
for i, record in enumerate(bars):
    freqs = record.freqs.mean(axis=0)
    std = record.freqs.std(axis=0)
    dx = -(len(bars) - 1) / 2 + i
    ax.bar([x+width*dx for x in range(len(alphabet))], freqs, yerr=std, label=record.label, width=width)
ax.set_xticks(range(len(alphabet)), alphabet)
ax.set_xlabel('Amino acid')
ax.set_ylabel('Frequency')
ax.legend()
fig.savefig('out/bar_freqs.png')
plt.close()

freqs1, freqs2 = records['50R_disorder'].freqs.mean(axis=0), records['50R_order'].freqs.mean(axis=0)
std1, std2 = records['50R_disorder'].freqs.std(axis=0), records['50R_order'].freqs.std(axis=0)
ys = freqs1 / freqs2
std = ys * ((std1 / freqs1) ** 2 + (std2 / freqs2) ** 2) ** 0.5  # Propagation of error formula for ratios
fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
ax.bar(range(len(alphabet)), ys, yerr=std, width=0.5)
ax.set_xticks(range(len(alphabet)), alphabet)
ax.set_xlabel('Amino acid')
ax.set_ylabel('Frequency ratio')
fig.savefig('out/bar_ratios.png')
plt.close()

# 8 TREE LENGTHS
ys = []
yerr = []
for label in labels:
    length = records[label].length
    ys.append(length.mean())
    yerr.append(length.std())
fig, ax = plt.subplots()
ax.bar(range(len(ys)), ys, yerr=yerr, width=0.5)
ax.set_xticks(range(len(ys)), labels, fontsize=8)
ax.set_xlabel('Meta-alignment')
ax.set_ylabel('Total tree length')
fig.savefig('out/bar_length.png')
plt.close()
