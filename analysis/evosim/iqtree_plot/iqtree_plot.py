"""Plot results from fitting evolutionary parameters."""

import os
import re
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from src.utils import read_paml

alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
labels = ['0R_disorder', '50R_disorder', '100R_disorder',
          '0R_order', '50R_order', '100R_order']
labels_suffix = r'_[0-9]+\.iqtree'
Record = namedtuple('Record', ['label', 'matrix', 'rates', 'freqs'])

# Load LG model
matrix, freqs = read_paml('../../../data/matrices/LG.paml', norm=True)
rates = freqs * matrix
LG_record = Record('LG', np.stack([matrix]), np.stack([rates]), np.stack([freqs]))

# Load IQ-TREE matrices
records = {}
for label in labels:
    file_labels = []
    for path in os.listdir('../iqtree_fit/out/'):
        match = re.match(f'{label}{labels_suffix}', path)
        if match:
            file_labels.append(match.group(0))

    matrix_stack = []
    rates_stack = []
    freqs_stack = []
    for file_label in file_labels:
        with open(f'../iqtree_fit/out/{file_label}') as file:
            # Move to exchangeability matrix and load
            line = file.readline()
            while not line.startswith('Substitution parameters'):
                line = file.readline()
            for _ in range(2):
                line = file.readline()

            rows = []
            while line != '\n':
                rows.append([float(value) for value in line.split()])
                line = file.readline()

            # Move to equilibrium frequencies and load
            for _ in range(3):
                line = file.readline()

            syms, freqs = [], []
            while line != '\n':
                match = re.search(r'pi\(([A-Z])\) = (0.[0-9]+)', line)
                syms.append(match.group(1))
                freqs.append(float(match.group(2)))
                line = file.readline()
            freqs = np.array(freqs)

            if syms != alphabet:
                raise RuntimeError('Symbols in matrix are not in expected order.')

            # Make matrix and scale
            matrix = np.zeros((len(syms), len(syms)))
            for i, row in enumerate(rows[:-1]):
                for j, value in enumerate(row):
                    matrix[i+1, j] = value
                    matrix[j, i+1] = value

            rate = (freqs * (freqs * matrix).sum(axis=1)).sum()
            matrix = matrix / rate
            rates = freqs * matrix

            matrix_stack.append(matrix)
            rates_stack.append(rates)
            freqs_stack.append(freqs)
    records[label] = Record(label, np.stack(matrix_stack), np.stack(rates_stack), np.stack(freqs_stack))

# Make plots
if not os.path.exists('out/'):
    os.mkdir('out/')

# 1 HEATMAP
vmax = max([record.matrix.mean(axis=0).max() for record in records.values()])
fig, axs = plt.subplots(2, 3, figsize=(8, 6), layout='constrained')
for ax, record in zip(axs.ravel(), records.values()):
    ax.imshow(record.matrix.mean(axis=0), vmax=vmax, cmap='Greys')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_title(record.label)
fig.colorbar(ScalarMappable(Normalize(0, vmax), cmap='Greys'), ax=axs, fraction=0.025)
fig.savefig('out/heatmap_all.png')
plt.close()

# 2 VARIATION
plots = [records['50R_disorder'], records['50R_order']]
for record in plots:
    fig = plt.figure()
    gs = plt.GridSpec(2, 2, left=0.1, right=0.95, top=0.95, bottom=0.1, height_ratios=[2, 1])

    mean = record.matrix.mean(axis=0)
    std = record.matrix.std(axis=0, ddof=1)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mean, cmap='Greys')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_title('Mean')
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(std / mean, cmap='Greys')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_title('Coefficient of variation')
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[1, :])
    ax.scatter(mean, std / mean, s=10, alpha=0.5, edgecolor='none')
    ax.set_xlabel('Mean')
    ax.set_ylabel('Coefficient of variation')

    fig.savefig(f'out/panel_CV_{record.label}.png')
    plt.close()

# 3 BUBBLE PLOT
scale = 0.15
fig, axs = plt.subplots(2, 3, figsize=(8, 6), layout='constrained')
for ax, record in zip(axs.ravel(), records.values()):
    ax.set_xlim(0, len(alphabet))
    ax.set_xticks(range(1, len(alphabet)+1), alphabet, fontsize=7)
    ax.set_ylim(0, len(alphabet))
    ax.set_yticks(range(1, len(alphabet)+1), alphabet[::-1], fontsize=7)
    ax.set_title(record.label)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_aspect(1)

    for i, row in enumerate(record.matrix.mean(axis=0)):
        for j, value in enumerate(row):
            if j >= i:
                continue
            c = Circle((j+1, len(alphabet)-i), scale*value**0.5, color='black')
            ax.add_patch(c)
fig.savefig('out/bubble_all.png')
plt.close()

# 4 RATIO HEATMAPS
plots = [(records['50R_disorder'], LG_record),
         (records['50R_order'], LG_record),
         (records['50R_disorder'], records['50R_order'])]
for record1, record2 in plots:
    matrix1 = record1.matrix.mean(axis=0)
    matrix2 = record2.matrix.mean(axis=0)

    rs = np.log10(matrix1 / matrix2)
    vext = np.nanmax(np.abs(rs))
    fig, ax = plt.subplots()
    im = ax.imshow(rs, vmin=-vext, vmax=vext, cmap='RdBu')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_title(f'log10 ratio of {record1.label} to {record2.label}')
    fig.colorbar(im)
    fig.savefig(f'out/heatmap_ratio_{record1.label}-{record2.label}.png')
    plt.close()

vext = -np.inf
for record1, record2 in plots:
    matrix1 = record1.matrix.mean(axis=0)
    matrix2 = record2.matrix.mean(axis=0)

    rs = np.log10(matrix1 / matrix2)
    v = np.nanmax(np.abs(rs))
    if v > vext:
        vext = v
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), layout='constrained')
for ax, plot in zip(axs.ravel(), plots):
    record1, record2 = plot
    matrix1 = record1.matrix.mean(axis=0)
    matrix2 = record2.matrix.mean(axis=0)

    rs = np.log10(matrix1 / matrix2)
    im = ax.imshow(rs, vmin=-vext, vmax=vext, cmap='RdBu')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_title(f'{record1.label} to {record2.label}')
fig.suptitle('log10 ratios of matrix pairs')
fig.colorbar(ScalarMappable(Normalize(-vext, vext), cmap='RdBu'), ax=axs[-1])
fig.savefig('out/heatmap_ratio.png')
plt.close()

# 5 DIFFERENCE HEATMAPS
scale = 0.25
plots = [(records['50R_disorder'], LG_record),
         (records['50R_order'], LG_record),
         (records['50R_disorder'], records['50R_order'])]
for record1, record2 in plots:
    matrix1 = record1.matrix.mean(axis=0)
    matrix2 = record2.matrix.mean(axis=0)

    ds = matrix1 - matrix2
    vext = np.nanmax(np.abs(ds))
    fig, ax = plt.subplots()
    im = ax.imshow(ds, vmin=-vext, vmax=vext, cmap='RdBu')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_title(f'Difference of {record1.label} to {record2.label}')
    fig.colorbar(im)
    fig.savefig(f'out/heatmap_diff_{record1.label}-{record2.label}.png')
    plt.close()

vext = -np.inf
for record1, record2 in plots:
    matrix1 = record1.matrix.mean(axis=0)
    matrix2 = record2.matrix.mean(axis=0)

    ds = matrix1 - matrix2
    v = np.nanmax(np.abs(ds))
    if v > vext:
        vext = v
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), layout='constrained')
for ax, plot in zip(axs.ravel(), plots):
    record1, record2 = plot
    matrix1 = record1.matrix.mean(axis=0)
    matrix2 = record2.matrix.mean(axis=0)

    ds = matrix1 - matrix2
    im = ax.imshow(ds, vmin=-vext, vmax=vext, cmap='RdBu')
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_title(f'{record1.label} to {record2.label}')
fig.suptitle('Differences of matrix pairs')
fig.colorbar(ScalarMappable(Normalize(-vext, vext), cmap='RdBu'), ax=axs[-1])
fig.savefig('out/heatmap_diff.png')
plt.close()

# 6 FREQUENCIES
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
plt.savefig('out/bar_freqs.png')
plt.close()
