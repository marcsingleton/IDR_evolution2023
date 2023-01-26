"""Plot results from fitting evolutionary parameters."""

import os
import re
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
labels = ['0R_disorder', '50R_disorder', '100R_disorder',
          '0R_order', '50R_order', '100R_order']
labels_suffix = r'_[0-9]+\.iqtree'
Record = namedtuple('Record', ['label', 'matrix', 'freqs'])

# Load LG model
with open('../config/LG.paml') as file:
    # Load exchangeability matrix
    LG_matrix = np.zeros((len(alphabet), len(alphabet)))
    for i in range(len(alphabet)-1):
        line = file.readline()
        for j, value in enumerate(line.split()):
            LG_matrix[i + 1, j] = float(value)
            LG_matrix[j, i + 1] = float(value)

    # Load equilibrium frequencies
    for _ in range(2):
        line = file.readline()
    LG_freqs = np.array([float(value) for value in line.split()])
    LG_freqs = LG_freqs / LG_freqs.sum()  # Re-normalize to ensure sums to 1
rate = (LG_freqs * (LG_freqs * LG_matrix).sum(axis=1)).sum()
LG_matrix = LG_matrix / rate  # Normalize average rate to 1

# Load IQ-TREE matrices
records = []
for label in labels:
    file_labels = []
    for path in os.listdir('../iqtree_fit/out/'):
        match = re.match(f'{label}{labels_suffix}', path)
        if match:
            file_labels.append(match.group(0))

    matrix_stack = []
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

            matrix_stack.append(matrix)
            freqs_stack.append(freqs)
    records.append(Record(label, np.stack(matrix_stack), np.stack(freqs_stack)))

# Make plots
if not os.path.exists('out/'):
    os.mkdir('out/')

# 1 HEATMAP
vmax = max([record.matrix.mean(axis=0).max() for record in records])
fig, axs = plt.subplots(2, 3, figsize=(8, 6), layout='constrained')
for ax, record in zip(axs.ravel(), records):
    ax.imshow(record.matrix.mean(axis=0), vmax=vmax, cmap='Greys')
    ax.set_title(record.label)
    ax.set_xticks(range(len(alphabet)), alphabet, fontsize=7)
    ax.set_yticks(range(len(alphabet)), alphabet, fontsize=7)
fig.colorbar(ScalarMappable(Normalize(0, vmax), cmap='Greys'), ax=axs, fraction=0.025)
fig.savefig('out/heatmap_all.png')
plt.close()

# 2 BUBBLE PLOT
scale = 0.15
fig, axs = plt.subplots(2, 3, figsize=(8, 6), layout='constrained')
for ax, record in zip(axs.ravel(), records):
    ax.set_title(record.label)
    ax.set_xlim(0, len(alphabet))
    ax.set_xticks(range(1, len(alphabet)+1), alphabet, fontsize=7)
    ax.set_ylim(0, len(alphabet))
    ax.set_yticks(range(1, len(alphabet)+1), alphabet[::-1], fontsize=7)
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

# 3 BUBBLE COMPARISONS
scale = 0.12
plots = [(records[1].label, records[1].matrix.mean(axis=0), 'black', 'LG', LG_matrix, 'white'),
         (records[4].label, records[4].matrix.mean(axis=0), 'grey', 'LG', LG_matrix, 'white'),
         (records[1].label, records[1].matrix.mean(axis=0), 'black', records[4].label, records[4].matrix.mean(axis=0), 'grey')]
for l1, m1, c1, l2, m2, c2 in plots:
    fig, ax = plt.subplots(figsize=(9, 4), gridspec_kw={'left': 0.025, 'right': 0.85, 'top': 0.99, 'bottom': 0.05})
    ax.set_xlim(0, len(alphabet))
    ax.set_xticks(range(1, len(alphabet)+1), alphabet, fontsize=7)
    ax.set_ylim(0, len(alphabet))
    ax.set_yticks(range(1, len(alphabet)+1), alphabet[::-1], fontsize=7)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_aspect(0.5)  # Scale vertical axis half of horizontal axis

    for i, row in enumerate(m1):
        for j, value in enumerate(row):
            if j >= i:
                continue
            c = Ellipse((j+1, len(alphabet)-i), height=2*scale*value**0.5, width=scale*value**0.5, facecolor=c1, edgecolor='black')
            ax.add_patch(c)
    for i, row in enumerate(m2):
        for j, value in enumerate(row):
            if j >= i:
                continue
            c = Ellipse((j+1.5, len(alphabet)-i), height=2*scale*value**0.5, width=scale*value**0.5, facecolor=c2, edgecolor='black')
            ax.add_patch(c)
    handles = [Line2D([], [], label=l1, marker='o', markerfacecolor=c1, markeredgecolor='black', markersize=8, linestyle='None'),
               Line2D([], [], label=l2, marker='o', markerfacecolor=c2, markeredgecolor='black', markersize=8, linestyle='None')]
    ax.legend(handles=handles, bbox_to_anchor=(1, 0.5), loc='center left', handletextpad=0)

    fig.savefig(f'out/bubble_adj_{l1}-{l2}.png')
    plt.close()

# 4 RATIO BUBBLE COMPARISONS
scale = 0.35
plots = [(records[1].label, records[1].matrix.mean(axis=0), 'LG', LG_matrix),
         (records[4].label, records[4].matrix.mean(axis=0), 'LG', LG_matrix),
         (records[1].label, records[1].matrix.mean(axis=0), records[4].label, records[4].matrix.mean(axis=0))]
for l1, m1, l2, m2 in plots:
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title(f'log10 ratio of {l1} to {l2}')
    ax.set_xlim(0, len(alphabet))
    ax.set_xticks(range(1, len(alphabet)+1), alphabet, fontsize=7)
    ax.set_ylim(0, len(alphabet))
    ax.set_yticks(range(1, len(alphabet)+1), alphabet[::-1], fontsize=7)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_aspect(1)

    for i, row in enumerate(np.log10(m1/m2)):
        for j, value in enumerate(row):
            if j >= i:
                continue
            c = Circle((j+1, len(alphabet)-i), scale*abs(value)**0.5, facecolor='black' if value > 0 else 'white', edgecolor='black')
            ax.add_patch(c)

    # Make legend
    y1, y2 = 10, 15
    vs = [0.5, 1, 2]
    dy = (y2 - y1 - sum([2*scale*v for v in vs])) / len(vs)
    ys, y = [], y1
    for v in vs:
        ys.append(y + scale*v**0.5)
        y += 2*scale*v**0.5 + dy
    for v, y in zip(vs, ys):
        ax.add_patch(Circle((22, y), scale*v**0.5, facecolor='grey', edgecolor='black', clip_on=False))
        ax.text(23, y, f'|log(R)| = {v}', size=8, va='center', clip_on=False)

    ax.add_patch(Circle((22, 8), 1*scale, facecolor='black', edgecolor='black', clip_on=False))
    ax.text(23, 8, 'R > 1', size=8, va='center', clip_on=False)
    ax.add_patch(Circle((22, 6.5), 1*scale, facecolor='white', edgecolor='black', clip_on=False))
    ax.text(23, 6.5, 'R < 1', size=8, va='center', clip_on=False)

    fig.savefig(f'out/bubble_ratio_{l1}-{l2}.png')
    plt.close()

# 5 DIFFERENCE BUBBLE COMPARISONS
scale = 0.25
plots = [(records[1].label, records[1].matrix.mean(axis=0), 'LG', LG_matrix),
         (records[4].label, records[4].matrix.mean(axis=0), 'LG', LG_matrix),
         (records[1].label, records[1].matrix.mean(axis=0), records[4].label, records[4].matrix.mean(axis=0))]
for l1, m1, l2, m2 in plots:
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title(f'Difference of {l1} and {l2}')
    ax.set_xlim(0, len(alphabet))
    ax.set_xticks(range(1, len(alphabet)+1), alphabet, fontsize=7)
    ax.set_ylim(0, len(alphabet))
    ax.set_yticks(range(1, len(alphabet)+1), alphabet[::-1], fontsize=7)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_aspect(1)

    for i, row in enumerate(m1-m2):
        for j, value in enumerate(row):
            if j >= i:
                continue
            c = Circle((j+1, len(alphabet)-i), scale*abs(value)**0.5, facecolor='black' if value > 0 else 'white', edgecolor='black')
            ax.add_patch(c)

    # Make legend
    y1, y2 = 10, 15
    vs = [0.25, 1, 4]
    dy = (y2 - y1 - sum([2*scale*v for v in vs])) / len(vs)
    ys, y = [], y1
    for v in vs:
        ys.append(y + scale*v**0.5)
        y += 2*scale*v**0.5 + dy
    for v, y in zip(vs, ys):
        ax.add_patch(Circle((22, y), scale*v**0.5, facecolor='grey', edgecolor='black', clip_on=False))
        ax.text(23, y, f'|D| = {v}', size=8, va='center', clip_on=False)

    ax.add_patch(Circle((22, 8), 2*scale, facecolor='black', edgecolor='black', clip_on=False))
    ax.text(23, 8, 'D > 0', size=8, va='center', clip_on=False)
    ax.add_patch(Circle((22, 6.5), 2*scale, facecolor='white', edgecolor='black', clip_on=False))
    ax.text(23, 6.5, 'D < 0', size=8, va='center', clip_on=False)

    fig.savefig(f'out/bubble_diff_{l1}-{l2}.png')
    plt.close()

# 6 FREQUENCIES
width = 0.2
bars = [(records[1].label, records[1].freqs.mean(axis=0), 'black'),
        (records[4].label, records[4].freqs.mean(axis=0), 'grey'),
        ('LG', LG_freqs, 'white')]
plt.figure(figsize=(8, 4), layout='constrained')
for i, (label, freqs, color) in enumerate(bars):
    dx = -len(bars) // 2 + i + (1.5 if len(bars) % 2 == 0 else 1)
    plt.bar([x+width*dx for x in range(len(alphabet))], freqs, tick_label=alphabet, label=label,
            facecolor=color, edgecolor='black', width=width)
plt.xlabel('Amino acid')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('out/bar_freqs.png')
plt.close()
