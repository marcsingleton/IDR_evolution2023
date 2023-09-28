"""Plot statistics of overlaps of regions with Pfam domains."""

import os

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_table('out/overlaps.tsv')
df['fraction'] = df['overlap'] / df['length']
disorder = df[df['disorder']]
order = df[~df['disorder']]

if not os.path.exists('out/'):
    os.mkdir('out/')

# Pie charts
hs_disorder = [(disorder['fraction'] == 0).sum(), (disorder['fraction'] != 0).sum()]
hs_order = [(order['fraction'] == 0).sum(), (order['fraction'] != 0).sum()]
hs_stack = list(zip(hs_disorder, hs_order))
hs_labels = ['No overlap', 'Overlap']
hs_colors = ['white', 'darkgray']
hs_hatches = [None, None]

xs = list(range(len(hs_stack)))
xs_labels = ['disorder', 'order']
xs_lim = [-1, 2]

fig, ax = plt.subplots()
bs = [0 for _ in range(len(hs_stack))]
for hs, label, color, hatch in zip(hs_stack, hs_labels, hs_colors, hs_hatches):
    ax.bar(xs, hs, bottom=bs, width=0.5, label=label, color=color, hatch=hatch, linewidth=1.25, edgecolor='black')
    bs = [h + b for h, b in zip(hs, bs)]
ax.set_xlim(xs_lim)
ax.set_xticks(xs, xs_labels)
ax.set_ylabel('Number of regions')
ax.legend()
fig.savefig('out/bar_overlap.png')
plt.close()

# Histograms of non-zero overlap
fig, axs = plt.subplots(2)
axs[0].hist(disorder.loc[disorder['fraction'] != 0, 'fraction'], bins=50, label='disorder', color='C0')
axs[1].hist(order.loc[order['fraction'] != 0, 'fraction'], bins=50, label='order', color='C1')
axs[1].set_xlabel('Fraction overlap with Pfam domain')
for ax in axs:
    ax.set_ylabel('Number of regions')
    ax.legend()
fig.savefig('out/hist_regionnum-fraction.png')
plt.close()
