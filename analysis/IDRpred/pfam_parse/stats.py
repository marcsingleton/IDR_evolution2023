"""Plot statistics from raw output of Pfam hits."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_table('out/domains.tsv', sep='\t')
query_groups = df.groupby('OGid')

# Number of alignments with and without at least one hit
all_OGids = set([path.removesuffix('.txt') for path in os.listdir('../pfam_search/out/') if path.endswith('.txt')])
pfam_OGids = set(df['OGid'])
null_OGids = all_OGids - pfam_OGids
xs = [len(pfam_OGids), len(null_OGids)]
labels = ['Alignments w/ Pfam hits', 'Aligments w/o Pfam hits']

fig, ax = plt.subplots()
ax.pie(xs, labels=[f'{label}\n({x/len(all_OGids):.1%})' for x, label in zip(xs, labels)],
       labeldistance=1.5, textprops={'ha': 'center'})
fig.savefig('out/pie_hits.png')
plt.close()

# Distribution of number of domain hits
counts = query_groups['target_accession'].count().value_counts()

fig, ax = plt.subplots()
ax.bar(counts.index, counts.values, width=1)
ax.set_xlabel('Number of Pfam hits to alignment')
ax.set_ylabel('Number of alignments')
fig.savefig('out/hist_numalignments-numhits.png')
plt.close()

# Distribution of number of domain hits
counts = query_groups['target_accession'].nunique().value_counts()

fig, ax = plt.subplots()
ax.bar(counts.index, counts.values, width=1)
ax.set_xlabel('Number of unique Pfam hits to alignment')
ax.set_ylabel('Number of alignments')
fig.savefig('out/hist_numalignments-numhits_unique.png')
plt.close()

# Distribution of maximum Evalues
xs = np.log(query_groups['seq_evalue'].max())
xs = np.nan_to_num(xs, copy=False, neginf=np.nan)

fig, ax = plt.subplots()
ax.hist(xs, bins=50)
ax.set_xlabel('Maximum log(Evalue) across all Pfam hits to alignment')
ax.set_ylabel('Number of alignments')
fig.savefig('out/hist_numalignments-evalue.png')
plt.close()

# Distribution of maximum c-Evalues
xs = np.log(query_groups['c_evalue'].max())
xs = np.nan_to_num(xs, copy=False, neginf=np.nan)

fig, ax = plt.subplots()
ax.hist(xs, bins=50)
ax.set_xlabel('Maximum log(c-Evalue) across all Pfam hits to alignment')
ax.set_ylabel('Number of alignments')
fig.savefig('out/hist_numalignments-c_evalue.png')
plt.close()

# Distribution of maximum i-Evalues
xs = np.log(query_groups['i_evalue'].max())
xs = np.nan_to_num(xs, copy=False, neginf=np.nan)

fig, ax = plt.subplots()
ax.hist(xs, bins=50)
ax.set_xlabel('Maximum log(i-Evalue) across all Pfam hits to alignment')
ax.set_ylabel('Number of alignments')
fig.savefig('out/hist_numalignments-i_evalue.png')
plt.close()
