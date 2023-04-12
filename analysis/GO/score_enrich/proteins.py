"""Calculate GO term enrichment on proteins with high rates of score evolution."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.patches import Patch


def hypergeom_test(k, M, n, N):
    """Return the p-value for a hypergeometric test."""
    kmax = min(n, N)
    ks = np.arange(k, kmax + 1)
    pmfs = stats.hypergeom.pmf(k=ks, M=M, n=n, N=N)
    pvalue = pmfs.sum()
    return pvalue


ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
gnid_regex = r'gnid=([A-Za-z0-9_.]+)'

color3 = '#b07aa1'

gaf = pd.read_table('../filter_GAF/out/proteins/GAF_propagate.tsv')  # Use all terms

contrasts = pd.read_table(f'../../IDRpred/get_contrasts/out/contrasts.tsv')
contrasts = contrasts.set_index(['OGid', 'contrast_id'])
all_proteins = contrasts.index.get_level_values('OGid').drop_duplicates().to_frame(index=False)

rates = (contrasts ** 2).groupby(['OGid']).mean()
quantile = rates['score_fraction'].quantile(0.9, interpolation='higher')  # Capture at least 90% of data with higher

reference_gaf = all_proteins.merge(gaf, how='inner', on=['OGid'])
enrichment_keys = rates[rates['score_fraction'] > quantile].index.to_frame(index=False)  # False forces re-index
enrichment_gaf = reference_gaf.merge(enrichment_keys, how='inner', on=['OGid'])
terms = list(enrichment_gaf[['aspect', 'GOid', 'name']].drop_duplicates().itertuples(index=False, name=None))

rows = []
for aspect, GOid, name in terms:
    k = enrichment_gaf[enrichment_gaf['GOid'] == GOid].groupby(['OGid']).ngroups
    M = reference_gaf.groupby(['OGid']).ngroups
    n = reference_gaf[reference_gaf['GOid'] == GOid].groupby(['OGid']).ngroups
    N = enrichment_gaf.groupby(['OGid']).ngroups
    pvalue = hypergeom_test(k, M, n, N)
    rows.append({'pvalue': pvalue, 'aspect': aspect, 'GOid': GOid, 'name': name})
result = pd.DataFrame(rows).sort_values(by=['aspect', 'pvalue'], ignore_index=True)

if not os.path.exists('out/'):
    os.mkdir('out/')

result.to_csv('out/pvalues_proteins.tsv', sep='\t', index=False)

fig, axs = plt.subplots(2, 1, gridspec_kw={'right': 0.85})
for ax in axs:
    ax.axvspan(quantile, rates['score_fraction'].max(), color='#e6e6e6')
    ax.hist(rates['score_fraction'], bins=150, color=color3)
    ax.set_ylabel('Number of proteins')
axs[1].set_xlabel('Score rate')
axs[1].set_yscale('log')
fig.legend(handles=[Patch(facecolor=color3, label='all')], bbox_to_anchor=(0.85, 0.5), loc='center left')
fig.savefig('out/hist_numproteins-score_rate.png')
plt.close()

fig, ax = plt.subplots(figsize=(8, 6.4), layout='constrained')
bars = [('P', 'Process'), ('F', 'Function'), ('C', 'Component')]
y0, labels = 0, []
for aspect, aspect_label in bars:
    data = result[(result['aspect'] == aspect) & (result['pvalue'] <= 0.01)]
    xs = -np.log10(data['pvalue'])
    ys = np.arange(y0, y0 + 2 * len(xs), 2)
    for GOid, name in zip(data['GOid'], data['name']):
        labels.append(f'{name} ({GOid})')
    y0 += 2 * len(xs)
    ax.barh(ys, xs, label=aspect_label, height=1.25)
ax.invert_yaxis()
ax.set_ymargin(0.01)
ax.set_yticks(np.arange(0, 2 * len(labels), 2), labels, fontsize=6)
ax.set_xlabel('$\mathregular{-log_{10}}$(p-value)')
ax.set_ylabel('Term')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=len(bars))
fig.savefig('out/bar_enrichment_proteins.png')
plt.close()
