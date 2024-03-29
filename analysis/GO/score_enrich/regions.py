"""Calculate GO term enrichment on regions with high rates of score evolution."""

import os
from textwrap import dedent, fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from src.GO.enrich import hypergeom_test

min_length = 30

color3 = '#b07aa1'

# Load regions as segments
rows = []
with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
all_regions = pd.DataFrame(rows)

gaf = pd.read_table('../filter_GAF/out/regions/GAF_propagate.tsv')  # Use all terms

contrasts = pd.read_table(f'../../brownian/contrast_compute/out/scores/contrasts_{min_length}.tsv')
contrasts = all_regions.merge(contrasts, how='left', on=['OGid', 'start', 'stop'])
contrasts = contrasts.set_index(['OGid', 'start', 'stop', 'disorder', 'contrast_id'])

rates = (contrasts ** 2).groupby(['OGid', 'start', 'stop', 'disorder']).mean()
quantile = rates['score_fraction'].quantile(0.9, interpolation='higher')  # Capture at least 90% of data with higher

reference_gaf = all_regions.merge(gaf, how='inner', on=['OGid', 'start', 'stop', 'disorder'])
enrichment_keys = rates[rates['score_fraction'] > quantile].index.to_frame(index=False)  # False forces re-index
enrichment_gaf = reference_gaf.merge(enrichment_keys, how='inner', on=['OGid', 'start', 'stop', 'disorder'])

terms = list(enrichment_gaf[['aspect', 'GOid', 'name']].drop_duplicates().itertuples(index=False, name=None))
M = reference_gaf.groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
N = enrichment_gaf.groupby(['OGid', 'start', 'stop', 'disorder']).ngroups

rows = []
for aspect, GOid, name in terms:
    k = enrichment_gaf[enrichment_gaf['GOid'] == GOid].groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
    n = reference_gaf[reference_gaf['GOid'] == GOid].groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
    pvalue = hypergeom_test(k, M, n, N)
    rows.append({'pvalue': pvalue, 'k': k, 'n': n,
                 'aspect': aspect, 'GOid': GOid, 'name': name})
pvalues = pd.DataFrame(rows).sort_values(by=['aspect', 'pvalue'], ignore_index=True)

if not os.path.exists('out/'):
    os.mkdir('out/')

pvalues.to_csv('out/pvalues_regions.tsv', sep='\t', index=False)
with open('out/output_regions.txt', 'w') as file:
    output = f"""\
    M (reference size): {M}
    N (enrichment size): {N}
    Number of tests: {len(terms)}
    """
    file.write(dedent(output))  # Remove leading whitespace

fig, axs = plt.subplots(2, 1, gridspec_kw={'right': 0.85})
for ax in axs:
    ax.axvspan(quantile, rates['score_fraction'].max(), color='#e6e6e6')
    ax.hist(rates['score_fraction'], bins=150, color=color3)
    ax.set_ylabel('Number of regions')
axs[1].set_xlabel('Score rate')
axs[1].set_yscale('log')
fig.legend(handles=[Patch(facecolor=color3, label='all')], bbox_to_anchor=(0.85, 0.5), loc='center left')
fig.savefig('out/hist_numregions-score_rate.png')
plt.close()

fig, ax = plt.subplots(figsize=(6.4, 6.4), layout='constrained')
bars = [('P', 'Process'), ('F', 'Function'), ('C', 'Component')]
y0, labels = 0, []
for aspect, aspect_label in bars:
    data = pvalues[(pvalues['aspect'] == aspect) & (pvalues['pvalue'] <= 0.001)]
    xs = -np.log10(data['pvalue'])
    ys = np.arange(y0, y0 + 2 * len(xs), 2)
    for GOid, name in zip(data['GOid'], data['name']):
        labels.append(fill(f'{name} ({GOid})', 45))
    y0 += 2 * len(xs)
    ax.barh(ys, xs, label=aspect_label, height=1.25)
ax.invert_yaxis()
ax.set_ymargin(0.01)
ax.set_yticks(np.arange(0, 2 * len(labels), 2), labels, fontsize=8)
ax.set_xlabel('$\mathregular{-log_{10}}$(p-value)')
ax.set_ylabel('Term')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=len(bars))
fig.savefig('out/bar_enrichment_regions.png')
plt.close()
