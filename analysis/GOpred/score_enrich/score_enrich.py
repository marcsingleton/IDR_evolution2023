"""Calculate GO term enrichment on regions with high rates of score evolution."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def hypergeom_test(k, M, n, N):
    """Return the p-value for a hypergeometric test."""
    kmax = min(n, N)
    ks = np.arange(k, kmax + 1)
    pmfs = stats.hypergeom.pmf(k=ks, M=M, n=n, N=N)
    pvalue = pmfs.sum()
    return pvalue


min_length = 30

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 0.5
min_indel_rate = 0.1

# Load regions as segments
rows = []
with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
all_regions = pd.DataFrame(rows)

gaf = pd.read_table('../filter_GAF/out/GAF_drop.tsv')

asr_rates = pd.read_table(f'../../evofit/asr_stats/out/regions_{min_length}/rates.tsv')
asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
column_idx = ['OGid', 'start', 'stop', 'disorder']
region_keys = asr_rates.loc[row_idx, column_idx]

contrasts = pd.read_table(f'../../brownian/get_contrasts/out/scores/contrasts_{min_length}.tsv', skiprows=[1])  # Skip group row
contrasts = region_keys.merge(contrasts, how='left', on=['OGid', 'start', 'stop'])
contrasts = contrasts.set_index(['OGid', 'start', 'stop', 'disorder', 'contrast_id'])

reference_gaf = region_keys.merge(gaf, how='inner', on=['OGid', 'start', 'stop', 'disorder'])

rates = (contrasts ** 2).groupby(['OGid', 'start', 'stop', 'disorder']).mean()
quantile = rates['score_fraction'].quantile(0.9, interpolation='higher')  # Capture at least 90% of data with higher
enrichment_keys = rates[rates['score_fraction'] > quantile].index.to_frame(index=False)  # False forces re-index
enrichment_gaf = reference_gaf.merge(enrichment_keys, how='inner', on=['OGid', 'start', 'stop', 'disorder'])
terms = list(enrichment_gaf[['aspect', 'GOid', 'name']].drop_duplicates().itertuples(index=False, name=None))

rows = []
for aspect, GOid, name in terms:
    k = enrichment_gaf[enrichment_gaf['GOid'] == GOid].groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
    M = reference_gaf.groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
    n = reference_gaf[reference_gaf['GOid'] == GOid].groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
    N = enrichment_gaf.groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
    pvalue = hypergeom_test(k, M, n, N)
    rows.append({'pvalue': pvalue, 'aspect': aspect, 'GOid': GOid, 'name': name})
result = pd.DataFrame(rows).sort_values(by=['aspect', 'pvalue'], ignore_index=True)

if not os.path.exists('out/'):
    os.mkdir('out/')

result.to_csv('out/pvalues.tsv', sep='\t', index=False)

y0, labels = 0, []
fig, ax = plt.subplots(figsize=(6.4, 6.4), layout='constrained')
for aspect, aspect_label in [('P', 'Process'), ('F', 'Function'), ('C', 'Component')]:
    data = result[(result['aspect'] == aspect) & (result['pvalue'] <= 0.2)]
    xs = -np.log10(data['pvalue'])
    ys = np.arange(y0, y0 + len(xs))
    labels.extend([f'{name} ({GOid})' for GOid, name in zip(data['GOid'], data['name'])])
    y0 += len(xs)
    ax.barh(ys, xs, label=aspect_label)
ax.invert_yaxis()
ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
ax.set_xlabel('$\mathregular{-log_{10}}$(p-value)')
ax.set_ylabel('Term')
ax.legend()
fig.savefig('out/bar_enrichment.png')
plt.close()
