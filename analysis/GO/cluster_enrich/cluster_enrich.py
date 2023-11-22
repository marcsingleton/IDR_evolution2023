"""Calculate GO term enrichment on regions selected from clusters of evolutionary signatures."""

import os

import pandas as pd
import skbio
from scipy.stats import false_discovery_control
from src.GO.enrich import hypergeom_test

pdidx = pd.IndexSlice
min_length = 30

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 1
min_indel_rate = 0.1

min_k = 2  # Minimum number of positive annotations in enrichment set

tree = skbio.read('../../brownian/model_stats/out/regions_30/hierarchy/heatmap_all_correlation.nwk', 'newick', skbio.TreeNode)
clusters = [('15126', '1'),
            ('15136', '2'),
            ('15107', '3'),
            ('15065', '4'),
            ('14930', '5'),
            ('14935', '6'),
            ('14890', '7'),
            ('14971', '8'),
            ('15086', '9'),
            ('14939', '10'),
            ('15104', '11'),
            ('15056', '12'),
            ('15100', '13'),
            ('15132', '14'),
            ('14889', '15'),
            ('15134', '16'),
            ('15053', '17'),
            ('15072', '18'),
            ('14948', '19'),
            ('15081', '20'),
            ('14731', '21'),
            ('15146', '22'),
            ('14741', '23'),
            ('14379', '24'),
            ('14944', '25'),
            ('15083', '26'),
            ('14988', '27'),
            ('15165', '28'),
            ('15123', '29'),
            ('14743', '30'),
            ('15098', '31'),
            ('15153', '32'),
            ('15035', '33'),
            ('15062', '34'),
            ('15159', '35'),
            ('15047', '36'),
            ('14916', '37'),
            ('15026', '38'),
            ('15102', '39')]

# Load regions
rows = []
with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
all_regions = pd.DataFrame(rows)

gaf = pd.read_table('../filter_GAF/out/regions/GAF_propagate.tsv')  # Use all terms

# Filter by rates
asr_rates = pd.read_table(f'../../evofit/asr_stats/out/regions_{min_length}/rates.tsv')
asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
column_idx = ['OGid', 'start', 'stop', 'disorder']
region_keys = asr_rates.loc[row_idx, column_idx]

reference_keys = region_keys[region_keys['disorder']].reset_index(drop=True)  # Filter again by disorder
reference_gaf = region_keys.merge(gaf, how='inner', on=['OGid', 'start', 'stop', 'disorder'])

pvalue_rows = []
cluster_rows = []
for root_id, cluster_id in clusters:
    root_node = tree.find(root_id)
    node_ids = [int(tip.name) for tip in root_node.tips()]
    enrichment_keys = reference_keys.iloc[node_ids]
    enrichment_gaf = reference_gaf.merge(enrichment_keys, how='inner', on=['OGid', 'start', 'stop', 'disorder'])

    terms = list(enrichment_gaf[['aspect', 'GOid', 'name']].drop_duplicates().itertuples(index=False, name=None))
    M = reference_gaf.groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
    N = enrichment_gaf.groupby(['OGid', 'start', 'stop', 'disorder']).ngroups

    for aspect, GOid, name in terms:
        k = enrichment_gaf[enrichment_gaf['GOid'] == GOid].groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
        n = reference_gaf[reference_gaf['GOid'] == GOid].groupby(['OGid', 'start', 'stop', 'disorder']).ngroups
        if k < min_k:
            continue
        pvalue = hypergeom_test(k, M, n, N)
        pvalue_rows.append({'cluster_id': int(cluster_id),
                            'pvalue': pvalue, 'k': k, 'n': n,
                            'aspect': aspect, 'GOid': GOid, 'name': name})

    cluster_rows.append({'cluster_id': int(cluster_id),
                         'num_regions': len(enrichment_keys), 'num_OGids': enrichment_keys['OGid'].nunique(),
                         'num_tests': len(terms),
                         'regions': ','.join([f'{row.OGid}-{row.start}-{row.stop}' for row in enrichment_keys.itertuples()])})
pvalues = pd.DataFrame(pvalue_rows).sort_values(by=['cluster_id', 'aspect', 'pvalue'], ignore_index=True)
clusters = pd.DataFrame(cluster_rows)

dfs = []
for _, group in pvalues.groupby('cluster_id'):
    group['pvalue_adj'] = false_discovery_control(group['pvalue'])
    dfs.append(group)
pvalues = pd.concat(dfs)[['cluster_id', 'pvalue', 'pvalue_adj', 'k', 'n', 'aspect', 'GOid', 'name']]

if not os.path.exists('out/'):
    os.mkdir('out/')

pvalues.to_csv('out/pvalues.tsv', sep='\t', index=False)
clusters.to_csv('out/clusters.tsv', sep='\t', index=False)
