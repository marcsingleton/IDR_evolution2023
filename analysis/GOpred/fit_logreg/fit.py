"""Fit rates to GO term logistic regression models."""

import os
import re

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from src.utils import read_fasta


def zscore(df):
    return (df - df.mean()) / df.std()


pdidx = pd.IndexSlice

ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
gnid_regex = r'gnid=([A-Za-z0-9_.]+)'
min_length = 30

# Load sequence data
ppid2gnid = {}
OGids = sorted([path.removesuffix('.afa') for path in os.listdir('../../../data/alignments/fastas/') if path.endswith('.afa')])
for OGid in OGids:
    for header, _ in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        gnid = re.search(gnid_regex, header).group(1)
        ppid2gnid[ppid] = gnid

# Load regions
rows = []
with open(f'../../IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        ppid = re.search(r'(FBpp[0-9]+)', fields['ppids']).group(1)
        rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder, 'gnid': ppid2gnid[ppid]})
regions = pd.DataFrame(rows)

# Load GOids
gnid2GOids = {}
with open('../filter_GAF/out/GAF_drop.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        gnid, GOid = fields['gnid'], fields['GOid']
        try:
            gnid2GOids[gnid].add(GOid)
        except KeyError:
            gnid2GOids[gnid] = {GOid}
GOids = set.union(*gnid2GOids.values())

contrasts = pd.read_table(f'../../brownian/get_contrasts/out/features/contrasts_{min_length}.tsv', skiprows=[1])
df1 = regions.merge(contrasts, how='right', on=['OGid', 'start', 'stop'])
df1 = df1.set_index(['OGid', 'start', 'stop', 'disorder', 'gnid', 'contrast_id'])

rates = (df1**2).groupby(['OGid', 'start', 'stop', 'disorder', 'gnid']).mean()
rates = zscore(rates)
disorder = rates.loc[pdidx[:, :, :, True], :]
order = rates.loc[pdidx[:, :, :, False], :]

if not os.path.exists('out/'):
    os.mkdir('out/')

rows = []
for data, label in [(disorder, 'disorder'), (order, 'order'), (rates, 'all')]:
    pca = PCA(n_components=10)
    transform = pca.fit_transform(data.to_numpy())[:, :5]

    for GOid in GOids:
        y_true = [GOid in gnid2GOids.get(gnid, set()) for gnid in data.index.get_level_values('gnid')]
        w = (len(y_true) - sum(y_true)) / sum(y_true)
        weights = [w if y else 1 for y in y_true]
        logreg = LogisticRegression(max_iter=500, penalty='none')
        logreg.fit(transform, y_true, sample_weight=weights)

        y_pred = logreg.predict(transform)
        acc = sum([y1 == y2 for y1, y2 in zip(y_true, y_pred)]) / len(y_true)
        sens = sum([(y1 == y2) and y1 for y1, y2 in zip(y_true, y_pred)]) / sum(y_true)
        spec = sum([(y1 == y2) and not y1 for y1, y2 in zip(y_true, y_pred)]) / (len(y_true) - sum(y_true))
        prec = sum([(y1 == y2) and y1 for y1, y2 in zip(y_true, y_pred)]) / sum(y_pred)

        rows.append({'GOid': GOid, 'label': label,
                     'accuracy': acc, 'sensitivity': sens, 'specificity': spec, 'precision': prec,
                     **{f'beta{i}': beta for i, beta in enumerate(logreg.coef_[0])}})
results = pd.DataFrame(rows)
results.to_csv('out/models.tsv', sep='\t', index=False)
