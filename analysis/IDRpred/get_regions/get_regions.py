"""Segment trimmed alignments into regions by averaging raw AUCpreD scores."""

import os
import re

import scipy.ndimage as ndimage
import numpy as np
import skbio
from src.phylo import get_brownian_weights
from src.utils import read_fasta


def get_complement_slices(slices, start=0, stop=None):
    """Return slices of complement of input slices.

    Parameters
    ----------
    slices: list of slice
        Must be sorted and merged.
    start: int
        Start of interval from which complement slices are given.
    stop: int
        Stop of interval from which complement slices are given.

    Returns
    -------
    complement: list of slice
    """
    complement = []
    if slices:
        start0, stop0 = slices[0].start, slices[0].stop
        if start < start0:
            complement.append(slice(start, start0))
        for s in slices[1:]:
            complement.append(slice(stop0, s.start))
            stop0 = s.stop
        if stop is None or stop0 < stop:
            complement.append(slice(stop0, stop))
    else:
        complement.append(slice(start, stop))
    return complement


def get_merged_slices(slices):
    """Return slices where overlapping slices are merged.

    Parameters
    ----------
    slices: list of slice

    Returns
    -------
    merged: list of slice
    """
    merged = []
    if slices:
        slices = sorted(slices, key=lambda x: x.start)
        start0, stop0 = slices[0].start, slices[0].stop
        for s in slices[1:]:
            if s.start > stop0:
                merged.append(slice(start0, stop0))
                start0, stop0 = s.start, s.stop
            elif s.stop > stop0:
                stop0 = s.stop
        merged.append((slice(start0, stop0)))  # Append final slice
    return merged


def load_scores(path):
    with open(path) as file:
        scores = []
        for line in file:
            if not line.startswith('#'):
                score = line.split()[3]
                scores.append(score)
    return scores


ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'

cutoff_high = 0.6
cutoff_low = 0.4
min_length = 10
structure = np.ones(3)

tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_template.tips())}

# Load error flags
OGid2flags = {}
with open('../get_scores/out/errors.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, ppid, error_flag = fields['OGid'], fields['ppid'], fields['error_flag'] == 'True'
        try:
            OGid2flags[OGid].append(error_flag)
        except KeyError:
            OGid2flags[OGid] = [error_flag]

# Convert error flags to successful OGids
OGids = []
for OGid, error_flags in sorted(OGid2flags.items()):
    if not any(error_flags):
        OGids.append(OGid)

records = []
for OGid in OGids:
    # Load MSA
    msa = []
    for header, seq in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        spid = re.search(spid_regex, header).group(1)
        msa.append({'ppid': ppid, 'spid': spid, 'seq': seq})
    msa = sorted(msa, key=lambda x: tip_order[x['spid']])

    # Get missing segments
    ppid2missing = {}
    with open(f'../../../data/alignments/missing/{OGid}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            missing = []
            for s in fields['slices'].split(','):
                if s:
                    start, stop = s.split('-')
                    missing.append((int(start), int(stop)))
            ppid2missing[fields['ppid']] = missing

    # Align scores and interpolate between gaps that are not missing segments
    aligned_scores = np.full((len(msa), len(msa[0]['seq'])), np.nan)
    for i, record in enumerate(msa):
        ppid, seq = record['ppid'], record['seq']
        scores = load_scores(f'../get_scores/out/{OGid}/{ppid}.diso_noprof')  # Remove anything after trailing .
        idx = 0
        for j, sym in enumerate(seq):
            if sym not in ['-', '.']:
                aligned_scores[i, j] = scores[idx]
                idx += 1

        nan_idx = np.isnan(aligned_scores[i])
        range_scores = np.arange(len(msa[0]['seq']))
        interp_scores = np.interp(range_scores[nan_idx], range_scores[~nan_idx], aligned_scores[i, ~nan_idx],
                                  left=np.nan, right=np.nan)
        aligned_scores[i, nan_idx] = interp_scores

        for start, stop in missing[ppid]:
            aligned_scores[i, start:stop] = np.nan
    aligned_scores = np.ma.masked_invalid(aligned_scores)

    # Get Brownian weights and calculate root score
    spids = [record['spid'] for record in msa]
    tree = tree_template.shear(spids)

    tips, weights = get_brownian_weights(tree)
    weight_dict = {tip.name: weight for tip, weight in zip(tips, weights)}
    weight_array = np.zeros((len(msa), 1))
    for i, record in enumerate(msa):
        weight_array[i] = weight_dict[record['spid']]

    weight_sum = (weight_array * ~aligned_scores.mask).sum(axis=0)
    root_scores = (weight_array * aligned_scores).sum(axis=0) / weight_sum

    slices = []
    binary1 = root_scores >= cutoff_high
    binary2 = ndimage.binary_dilation(root_scores >= cutoff_low, structure=structure)
    for s, in ndimage.find_objects(ndimage.label(binary1)[0]):
        if s.stop - s.start < min_length:
            continue

        start = s.start
        while start-1 >= 0 and binary2[start-1]:
            start -= 1
        stop = s.stop
        while stop+1 <= len(root_scores) and binary2[stop]:
            stop += 1
        slices.append(slice(start, stop))
    disorder_slices = get_merged_slices(slices)
    order_slices = get_complement_slices(disorder_slices, stop=len(root_scores))

    for s in disorder_slices:
        records.append((OGid, s.start, s.stop, True))
    for s in order_slices:
        records.append((OGid, s.start, s.stop, False))

# Write segments to file
if not os.path.exists('out/'):
    os.mkdir('out/')

with open('out/regions.tsv', 'w') as file:
    file.write('OGid\tstart\tstop\tdisorder\n')
    for record in sorted(records, key=lambda x: (x[0], x[1])):
        file.write('\t'.join([str(field) for field in record]) + '\n')
