"""Remove sequences and regions from segments that do not pass quality filters."""

import re
import os

from src.utils import read_fasta


def spid_filter(spids):
    conditions = [({'dnov', 'dvir'}, 1),
                  ({'dmoj', 'dnav'}, 1),
                  ({'dinn', 'dgri', 'dhyd'}, 2),
                  ({'dgua', 'dsob'}, 1),
                  ({'dbip', 'dana'}, 1),
                  ({'dser', 'dkik'}, 1),
                  ({'dele', 'dfik'}, 1),
                  ({'dtak', 'dbia'}, 1),
                  ({'dsuz', 'dspu'}, 1),
                  ({'dsan', 'dyak'}, 1),
                  ({'dmel'}, 1),
                  ({'dmau', 'dsim', 'dsec'}, 1)]
    return all([len(spids & group) >= num for group, num in conditions])


def has_overlap(start1, stop1, start2, stop2):
    if stop1 < start1:
        raise ValueError('stop1 < start1')
    if stop2 < start2:
        raise ValueError('stop2 < start2')
    if start1 == stop1 or start2 == stop2:
        return False
    if stop2 < stop1:
        start1, start2 = start2, start1
        stop1, stop2 = stop2, stop1
    return stop1 > start2


ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'

spid_min = 20
alphabet = {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-', '.'}

# Load regions
OGid2regions = {}
with open('../get_regions/out/regions.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder']
        try:
            OGid2regions[OGid].append((start, stop, disorder))
        except KeyError:
            OGid2regions[OGid] = [(start, stop, disorder)]

# Filter regions
record_sets = {min_length: [] for min_length in range(10, 105, 5)}
for OGid, regions in OGid2regions.items():
    # Load MSA
    msa = []
    for header, seq in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        spid = re.search(spid_regex, header).group(1)
        msa.append({'ppid': ppid, 'spid': spid, 'seq': seq})

    # Get missing segments
    ppid2trims = {}
    with open(f'../../../data/alignments/trims/{OGid}.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            trims = []
            for trim in fields['slices'].split(','):
                if trim:
                    start, stop = trim.split('-')
                    trims.append((int(start), int(stop)))
            ppid2trims[fields['ppid']] = trims

    for region in regions:
        # Get indices and length
        region_start, region_stop = region[0], region[1]
        disorder = region[2]

        # Extract and filter segments
        segment_sets = {min_length: [] for min_length in record_sets}
        for record in msa:
            ppid, spid, segment = record['ppid'], record['spid'], record['seq'][region_start:region_stop]
            trims = ppid2trims[ppid]

            # Filter by length, symbols, and overlap with missing trims
            length, is_standard = 0, True
            for sym in segment:
                if sym not in ['-', '.']:
                    length += 1
                if sym not in alphabet:
                    is_standard = False
            no_missing = all([not has_overlap(region_start, region_stop, trim_start, trim_stop) for trim_start, trim_stop in trims])
            for min_length, segments in segment_sets.items():
                if length >= min_length and is_standard and no_missing:
                    segments.append((ppid, spid))

        # Filter by phylogenetic diversity
        for min_length, segments in segment_sets.items():
            ppids = [ppid for ppid, _ in segments]
            spids = {spid for _, spid in segments}
            if len(spids) >= spid_min and spid_filter(spids):
                record_sets[min_length].append((OGid, str(region_start), str(region_stop), disorder, ','.join(ppids)))

# Write records to file
if not os.path.exists('out/'):
    os.mkdir('out/')

for min_length, records in record_sets.items():
    with open(f'out/regions_{min_length}.tsv', 'w') as file:
        file.write('OGid\tstart\tstop\tdisorder\tppids\n')
        for record in records:
            file.write('\t'.join(record) + '\n')
