"""Calculate overlaps of Pfam domains with IDRs."""

import os
import re
from itertools import groupby

import numpy as np
from utils import read_fasta


def make_get_line_OGid(field_names):
    idx = field_names.index('OGid')

    def get_line_OGid(line):
        fields = line.rstrip('\n').split('\t')
        return fields[idx]

    return get_line_OGid


spid_regex = r'spid=([a-z]+)'

# Load regions
OGid2regions = {}
with open('../../IDRpred/region_filter/out/regions_30.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        try:
            OGid2regions[OGid].append((start, stop, disorder))
        except KeyError:
            OGid2regions[OGid] = [(start, stop, disorder)]

# Load domains
OGid2domains = {}
with open('../../IDRpred/pfam_parse/out/domains.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    get_line_OGid = make_get_line_OGid(field_names)
    for OGid, group in groupby(file, get_line_OGid):
        domains = []
        for line in group:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            ali_from, ali_to = int(fields['ali_from']), int(fields['ali_to'])
            domains.append((ali_from, ali_to))
        OGid2domains[OGid] = domains

records = []
for OGid in sorted(OGid2regions):
    record = None
    for header, seq in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        spid = re.search(spid_regex, header).group(1)
        if spid == 'dmel':
            record = (header, seq)
    if record is None:
        raise RuntimeError(f'Alignment {OGid} does not have a sequence from dmel.')

    aligned_seq = record[1]
    unaligned_seq = aligned_seq.translate({ord('-'): None, ord('.'): None})
    offsets = np.concatenate([[0], np.cumsum([sym in ['-', '.'] for sym in aligned_seq])])  # Add initial of 0 so offsets don't include current index
    domain_mask = np.zeros(len(unaligned_seq), dtype=bool)
    for ali_from, ali_to in OGid2domains.get(OGid, []):  # If no domains, return empty list to prevent key error
        domain_mask[ali_from:ali_to] = True

    for aligned_start, aligned_stop, disorder in OGid2regions[OGid]:
        unaligned_start = aligned_start - offsets[aligned_start]
        unaligned_stop = aligned_stop - offsets[aligned_stop]
        length = unaligned_stop - unaligned_start
        overlap = domain_mask[unaligned_start: unaligned_stop].sum()
        record = {'OGid': OGid, 'start': aligned_start, 'stop': aligned_stop, 'disorder': disorder,
                  'length': length, 'overlap': overlap}
        records.append(record)

if not os.path.exists('out/'):
    os.mkdir('out/')

with open('out/overlaps.tsv', 'w') as file:
    if records:
        field_names = list(records[0])
        file.write('\t'.join(field_names) + '\n')
    for record in records:
        file.write('\t'.join(str(record[field_name]) for field_name in field_names) + '\n')
