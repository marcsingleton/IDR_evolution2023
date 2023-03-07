"""Calculate features of segments in regions."""

import multiprocessing as mp
import os
import re
from collections import namedtuple

import src.brownian.features as features
from src.utils import read_fasta


ArgsRecord = namedtuple('ArgsRecord', ['OGid', 'start', 'stop', 'ppid', 'disorder', 'segment'])


def get_features(args):
    record = {('OGid', 'ids_group'): args.OGid,
              ('start', 'ids_group'): args.start,
              ('stop', 'ids_group'): args.stop,
              ('ppid', 'ids_group'): args.ppid}
    if not (len(args.segment) == 0 or 'X' in args.segment or 'U' in args.segment):
        record.update(features.get_features(args.segment, features.repeat_groups, features.motif_regexes))
    return record


num_processes = int(os.environ.get('SLURM_CPUS_ON_NODE', 1))
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'

if __name__ == '__main__':
    # Load regions
    OGid2regions = {}
    with open('../../IDRpred/get_regions/out/regions.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder']
            try:
                OGid2regions[OGid].append((start, stop, disorder))
            except KeyError:
                OGid2regions[OGid] = [(start, stop, disorder)]

    # Extract segments
    args = []
    for OGid, regions in OGid2regions.items():
        msa = read_fasta(f'../../../data/alignments/fastas/{OGid}.afa')
        msa = {re.search(ppid_regex, header).group(1): seq for header, seq in msa}

        for start, stop, disorder in regions:
            for ppid, seq in msa.items():
                segment = seq[start:stop].translate({ord('-'): None, ord('.'): None})
                args.append(ArgsRecord(OGid, start, stop, ppid, disorder, segment))

    # Calculate features
    with mp.Pool(processes=num_processes) as pool:
        records = pool.map(get_features, args, chunksize=50)

    # Write features to file
    if not os.path.exists('out/'):
        os.mkdir('out/')

    with open('out/features.tsv', 'w') as file:
        if records:
            field_names = list(records[0])
            file.write('\t'.join([feature_label for feature_label, _ in field_names]) + '\n')
            file.write('\t'.join([group_label for _, group_label in field_names]) + '\n')
        for record in records:
            file.write('\t'.join(str(record.get(field_name, 'nan')) for field_name in field_names) + '\n')
