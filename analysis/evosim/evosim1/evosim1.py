"""Simulate sequence evolution."""

import json
import os
import re

import numpy as np
import scipy.stats as stats
import skbio
from src.utils import read_fasta, read_paml
from src.evosim.asr import SeqEvolver, simulate_tree


def load_model(path):
    matrix, freqs = read_paml(path, norm=True)
    matrix = freqs * matrix
    np.fill_diagonal(matrix, -matrix.sum(axis=1))
    return matrix, freqs


spid_regex = r'spid=([a-z]+)'

alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
sym2idx = {sym: i for i, sym in enumerate(alphabet)}
idx2sym = {i: sym for i, sym in enumerate(alphabet)}

# Load models
tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
indel_ratio = 0.5  # Fraction of indel events which are insertions

LG_matrix, LG_freqs = load_model('../../../data/matrices/LG.paml')
disorder_matrix, disorder_freqs = load_model('../iqtree_merge/out/50R_disorder.paml')
rate_matrices = {0: disorder_matrix,
                 1: LG_matrix}
sym_dists = {0: disorder_freqs,
             1: LG_freqs}
name2id = {'disorder': 0,
           'order': 1}

# Make indel dists
insertion_dists = {0: stats.geom(0.8), 1: stats.geom(0.75)}
deletion_dists = {0: stats.geom(0.6), 1: stats.geom(0.55)}

OGids = [path.removesuffix('.afa') for path in os.listdir('../asr_generate/out/') if path.endswith('.afa')]
for OGid in OGids:
    # Load FASTA
    fasta = list(read_fasta(f'../asr_generate/out/{OGid}.afa'))
    length = len(fasta[0][1])

    # Load tree
    spids = []
    for header, _ in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        spid = re.search(spid_regex, header).group(1)
        spids.append(spid)
    tree = tree_template.shear(spids)
    tree.length = 0

    # Amino acid rates
    aa_dist = np.load(f'../asr_root/out/{OGid}_aa.npy')
    aa_conditional = aa_dist / aa_dist.sum(axis=0)

    with open(f'../asr_root/out/{OGid}_aa_model.json') as file:
        partitions = json.load(file)
    aa_rate_values = np.zeros_like(aa_conditional)
    for partition in partitions.values():
        partition_regions = partition['regions']
        partition_rates = partition['speed'] * np.array([[[r]] for r, _ in partition['rates']])
        for start, stop in partition_regions:
            aa_rate_values[:, :, start:stop] = partition_rates
    aa_rates = (aa_conditional * aa_rate_values).sum(axis=0)

    # Indel rates
    indel_rates = np.zeros(aa_rates.shape[-1])
    if os.path.exists(f'../asr_root/out/{OGid}_indel.npy'):
        character_dist = np.load(f'../asr_root/out/{OGid}_indel.npy')
        character_rate_dist = character_dist.sum(axis=1)

        with open(f'../asr_root/out/{OGid}_indel_model.json') as file:
            partition = json.load(file)
        character_rate_values = partition['speed'] * np.array([[r] for r, _ in partition['rates']])
        character_rates = (character_rate_dist * character_rate_values).sum(axis=0)

        id2indel = {}
        with open(f'../asr_indel/out/{OGid}.tsv') as file:
            field_names = file.readline().rstrip('\n').split('\t')
            for line in file:
                fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
                character_id, start, stop = int(fields['character_id']), int(fields['start']), int(fields['stop'])
                id2indel[character_id] = (start, stop)

        for character_id, indel in id2indel.items():
            start, stop = indel
            indel_rates[start] += character_rates[character_id] / 2
            indel_rates[stop - 1] += character_rates[character_id] / 2

    # Partition regions
    partition_template = np.empty(length, dtype=int)
    with open(f'../asr_root/out/{OGid}_aa_model.json') as file:
        partitions = json.load(file)
    for partition_name, partition in partitions.items():
        partition_id = name2id[partition_name]
        for start, stop in partition['regions']:
            partition_template[start:stop] = partition_id

    # Evolve sequences along tree
    for sample_id, (header, seq) in enumerate(fasta):
        # Construct root-specific SeqEvolver arrays
        seq = np.array([sym2idx.get(sym, -1) for sym in seq])  # Use -1 for gap symbols
        activities = np.array([False if sym == -1 else True for sym in seq])
        residue_ids = np.arange(-1, length)
        rate_coefficients = np.stack([[aa_rates[idx, j] if idx != -1 else 0 for j, idx in enumerate(seq)],
                                      indel_ratio * indel_rates,
                                      ((1 - indel_ratio) * indel_rates)])

        # Insert immortal link
        j = np.nonzero(activities)[0][0]  # Index of first active symbol
        seq = np.insert(seq, 0, seq[j])
        rate_coefficients = np.insert(rate_coefficients, 0, [0, rate_coefficients[1, j], 0], axis=1)
        activities = np.insert(activities, 0, True)
        partition_ids = np.insert(partition_template, 0, partition_template[j])

        # Evolve! (and extract results)
        rng = np.random.default_rng(sample_id)
        evoseq = SeqEvolver(seq, rate_coefficients, activities, residue_ids, partition_ids,
                            rate_matrices, sym_dists, insertion_dists, deletion_dists, rng)
        _, evoseqs = simulate_tree(tree, evoseq, rng)

        unaligned_records = []
        for spid, evoseq in evoseqs:
            seq = []
            for idx, activity in zip(evoseq.seq[1:], evoseq.activities[1:]):  # Exclude immortal link
                if activity:
                    seq.append(idx2sym.get(idx, '-'))
                else:
                    seq.append('-')
            unaligned_records.append((spid, seq, list(evoseq.residue_ids[1:])))

        # Align sequences
        aligned_ids = []
        aligned_records = [(spid, []) for spid, _, _ in unaligned_records]
        spid2idx = {spid: 0 for spid, _, _ in unaligned_records}
        while any([spid2idx[spid] < len(seq) for spid, seq, _ in unaligned_records]):
            # Collect residue ids
            idx_records = []
            for spid, seq, residue_ids in unaligned_records:
                idx = spid2idx[spid]
                residue_id = residue_ids[idx] if idx < len(seq) else -1
                idx_records.append((spid, residue_id))

            # Find spids with priority id
            max_id = max([residue_id for _, residue_id in idx_records])
            spids = {spid for spid, residue_id in idx_records if residue_id == max_id}
            aligned_ids.append(str(max_id))

            # Append symbols to sequences
            for (spid, unaligned_seq, _), (spid, aligned_seq) in zip(unaligned_records, aligned_records):
                if spid in spids:
                    idx = spid2idx[spid]
                    aligned_seq.append(unaligned_seq[idx])
                    spid2idx[spid] += 1
                else:
                    aligned_seq.append('-')

        # Write alignments
        if not os.path.exists(f'out/{OGid}/'):
            os.makedirs(f'out/{OGid}/')

        with open(f'out/{OGid}/{header[1:]}.txt', 'w') as file:
            file.write(','.join(aligned_ids) + '\n')
        with open(f'out/{OGid}/{header[1:]}.afa', 'w') as file:
            for spid, seq in sorted(aligned_records):
                seqstring = '\n'.join([''.join(seq)[i:i+80] for i in range(0, len(seq), 80)])
                file.write(f'>{spid}\n{seqstring}\n')
