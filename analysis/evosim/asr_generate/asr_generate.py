"""Generate samples from ASR distributions."""

import os

import numpy as np

rng = np.random.default_rng(930715)
num_samples = 1000
alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
sym2idx = {sym: idx for idx, sym in enumerate(alphabet)}
idx2sym = {idx: sym for idx, sym in enumerate(alphabet)}

if not os.path.exists('out/'):
    os.mkdir('out/')

OGids = sorted([path.removesuffix('_aa.npy') for path in os.listdir('../asr_root/out/') if path.endswith('_aa.npy')])  # Ensure consistent order and thus consistent samples
for OGid in OGids:
    # Generate amino acid sequence
    aa_dist = np.load(f'../asr_root/out/{OGid}_aa.npy')
    aa_sym_dist = aa_dist.sum(axis=0)

    seqs = []
    for i in range(aa_sym_dist.shape[1]):
        column = rng.choice(20, size=num_samples, p=aa_sym_dist[:, i])
        seqs.append(column)
    seqs = np.stack(seqs, axis=1)

    # Generate indels (if applicable)
    if os.path.exists(f'../asr_root/out/{OGid}_indel.npy'):
        indel_dist = np.load(f'../asr_root/out/{OGid}_indel.npy')
        indel_sym_dist = indel_dist.sum(axis=0)

        id2indel = {}
        with open(f'../asr_indel/out/{OGid}.tsv') as file:
            field_names = file.readline().rstrip('\n').split('\t')
            for line in file:
                fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
                character_id, start, stop = int(fields['character_id']), int(fields['start']), int(fields['stop'])
                id2indel[character_id] = (start, stop)

        for j in range(indel_sym_dist.shape[1]):
            ps = rng.random(size=num_samples) > indel_sym_dist[0, j]
            for i, p in enumerate(ps):
                if p:
                    start, stop = id2indel[j]
                    seqs[i, start:stop] = sym2idx['-']

    with open(f'out/{OGid}.afa', 'w') as file:
        for i, seq in enumerate(seqs):
            seq = ''.join([idx2sym[idx] for idx in seq])
            seqstring = '\n'.join([seq[i:i+80] for i in range(0, len(seq), 80)])
            file.write(f'>seq{i:04}\n{seqstring}\n')

"""
NOTES
The output of the ancestral reconstructions gives a probability for each rate and symbol combination. One method of
sampling sequences and rates is directly from this posterior distribution. Thus, every draw for a position yields both a
symbol and rate, where the rates are the categories fit by the model. While this method would capture the uncertainty in
the rate inference, I feel it's both overly complex and a poor interpretation of the model in this context. The gamma
distribution actually describes continuous variation in rates even though it's discretized for computational
convenience here. It's also more realistic to expect the biological variation in rates to be continuous. Thus, rather
than sampling the rate of site as "invariant 10% of the time, slow 25% of the time...", it's more appropriate to take
the rate as a posterior average over rate categories. I feel this will yield better simulated alignments since the
rate variation will manifest as a continuum rather than a single site being fast in some simulations and slow in others.
"""