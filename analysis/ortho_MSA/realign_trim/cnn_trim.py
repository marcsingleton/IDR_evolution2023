"""Trim alignments, removing spurious sequences and regions."""

import os

import numpy as np
import scipy.ndimage as ndimage
import skbio
import tensorflow as tf
from src.utils import read_fasta


def get_slices(posterior, posterior_high, posterior_low):
    slices = []
    for region, in ndimage.find_objects(ndimage.label(posterior >= posterior_high)[0]):
        start = region.start  # start of left margin
        while start-1 >= 0 and posterior[start-1] >= posterior_low:
            start -= 1

        stop = region.stop - 1  # stop of right margin
        while stop+1 < len(posterior) and posterior[stop+1] >= posterior_low:
            stop += 1

        slices.append(slice(start, stop))

    return slices


alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-']
sym2idx = {sym: i for i, sym in enumerate(alphabet)}
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'

posterior_high = 0.75
posterior_low = 0.01
sym_min = 5  # Minimum number of symbols in consensus column

OGids = []
with open('../OG_filter/out/OG_filter.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGids.append(fields['OGid'])

model = tf.keras.models.load_model('../realign_cnn/out/model.h5')

if not os.path.exists('out/'):
    os.mkdir('out')

for OGid in OGids:
    try:
        msa1 = read_fasta(f'../align_fastas1/out/{OGid}.afa')
    except FileNotFoundError:
        msa1 = read_fasta(f'../align_fastas2/out/{OGid}.afa')

    # Convert alignment to indices and make profile
    msa2 = []
    for i, (header, seq1) in enumerate(msa1):
        seq2 = [sym2idx.get(sym, sym2idx['X']) for sym in seq1]  # All non-standard symbols mapped to X
        msa2.append(seq2)
    msa2 = tf.keras.utils.to_categorical(msa2, len(alphabet))

    profile = np.empty(msa2.shape)
    profile[:] = msa2.sum(axis=0) / len(msa2)

    # Identify trims
    outputs = tf.squeeze(model([profile, msa2]))

    trim_array = np.full((len(msa1), len(msa1[0][1])), False)
    for i, output in enumerate(outputs):
        slices = get_slices(output, posterior_high, posterior_low)
        for s in slices:
            trim_array[i, s] = True

    # Identify gaps
    gap_array = np.full((len(msa1), len(msa1[0][1])), False)
    for i, (_, seq) in enumerate(msa1):
        for j, sym in enumerate(seq):
            if sym == '-':
                gap_array[i, j] = True

    # Identify consensus columns
    counts = len(msa1) - np.logical_or(trim_array, gap_array).sum(axis=0)  # Number of non-gap symbols after trimming
    rfs = ['x' if count >= sym_min else '.' for count in counts]  # Metadata for marking consensus columns in profile HMM (ReFerence annotation)

    # Replace trimmed sequences with gaps in consensus columns
    msa3 = []
    for (header, seq1), trim_seq in zip(msa1, trim_array):
        seq3 = []
        for sym, trim, rf in zip(seq1, trim_seq, rfs):
            if rf == 'x' and trim:
                seq3.append('-')
            else:
                seq3.append(sym)
        msa3.append((header, ''.join(seq3)))

    # Write to file
    msa3 = skbio.TabularMSA([skbio.Protein(seq, metadata={'description': header}) for header, seq in msa3],
                            positional_metadata={'RF': rfs})
    msa3.write(f'out/{OGid}.sto', 'stockholm')

"""
../align_fastas1/align_fastas1.py
    ../align_fastas1/out/*.afa
../align_fastas2/align_fastas2.py
    ../align_fastas2/out/*.afa
../realign_cnn/realign_cnn.py
    ../realign_cnn/out/model.h5
../OG_filter/OG_filter.py
    ../OG_filter/out/OG_filter.tsv
"""