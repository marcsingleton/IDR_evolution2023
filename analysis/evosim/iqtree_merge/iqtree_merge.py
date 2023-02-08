"""Merge individual IQ-TREE models into consensus model."""

import os
import re
from collections import namedtuple

import numpy as np
from src.utils import read_iqtree

paml_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
labels = ['0R_disorder', '50R_disorder', '100R_disorder', '0R_order', '50R_order', '100R_order']
labels_suffix = r'_[0-9]+'
Record = namedtuple('Record', ['label', 'ematrix', 'freqs'])

# Load IQ-TREE matrices
records = {}
for label in labels:
    file_labels = []
    for path in os.listdir('../iqtree_fit/out/'):
        match = re.match(f'({label}{labels_suffix})\.iqtree', path)
        if match:
            file_labels.append(match.group(1))

    ematrix_stack = []
    freqs_stack = []
    for file_label in sorted(file_labels):
        record = read_iqtree(f'../iqtree_fit/out/{file_label}.iqtree', norm=True)
        ematrix, freqs = record['ematrix'], record['freqs']

        ematrix_stack.append(ematrix)
        freqs_stack.append(freqs)
    records[label] = Record(label,
                            np.stack(ematrix_stack),
                            np.stack(freqs_stack))

if not os.path.exists('out/'):
    os.mkdir('out/')

for label, record in records.items():
    ematrix = record.ematrix.mean(axis=0)
    freqs = record.freqs.mean(axis=0)

    with open(f'out/{label}.paml', 'w') as file:
        for i in range(1, len(paml_order)):
            row = ematrix[i, :i]
            file.write(' '.join([f'{value:f}' for value in row]) + '\n')
        file.write('\n' + ' '.join([f'{value:f}' for value in freqs]))
