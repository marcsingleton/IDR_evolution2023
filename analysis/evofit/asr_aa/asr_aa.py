"""Infer ancestral amino acid distributions of IDRs."""

import os
import re
from collections import Counter
from subprocess import run

import skbio
from src.utils import read_fasta

ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'
min_length = 30
min_seqs = 20
tree_template = skbio.read('../../../data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)

OGid2regions = {}
with open('../../IDRpred/get_regions/out/regions.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        try:
            OGid2regions[OGid].append((start, stop, disorder))
        except KeyError:
            OGid2regions[OGid] = [(start, stop, disorder)]

if not os.path.exists('out/'):
    os.mkdir('out/')

for OGid, regions in OGid2regions.items():
    # Load MSA
    msa = []
    for header, seq in read_fasta(f'../../../data/alignments/fastas/{OGid}.afa'):
        ppid = re.search(ppid_regex, header).group(1)
        spid = re.search(spid_regex, header).group(1)
        msa.append({'ppid': ppid, 'spid': spid, 'seq': seq})

    # Check regions and merge if necessary
    partitions = []
    disorder_partition = {'name': 'disorder',
                          'matrix': '../iqtree_merge/out/50R_disorder.paml'}
    order_partition = {'name': 'order',
                       'matrix': 'LG'}

    disorder_lengths = []
    order_lengths = []
    for record in msa:
        seq = record['seq']
        disorder_seq = ''.join([seq[start:stop] for start, stop, disorder in regions if disorder])
        disorder_counts = Counter(disorder_seq)
        disorder_lengths.append(len(disorder_seq) - disorder_counts['-'] + disorder_counts['.'])

        seq = record['seq']
        order_seq = ''.join([seq[start:stop] for start, stop, disorder in regions if not disorder])
        order_counts = Counter(order_seq)
        order_lengths.append(len(order_seq) - order_counts['-'] + order_counts['.'])

    disorder_condition = sum([disorder_length >= min_length for disorder_length in disorder_lengths]) >= min_seqs
    order_condition = sum([order_length >= min_length for order_length in order_lengths]) >= min_seqs
    if disorder_condition and order_condition:
        disorder_partition['regions'] = [(start, stop) for start, stop, disorder in regions if disorder]
        disorder_partition['spids'] = [record['spid'] for record, disorder_length in zip(msa, disorder_lengths)
                                       if disorder_length >= min_length]
        partitions.append(disorder_partition)
        order_partition['regions'] = [(start, stop) for start, stop, disorder in regions if not disorder]
        order_partition['spids'] = [record['spid'] for record, order_length in zip(msa, order_lengths)
                                    if order_length >= min_length]
        partitions.append(order_partition)
    elif disorder_condition:
        disorder_partition['regions'] = [(0, len(msa[0]['seq']))]
        disorder_partition['spids'] = [record['spid'] for record in msa]
        partitions.append(disorder_partition)
    elif order_condition:
        order_partition['regions'] = [(0, len(msa[0]['seq']))]
        order_partition['spids'] = [record['spid'] for record in msa]
        partitions.append(order_partition)
    else:
        continue

    with open(f'out/{OGid}.tsv', 'w') as file:
        file.write('name\tregions\n')
        for partition in partitions:
            name, regions = partition['name'], partition['regions']
            regionstring = ','.join([f'{start}-{stop}' for start, stop in regions])
            file.write(f'{name}\t{regionstring}\n')

    for partition in partitions:
        name, matrix, regions, spids = partition['name'], partition['matrix'], partition['regions'], partition['spids']

        # Write region as MSA
        with open(f'out/{OGid}_{name}.afa', 'w') as file:
            for record in msa:
                ppid, spid, seq = record['ppid'], record['spid'], record['seq']
                if spid not in spids:
                    continue
                seq = ''.join([seq[start:stop] for start, stop in regions])
                seqstring = '\n'.join([seq[i:i+80] for i in range(0, len(seq), 80)])
                file.write(f'>{spid} {ppid}\n{seqstring}\n')

        # Prune missing species from tree
        tree = tree_template.shear(spids)
        skbio.io.write(tree, format='newick', into=f'out/{OGid}_{name}.nwk')

        cmd = (f'../../../bin/iqtree -s out/{OGid}_{name}.afa '
               f'-m {matrix}+I+G -keep-ident '
               f'-t out/{OGid}_{name}.nwk -blscale '
               f'-pre out/{OGid}_{name}')
        run(cmd, shell=True, check=True)

"""
NOTES
The documentation for tree and branch arguments to IQ-TREE does not entirely describe what is optimized and what is
fixed. For example, using -te fixes the tree topology, but it scales the branch lengths individually. Using -t in
combination with -blscale will also fix the tree topology but it will scale all branch lengths by a constant value.
Finally, using -t alone will perform a full tree search and branch length optimization starting at the given tree (as
described in the documentation.)

When the number of parameters exceeds the number of columns, IQ-TREE cautions that the parameter estimates are
unreliable. Since a majority of the parameters are branch lengths, I attempted to reduce the model complexity by using
the -blscale option. Unfortunately, the -blscale option does not play nicely with partitioned models, and IQ-TREE
crashes when both are used together. Originally, I ignored this issue and fit partitioned models with a fixed topology
(-te) since I was mostly interested in using the models to generate "plausible" ancestral sequences. However, when I
tried using the rate estimates from these models as a measure of the amount of sequence divergence in the alignment, I
found they were sometimes overfit and uninterpretable. Because the branch lengths were scaled individually, alignments
with many invariant columns would collapse many branch lengths to near 0. As a result, most rate categories had rates
which were effectively zero in these cases. This made it difficult to interpret the meaning of the rates or their
posterior probabilities. Thus, I re-wrote the code to manually partition the alignments.

To prevent poor fits from a lack of data, a partition is only created if there are a minimum of 20 sequences with at 
least 30 non-gap symbols in that class. If one of the classes meets these conditions but the other does not, 
the small class is consolidated into the large one. If neither class passes, the alignment is skipped. These rules 
ensure that any alignment with region in the final set of regions is reconstructed. They also ensure any regions in 
the final set are reconstructed with a model that is (largely) fit to its class.
"""