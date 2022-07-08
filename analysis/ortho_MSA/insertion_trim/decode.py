"""Decode segments into posterior state probabilities."""

import json
import multiprocessing as mp
import os
import re

import homomorph
import numpy as np
import skbio
from src.utils import read_fasta
from src.ortho_MSA import utils


def decode(OGid, model_json, tree_template):
    # Load MSA
    msa = []
    for header, seq in read_fasta(f'../realign_hmmer/out/mafft/{OGid}.afa'):
        spid = re.search(spid_regex, header).group(1)
        msa.append({'spid': spid, 'seq': seq})

    # Create emission sequence
    col0 = []
    emit_seq = []
    for j in range(len(msa[0]['seq'])):
        col = [1 if msa[i]['seq'][j] in ['-', '.'] else 0 for i in range(len(msa))]
        emit0 = sum([c0 == c for c0, c in zip(col0, col)])
        emit_seq.append(emit0)  # The tree probabilities are pre-calculated, so emission value is its index
        col0 = col
    emit_seq = np.array(emit_seq)

    # Load tree and convert to vectors at tips
    tree = tree_template.shear([record['spid'] for record in msa])
    tips = {tip.name: tip for tip in tree.tips()}
    for record in msa:
        spid, seq = record['spid'], record['seq']
        conditional = np.zeros((2, len(seq)))
        for j, sym in enumerate(seq):
            if sym in ['-', '.']:
                conditional[0, j] = 1
            else:
                conditional[1, j] = 1
        tip = tips[spid]
        tip.conditional = conditional

    # Instantiate model
    e_dists_rv = {}
    for s, e_dist in model_json['e_dists'].items():
        a, b, pi, q0, q1, p0, p1 = [e_dist[param] for param in ['a', 'b', 'pi', 'q0', 'q1', 'p0', 'p1']]
        array1 = utils.get_betabinom_pmf(emit_seq, len(msa), a, b)
        array2 = utils.get_tree_pmf(tree, pi, q0, q1, p0, p1)
        e_dists_rv[s] = utils.ArrayRV(array1 * array2)
    model = homomorph.HMM(model_json['t_dists'], e_dists_rv, model_json['start_dist'])

    # Decode states and write
    idx_seq = list(range(len(emit_seq)))  # Everything is pre-calculated, so emit_seq is the emit index
    fbs = model.forward_backward(idx_seq)

    state_set = sorted(model_json['t_dists'])
    with open(f'out/{OGid}.tsv', 'w') as file:
        file.write('\t'.join(state_set) + '\n')
        for fb in zip(*[fbs[state] for state in state_set]):
            file.write('\t'.join([str(v) for v in fb]) + '\n')


num_processes = int(os.environ['SLURM_CPUS_ON_NODE'])
spid_regex = r'spid=([a-z]+)'

if __name__ == '__main__':
    tree_template = skbio.read('../../ortho_tree/consensus_LG/out/100R_NI.nwk', 'newick', skbio.TreeNode)

    # Load OGids
    OGids = []
    with open('../realign_hmmer/out/errors.tsv') as file:
        field_names = file.readline().rstrip('\n').split('\t')
        for line in file:
            fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
            OGid, error_flag = fields['OGid'], fields['error_flag']
            if error_flag == 'False':
                OGids.append(OGid)

    with open('../insertion_hmm/out/model.json') as file:
        model_json = json.load(file)

    if not os.path.exists('out/'):
        os.mkdir('out/')

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(decode, [(OGid, model_json, tree_template) for OGid in OGids])

"""
DEPENDENCIES
../../ortho_tree/consensus_LG/consensus_LG.py
    ../../ortho_tree/consensus_LG/out/100R_NI.nwk
../insertion_hmm/fit.py
    ../insertion_hmm/out/model.json
../realign_hmmer/realign_hmmer.py
    ../realign_hmmer/out/errors.tsv
    ../realign_hmmer/out/mafft/*.afa
"""