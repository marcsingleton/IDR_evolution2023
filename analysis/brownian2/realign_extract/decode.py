"""Decode segments into posterior state probabilities."""

import json
import multiprocessing as mp
import os

import scipy.stats as stats
import src.hmm as hmm


class bernoulli_betabinom_gen:
    def pmf(self, x, p, n, a, b):
        pmf0 = stats.bernoulli.pmf(x[0], p)
        pmf1 = stats.betabinom.pmf(x[1], n, a, b)
        return pmf0 * pmf1

    def rvs(self, p, n, a, b, size=None, random_state=None):
        rvs0 = stats.bernoulli.rvs(p, size=size, random_state=random_state)
        rvs1 = stats.betabinom.rvs(n, a, b, size=size, random_state=random_state)
        if size is None:
            return rvs0, rvs1
        else:
            return list(zip(rvs0, rvs1))

bernoulli_betabinom = bernoulli_betabinom_gen()


class bernoulli_betabinom_frozen:
    def __init__(self, p, n, a, b):
        self._dist = bernoulli_betabinom_gen()
        self.p = p
        self.n = n
        self.a = a
        self.b = b

    def pmf(self, x):
        return self._dist.pmf(x, self.p, self.n, self.a, self.b)

    def rvs(self, size=None):
        return self._dist.rvs(self.p, self.n, self.a, self.b, size=size)


def load_msa(path):
    msa = []
    with open(path) as file:
        line = file.readline()
        while line:
            if line.startswith('>'):
                header = line
                line = file.readline()

            seqlines = []
            while line and not line.startswith('>'):
                seqlines.append(line.rstrip())
                line = file.readline()
            seq = ''.join(seqlines)
            msa.append((header, seq))
    return msa


def decode(OGid, params):
    msa = load_msa(f'../../ortho_MSA/realign_hmmer/out/{OGid}.mfa')

    idx = 0
    for j in range(len(msa[0][1])):
        for i in range(len(msa)):
            sym = msa[i][1][j]
            if sym == '.' or sym.islower():
                break
        else:
            idx = j
            break  # if no break exit
    msa = [(header, seq[idx:]) for header, seq in msa]

    idx = len(msa[0][1])
    for j in range(len(msa[0][1]), 0, -1):
        for i in range(len(msa)):
            sym = msa[i][1][j - 1]
            if sym == '.' or sym.islower():
                break
        else:
            idx = j
            break  # if no break exit
    msa = [(header, seq[:idx]) for header, seq in msa]

    # Create emission sequence
    col0 = []
    emits = []
    for j in range(len(msa[0][1])):
        col = [1 if msa[i][1][j] in ['-', '.'] else 0 for i in range(len(msa))]
        emit0 = all([c0 == c for c0, c in zip(col0, col)])
        emit1 = sum(col)
        emits.append((emit0, emit1))
        col0 = col

    # Instantiate model
    e_dists_rv = {state: bernoulli_betabinom_frozen(p, len(msa)-1, a, b) for state, (p, a, b) in params['e_dists'].items()}
    model = hmm.HMM(params['t_dists'], e_dists_rv, params['start_dist'])
    fbs = model.forward_backward(emits)

    # Decode states and write
    with open(f'out/{OGid}.tsv', 'w') as file:
        file.write('\t'.join(states) + '\n')
        for fb in zip(*[fbs[state] for state in states]):
            file.write('\t'.join([str(v) for v in fb]) + '\n')


num_processes = int(os.environ['SLURM_CPUS_ON_NODE'])
states = ['1A', '1B', '2', '3']

if __name__ == '__main__':
    with open('../segment_hmm3/out/model.json') as file:
        params = json.load(file)

    if not os.path.exists('out/'):
        os.mkdir('out/')

    with mp.Pool(processes=num_processes) as pool:
        args = [(path.split('.')[0], params) for path in os.listdir('../../ortho_MSA/realign_hmmer/out/') if path.endswith('.mfa')]
        pool.starmap(decode, args)

"""
DEPENDENCIES
../../ortho_MSA/realign_hmmer/realign_hmmer.py
../segment_hmm3/fit.py
    ../segment_hmm3/out/model.json
"""