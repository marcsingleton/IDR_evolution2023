"""Common functions for ASR."""

from copy import deepcopy

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import skbio
from numpy import log


class SeqEvolver:
    """A class for mutating sequences.

    Insertions and deletions are right-oriented. For insertions, the residues
    are inserted immediately left of the given index. For deletions, active
    residues are deleted beginning at the given index and moving left until the
    number of residues deleted is equal to the given length or no more residues
    remain.

    An "immortal link" (Thorne et al., 1991) can be included at the beginning
    to allow indels before the first symbol. The immortal link is never
    inactive to ensure a sequence can always generate new symbols. Immortal
    links are not officially supported in this implementation, but they can be
    easily simulated by including an additional symbol at the beginning with a
    non-zero insertion rate and a deletion rate of 0. As a result of how the
    rate array is initialized, the symbol of the immortal link must have an
    associated rate in the rate matrix. However, to keep this rate constant,
    it is recommended to set the substitution rate to 0 as well.

    Parameters
    ----------
    seq: ndarray
        One-dimensional array where symbols are stored as numerical indices.
        The indices must match the order given in jump_matrices and sym_dists.
    rate_coefficients: ndarray
        Array with shape (3, len(seq)). The rows correspond to substitution,
        insertion, and deletion rate coefficients, respectively. The columns
        correspond to the index in seq.
    activities: ndarray
        One-dimensional array with boolean values indicating if the symbol is
        active. Deleted symbols are inactive.
    residue_ids: ndarray
        One-dimensional array with unique integer identifier for each symbol in
        seq.
    partition_ids: ndarray
        One-dimensional array with the partition integer identifier for each
        symbol in seq.
    rate_matrices: dict of ndarrays
        Dict keyed by partition_id where values are normalized rate matrices.
    sym_dists: dict of nd arrays
        Dict keyed by partition_id where values are symbol distributions.
    insertion_dists: dict of rv_discrete
        Dict keyed by partition_id where values are rv_discrete for generating
        random insertion lengths. Values must support a rvs method.
    deletion_dists: dict of rv_discrete
        Dict keyed by partition_id where values are rv_discrete for generating
        random deletion lengths. Values must support a rvs method.
    rng: numpy Generator instance
        Generator provides source of randomness for all operations.
    """
    def __init__(self, seq, rate_coefficients, activities, residue_ids, partition_ids,
                 rate_matrices, sym_dists, insertion_dists, deletion_dists, rng=None):
        self.seq = seq
        self.rate_coefficients = rate_coefficients
        self.activities = activities
        self.residue_ids = residue_ids
        self.partition_ids = partition_ids
        self.rate_matrices = rate_matrices
        self.sym_dists = sym_dists
        self.insertion_dists = insertion_dists
        self.deletion_dists = deletion_dists
        self.rng = rng

        # Calculate rates from rate matrices and rate coefficients
        rates = np.empty(len(seq))
        for j, (idx, partition_id) in enumerate(zip(self.seq, self.partition_ids)):
            rate = 0 if idx == -1 else -self.rate_matrices[partition_id][idx, idx]  # Check for "out-of-alphabet" symbols
            rates[j] = rate
        self.rates = rates * self.rate_coefficients

        # Calculate jump matrices from rate matrices
        jump_matrices = {}
        for partition_id, rate_matrix in rate_matrices.items():
            jump_matrix = np.copy(rate_matrix)
            np.fill_diagonal(jump_matrix, 0)
            jump_matrix = jump_matrix / np.expand_dims(jump_matrix.sum(axis=1), -1)  # Normalize rows to obtain jump matrix
            jump_matrices[partition_id] = jump_matrix
        self.jump_matrices = jump_matrices

    def __deepcopy__(self, memodict={}):
        seq = np.copy(self.seq)
        rate_coefficients = np.copy(self.rate_coefficients)
        activities = np.copy(self.activities)
        residue_ids = np.copy(self.residue_ids)
        partition_ids = np.copy(self.partition_ids)
        return SeqEvolver(seq, rate_coefficients, activities, residue_ids, partition_ids,
                          self.rate_matrices, self.sym_dists, self.insertion_dists, self.deletion_dists, self.rng)

    def mutate(self, residue_index):
        """Mutate the sequence."""
        event_ids = np.arange(3*len(self.seq))
        active_rates = (self.rates * self.activities).flatten()
        p = active_rates / active_rates.sum()
        event_id = self.rng.choice(event_ids, p=p)
        i, j = event_id // len(self.seq), event_id % len(self.seq)
        if i == 0:
            return self.substitute(j, residue_index)
        elif i == 1:
            return self.insert(j, residue_index)
        elif i == 2:
            return self.delete(j, residue_index)

    def substitute(self, j, residue_index):
        """Substitute residue at index j."""
        partition_id = self.partition_ids[j]
        jump_dist = self.jump_matrices[partition_id][self.seq[j]]
        idx = self.rng.choice(np.arange(len(jump_dist)), p=jump_dist)
        rate = -self.rate_matrices[partition_id][idx, idx]

        self.seq[j] = idx
        self.rates[:, j] = rate * self.rate_coefficients[:, j]

        return residue_index

    def insert(self, j, residue_index):
        """Insert randomly generated residues at an index j."""
        partition_id = self.partition_ids[j]
        sym_dist = self.sym_dists[partition_id]
        length = self.insertion_dists[partition_id].rvs(random_state=self.rng)

        # Make insertion arrays
        # (rate_coefficients and rates are transposed since it makes the insertion syntax simpler)
        seq = self.rng.choice(np.arange(len(sym_dist)), size=length, p=sym_dist)
        rate_coefficients = np.full((length, 3), self.rate_coefficients[:, j])
        activities = np.full(length, True)
        residue_ids = np.arange(residue_index, residue_index+length)
        partition_ids = np.full(length, partition_id)
        rates = np.expand_dims([-self.rate_matrices[partition_id][idx, idx] for idx in seq], -1)

        # Insert insertion into arrays
        self.seq = np.insert(self.seq, j+1, seq)
        self.rate_coefficients = np.insert(self.rate_coefficients, j+1, rate_coefficients, axis=1)  # Insert matches first axis of inserted array to one given by axis keyword
        self.activities = np.insert(self.activities, j+1, activities)
        self.residue_ids = np.insert(self.residue_ids, j+1, residue_ids)
        self.partition_ids = np.insert(self.partition_ids, j+1, partition_ids)
        self.rates = np.insert(self.rates, j+1, rates * rate_coefficients, axis=1)

        return residue_index + length

    def delete(self, j, residue_index):
        """Delete residues at an index j."""
        partition_id = self.partition_ids[j]
        length = self.deletion_dists[partition_id].rvs(random_state=self.rng)

        self.activities[j:j+length] = False

        return residue_index


def get_conditional(tree, matrix, inplace=False):
    """Return conditional probabilities of tree given tips and node state."""
    if not inplace:
        tree = tree.copy()  # Make copy so computations do not change original tree

    for node in tree.postorder():
        if node.is_tip():
            node.s = np.zeros(node.value.shape[1])
            node.conditional = node.value
        else:
            ss, ps = [], []
            for child in node.children:
                s, conditional = child.s, child.conditional
                m = linalg.expm(matrix * child.length)
                p = np.matmul(m, conditional)

                ss.append(s)
                ps.append(p)

            conditional = np.product(np.stack(ps), axis=0)
            s = conditional.sum(axis=0)
            node.conditional = conditional / s  # Normalize to 1 to prevent underflow
            node.s = log(s) + np.sum(np.stack(ss), axis=0)  # Pass forward scaling constant in log space

    return tree.s, tree.conditional


def simulate_tree(tree, evoseq, rng):
    tree = tree.copy()  # Make copy so computations do not change original tree
    tree.evoseq, tree.t = evoseq, 0
    residue_index = max(evoseq.residue_ids) + 1

    evoseqs = []
    for node in tree.traverse():
        evoseq, t = node.evoseq, node.t
        while t < node.length:
            residue_index = evoseq.mutate(residue_index)
            scale = 1/(evoseq.rates * evoseq.activities).sum()
            t += stats.expon.rvs(scale=scale, random_state=rng)
        if node.children:
            if rng.random() > 0.5:
                child1, child2 = node.children
            else:
                child2, child1 = node.children
            evoseq1, evoseq2 = evoseq, deepcopy(evoseq)  # One sequence is the original, the other is copied
            child1.evoseq, child1.t = evoseq1, t - node.length
            child2.evoseq, child2.t = evoseq2, 0
        else:
            evoseqs.append((node.name, evoseq))

    return residue_index, evoseqs
