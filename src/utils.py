"""Functions for common operations in this project."""

import re

import numpy as np


def read_fasta(path):
    """Read FASTA file at path and return list of headers and sequences.

    Parameters
    ----------
    path: str
        Path to FASTA file

    Returns
    -------
    fasta: list of tuples of (header, seq)
        The first element is the header line with the >, and the second
        element is the corresponding sequence.
    """
    with open(path) as file:
        line = file.readline()
        while line:
            if line.startswith('>'):
                header = line.rstrip()
                line = file.readline()

            seqlines = []
            while line and not line.startswith('>'):
                seqlines.append(line.rstrip())
                line = file.readline()
            seq = ''.join(seqlines)
            yield header, seq


def read_iqtree(path, norm=False):
    """Read IQ-TREE file at path and return model parameters.

    Parameters
    ----------
    path: str
        Path to IQ-TREE file
    norm: bool
        If True, exchangeability matrix is normalized so average rate is 1.

    Returns
    -------
    record: dict
        Contains model parameters in the following keys:
            ematrix: Matrix of exchangeabilities
            freqs: Vector of frequencies
    """
    alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    with open(path) as file:
        # Move to exchangeability matrix and load
        line = file.readline()
        while not line.startswith('Substitution parameters'):
            line = file.readline()
        for _ in range(2):
            line = file.readline()

        rows = []
        while line != '\n':
            rows.append([float(value) for value in line.split()])
            line = file.readline()

        # Move to equilibrium frequencies and load
        for _ in range(3):
            line = file.readline()

        syms, freqs = [], []
        while line != '\n':
            match = re.search(r'pi\(([A-Z])\) = (0.[0-9]+)', line)
            syms.append(match.group(1))
            freqs.append(float(match.group(2)))
            line = file.readline()
        freqs = np.array(freqs)

        if syms != alphabet:
            raise RuntimeError('Symbols in matrix are not in expected order.')

        # Make matrix and scale
        ematrix = np.zeros((len(syms), len(syms)))
        for i, row in enumerate(rows[:-1]):
            for j, value in enumerate(row):
                ematrix[i + 1, j] = value
                ematrix[j, i + 1] = value

        if norm:
            rate = (freqs * (freqs * ematrix).sum(axis=1)).sum()
            ematrix = ematrix / rate
        record = {'ematrix': ematrix, 'freqs': freqs}

        return record


def read_paml(path, norm=False):
    """Read PAML file at path and return exchangeability matrix and frequencies.

    PAML files are formatted as the symmetric part of an amino acid
    substitution matrix (exchangeabilities) followed by the equilibrium amino
    acid frequency vector. The first 19 lines correspond to the lower triangle
    of the matrix and the 20th line is the frequency vector. Empty or
    whitespace lines are ignored when parsing the file.

    The amino acids are given in the following order:
    A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V

    A rate matrix Q is calculated from these parameters with the following
    formula
    Q_ij = s_ij * pi_j, i != j
         = sum_{i != j} Q_ij, i == j
    where s_ij is the ij-th entry in the exchangeability matrix and pi_j is the
    j-th entry in the frequency vector. The off diagonal entries represent the
    rate of replacement from amino acid i to j. The diagonal entries represent
    the total rate replacement of amino acid i.

    The average rate of replacement q of a matrix Q is given by
    q = sum_{i} pi_i * Q_ii
    where pi_i and Q_ii are defined as above. Rate matrices are conventionally
    scaled to make their average rate 1, which is accomplished by dividing the
    exchangeability or rate matrix by q.

    Parameters
    ----------
    path: str
        Path to PAML file
    norm: bool
        If True, exchangeabilities are normalized so average rate is 1.

    Returns
    -------
    matrix: ndarray
        Matrix of exchangeabilities
    freqs: ndarray
        Vector of frequencies
    """
    alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    with open(path) as file:
        lines = []
        for line in file:
            line = line.rstrip()
            if line:
                lines.append(line)
        if len(lines) < len(alphabet):
            raise RuntimeError(f'File has fewer than {len(alphabet)} non-empty or non-whitespace lines.')

    # Parse exchangeability matrix
    matrix = np.zeros((len(alphabet), len(alphabet)))
    for i in range(len(alphabet)-1):
        line = lines[i]
        for j, value in enumerate(line.split()):
            matrix[i+1, j] = float(value)
            matrix[j, i+1] = float(value)

    # Parse equilibrium frequencies
    line = lines[len(alphabet)-1]
    freqs = np.array([float(value) for value in line.split()])
    freqs = freqs / freqs.sum()  # Re-normalize to ensure sums to 1

    if norm:
        rate = (freqs * (freqs * matrix).sum(axis=1)).sum()
        matrix = matrix / rate  # Normalize average rate to 1

    return matrix, freqs
