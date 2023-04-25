"""Functions for calculating GO term enrichment tests."""

import numpy as np
import scipy.stats as stats


def hypergeom_test(k, M, n, N):
    """Return the p-value for a hypergeometric test.

    Parameters
    ----------
    k: int
        Number of objects selected from the collection with the property of interest.
    M: int
        Number of objects in the collection.
    n: int
        Number of objects in the collection with the property of interest.
    N: int
        Number of objects selected from the collection.

    Returns
    -------
    pvalue: float
        One-tailed p-value for hypergeometric test.
    """
    kmax = min(n, N)
    ks = np.arange(k, kmax + 1)
    pmfs = stats.hypergeom.pmf(k=ks, M=M, n=n, N=N)
    pvalue = pmfs.sum()
    return pvalue
