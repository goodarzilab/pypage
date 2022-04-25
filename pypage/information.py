"""Implementation of Information Utils
"""

import math
import numpy as np
import numba as nb
from typing import Optional


@nb.jit(
    cache=True,
    nopython=True,
    nogil=True)
def empirical_entropy(
        array: np.ndarray,
        total_number: int,
        base: Optional[int] = None) -> float:
    """Calculates the empirical entropy of a vector

    inputs:
        array: np.ndarray
            a 1D vector of counts
        total_number: int
            the sum to normalize to
        base: int
            the base of the logarithm (default = natural)

    outputs:
        entropy: float
            the calculated entropy of the array
    """
    probability = np.divide(array, total_number)
    entropy = 0.
    base = math.e if not base else base

    for i in probability:
        if i == 0:
            continue
        entropy -= i * math.log(i) / math.log(base)

    return entropy


@nb.jit(
    cache=True,
    nogil=True,
    nopython=True)
def mutual_information(
        contingency: np.ndarray) -> float:
    """Calculates mutual information from contingency table. 
    
    Calculated using the form:
        I(X; Y) = H(X) + H(Y) - H(X, Y)

    inputs:
        contingency: np.ndarray
            2xNe array where Ne is the number of expression bins.
    outputs:
        information: float
            The calculated mutual information
    """
    total = contingency.sum()
    cx = contingency.sum(axis=1)
    cy = contingency.sum(axis=0)
    
    Hx = empirical_entropy(cx, total, base=2)
    Hy = empirical_entropy(cy, total, base=2)
    Hxy = empirical_entropy(contingency.flatten(), total, base=2)

    return (Hx + Hy - Hxy)
