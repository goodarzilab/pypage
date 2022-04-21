"""Implementation of Information Utils
"""

import numpy as np
import numba as nb


@nb.jit(
    nopython=True, 
    fastmath=True)
def mutual_information(
        contingency: np.ndarray,
        eps: float = 1e-12) -> float:
    """Calculates mutual information from contingency table

    inputs:
        contingency: np.ndarray
            2xNe array where Ne is the number of expression bins.
        eps: float
            a small float to add to zeros to avoid divide by zeros

    outputs:
        information: float
            the sum of the calculated mutual information matrix
    """

    N = contingency.sum()
    p_ij = ((eps + contingency) / N)
    p_i = p_ij.sum(axis=1)
    p_j = p_ij.sum(axis=0)
    
    information = 0.
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if contingency[i][j] == 0:
                pass
            pxy = p_ij[i][j]
            px = p_i[i]
            py = p_j[j]
            information += pxy * np.log( pxy / (px * py) ) / np.log(2.0)
            
    return information
