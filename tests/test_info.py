"""Test information calculations
"""

import pytest
import numpy as np
from pypage.information import (
        entropy,
        conditional_entropy,
        joint_entropy,
        joint_entropy_3d,
        mutual_information,
        conditional_mutual_information)

EPSILON = 1e-9


def paired_arrays(
        size=1e4, 
        overlap=0.9, 
        bin_sizes=[0.05, 0.85, 0.05]):
    """janky solution to generating paired pseudo-bins and pseudo-ontologies.
    """

    size = int(size)
    bin_sizes = np.array(bin_sizes)
    bin_frac = bin_sizes / bin_sizes.sum()
    bin_sizes = (size * bin_sizes).astype(int)
    
    # create bins
    bins = np.random.choice(bin_sizes.size, p=bin_frac, size=size)
    
    # create ontology with overlap to first and last bin
    sig_max = np.max(bins)
    sig_min = np.min(bins)
    sig_choices = np.isin(bins, [sig_max, sig_min])
    sig_idx = np.flatnonzero(sig_choices)
    sig_total = sig_choices.sum()
    ontology = np.zeros(size, dtype=np.int32)
    mask = np.random.choice(sig_idx, size=int(overlap * sig_total), replace=False)
    ontology[mask] = 1

    x_bins = bins.max() + 1
    y_bins = ontology.max() + 1
    
    return (
        bins.astype(np.int32), 
        ontology.astype(np.int32), 
        x_bins, 
        y_bins)


def test_conditional_entropy():
    """test conditions of conditional entropy
    
    H(Y|X) 
        = H(X,Y) - H(X)
        = H(X|Y) - H(X) + H(Y)
    """
    x, y, x_bins, y_bins = paired_arrays()

    h_x = entropy(x, x_bins)
    h_y = entropy(y, y_bins)
    h_xy = joint_entropy(x, y, x_bins, y_bins)
    h_yGx = conditional_entropy(y, x, y_bins, x_bins)
    h_xGy = conditional_entropy(x, y, x_bins, y_bins)

    exp1 = h_xy - h_x
    exp2 = h_xGy - h_x + h_y

    assert np.abs(h_yGx - exp1) < EPSILON
    assert np.abs(h_yGx - exp2) < EPSILON

def test_mutual_information():
    """
    test alternative expressions of mutual information
    I(X;Y) 
        = H(X) - H(X|Y)
        = H(Y) - H(Y|X)
        = H(X) + H(Y) - H(X,Y)
        = H(X,Y) - H(X|Y) - H(Y|X)
        = I(Y;X)
    """
    x, y, x_bins, y_bins = paired_arrays()
    i_xy = mutual_information(x, y, x_bins, y_bins)
    i_yx = mutual_information(y, x, y_bins, x_bins)
    
    h_x = entropy(x, x_bins)
    h_y = entropy(y, y_bins)

    h_xGy = conditional_entropy(x, y, x_bins, y_bins)
    h_yGx = conditional_entropy(y, x, y_bins, x_bins)
    h_xy = joint_entropy(x, y, x_bins, y_bins)

    exp1 = h_x - h_xGy
    exp2 = h_y - h_yGx
    exp3 = h_x + h_y - h_xy
    exp4 = h_xy - h_xGy - h_yGx

    assert np.abs(i_xy - exp1) < EPSILON
    assert np.abs(i_xy - exp2) < EPSILON
    assert np.abs(i_xy - exp3) < EPSILON
    assert np.abs(i_xy - exp4) < EPSILON
    assert np.abs(i_xy - i_yx) < EPSILON

def test_joint_entropy():
    """test conditions of joint entropy using conditional and mutual information
    H(X,Y)
        = H(X) + H(Y) - I(X;Y)
        = H(X|Y) + H(Y|X) + I(X;Y)
    """
    x, y, x_bins, y_bins = paired_arrays()
    i_xy = mutual_information(x, y, x_bins, y_bins)
    h_xy = joint_entropy(x, y, x_bins, y_bins)
    
    h_x = entropy(x, x_bins)
    h_y = entropy(y, y_bins)
    h_xGy = conditional_entropy(x, y, x_bins, y_bins)
    h_yGx = conditional_entropy(y, x, y_bins, x_bins)
    
    exp1 = h_x + h_y - i_xy
    exp2 = h_xGy + h_yGx + i_xy

    assert np.abs(h_xy - exp1) < EPSILON
    assert np.abs(h_xy - exp2) < EPSILON

def test_conditional_mutual_information():
    """
    test alternative expressions of conditional mutual information
    I(X;Y|Z)
        = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    """
    x, y, x_bins, y_bins = paired_arrays()
    z = y.copy()
    np.random.shuffle(z)
    z_bins = y_bins

    cmi = conditional_mutual_information(x, y, z, x_bins, y_bins, z_bins)

    h_xz = joint_entropy(x, z, x_bins, z_bins)
    h_yz = joint_entropy(y, z, y_bins, z_bins)
    h_xyz = joint_entropy_3d(x, y, z, x_bins, y_bins, z_bins)
    h_z = entropy(z, z_bins)

    exp1 = h_xz + h_yz - h_xyz - h_z

    assert np.abs(cmi - exp1) < EPSILON
