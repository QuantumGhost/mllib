# coding=utf-8

import numpy as np
from numpy import linalg as LA
import exceptions


def normalize(m):
    """
    If m is a vector, normalize it as m = m - mean(m)
    If m is a matrix, normalize each column of this matrix.
    """
    shape = np.shape(m)
    if len(shape) == 1:
        # m is a vector
        return m - np.mean(m)
    elif len(shape) == 2:
        # m is a matrix
        return m - np.mean(axis=0)
    else:
        raise exceptions.InvalidArgument("Neither a vector nor a matrix.")


def scaling(m):
    """
    If m is a vector, scale it as m = m / std(m)
    If m is a matrix, scale each column of this matrix.
    """
    shape = np.shape(m)
    if len(shape) == 1:
        # m is a vector
        return m / np.std(m)
    elif len(shape) == 2:
        # m is a matrix
        return m / np.std(axis=0)
    else:
        raise exceptions.InvalidArgument("Neither a vector nor a matrix.")
