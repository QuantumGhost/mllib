# coding=utf-8

from __future__ import (
    division, print_function,
    unicode_literals, absolute_import
)
import numpy as np
from numpy import linalg as LA
from . import exceptions


def normalize(m):
    """
    If m is a vector, normalize it as m = m - mean(m)
    If m is a matrix, normalize each column of this matrix.
    """
    shape = np.shape(m)
    if len(shape) == 1 or len(shape) == 2:
        # m is a vector or a matrix
        mu = np.mean(m, axis=0)
        return m - mu, mu
    else:
        raise exceptions.InvalidArgument("Neither a vector nor a matrix.")


def scale(m):
    """
    If m is a vector, scale it as m = m / std(m)
    If m is a matrix, scale each column of this matrix.
    """
    shape = np.shape(m)
    if len(shape) == 1 or len(shape) == 2:
        # m is a vector or a matrix
        sigma = np.std(m, axis=0, ddof=1)
        return m / sigma, sigma
    else:
        raise exceptions.InvalidArgument("Neither a vector nor a matrix.")


def sigmoid(x):
    return 1 / (1 + np.power(np.e, -x))


def normalize_and_scale(m):
    m, mu = normalize(m)
    m, sigma = scale(m)
    return m, mu, sigma
