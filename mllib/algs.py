# coding=utf-8

from __future__ import (
    division, print_function,
    unicode_literals, absolute_import
)
import numpy as np
from numpy import linalg as LA
from .optimize import gradient_descent


def linreg(X, y, alpha, l, initial_theta=None, num_iters=1000):
    """
    Train a linear regression with regularization.

    :param X: Sample matrix.
    :type X: Matrix of m * n
    :param y:
    :type y: Vector of size m.
    :param alpha: Learning rate.
    :type alpha: float
    :param l: Regularization coefficient.
    :type l: float
    :param initial_theta: #TODO
    :type initial_theta: #TODO
    :param num_iters: Number of iterations, default value is 1000.
    :type num_iters: Integral
    """
    (m, n) = X.shape
    # Add the bias feature (all one's column)
    Xext = np.hstack((np.ones((m, 1)), X))

    def obj_fun(theta):
        """
        The objective function of linear regression. Theta is the parameter vector of
        this linear regression function.
        """
        d = Xext.dot(theta) - y
        # the value of objective function
        val = d.T.dot(d) / (2.0 * m)
        # grt rid of the parameter for bias term
        params = theta.copy()
        params[0] = 0
        # regularization part of objective function
        val += params.T.dot(params) / (2.0 * m) * l

        # gradient
        grad = 1 / m * d.T.dot(Xext).T
        # regularization part of objective function
        grad += l / m * params
        return val, grad

    if initial_theta is None:
        initial_theta = np.zeros(n + 1).T
    (val, theta, history) = gradient_descent(
        obj_fun, initial_theta, alpha, num_iters)
    return val, theta, history
