# coding=utf-8

import numpy as np
from numpy import linalg as LA
from optimize import gradient_descent


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
    # Add the bias feature (all one's column)
    # X =
    (m, n) = X.shape

    def obj_fun(theta):
        """
        The objective function of linear regression. Theta is the parameter vector of
        this linear regression function.
        """
        d = X.dot(theta) - y
        val = d.T.dot(d) / (2.0 * m)   # the value of function
        val += theta.T.dot(theta) / (2.0 * m) * l
        grad = None
        return val, grad

    if initial_theta is None:
        initial_theta = np.zeros(n).T
    (val, theta, history) = gradient_descent(obj_fun, initial_theta, alpha, num_iters)
