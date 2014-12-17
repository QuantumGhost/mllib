# coding=utf-8

from __future__ import (
    division, print_function,
    unicode_literals, absolute_import
)
import numpy as np
from numpy import linalg as LA
from .optimize import gradient_descent
from .utils import sigmoid


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
        j_val = d.T.dot(d) / (2.0 * m)
        # grt rid of the parameter for bias term

        # regularization part of objective function
        j_val += LA.norm(theta[1:]) / (2.0 * m) * l

        # gradient
        grad = 1 / m * d.T.dot(Xext).T
        # regularization part of objective function
        grad += l / m * theta
        grad[0] -= l / m * theta[0]    # Don't regularize theta for bias term
        return j_val, grad

    if initial_theta is None:
        initial_theta = np.zeros(n + 1).T
    (val, theta, history) = gradient_descent(
        obj_fun, initial_theta, alpha, num_iters)
    return val, theta, history


def logistic_reg(X, y, alpha, l, initial_theta=None, num_iters=1000):
    """
    Train a logistic regression with regularization.

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
        hx = sigmoid(Xext.dot(theta))    # H_theta(Xext) for all samples
        j_val = -1 / m * (
            y.T.dot(np.log(hx)) + (1 - y).dot(np.log(1 - hx)))
        # Regularization
        j_val += l / (2 * m) * LA.norm(theta[1:])

        grad = 1 / m * (hx - y).T.dot(Xext)
        # Regularization
        grad += l / m * theta
        grad[0] -= l / m * theta[0]
        return j_val, grad

    if initial_theta is None:
        initial_theta = np.zeros(n + 1).T
    (val, theta, history) = gradient_descent(
        obj_fun, initial_theta, alpha, num_iters)
    return val, theta, history
