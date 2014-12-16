# coding=utf-8

from __future__ import (
    division, print_function,
    unicode_literals, absolute_import
)

"""
Optimization algorithms
"""


def gradient_descent(func, initial_theta, alpha, num_iters=1000):
    """
    Using gradient descent algorithm to find the minimal (local or global)
    value of given function.

    :param func: A function which takes a parameter vector theta and \
        returns the value and gradient at given point theta.
    :type func: Vector(n) -> (float , Vector(n))
    :param initial_theta: The parameter vector of objective function
    :type initial_theta: Vector(n)
    :param alpha: The learning rate
    :type alpha: float
    :param num_iters: Number of iterations, default value is 1000.
    :type num_iters: Integral

    :returns: a tuple consists the minimum value of given function,
        the correspond theta and a list of training history.
    """
    if num_iters <= 0:
        raise ValueError("Iteration number should be greater than 0.")
    theta = initial_theta
    history = []
    for i in xrange(num_iters):
        val, grad = func(theta)
        history.append(val)
        theta -= alpha * grad
    return val, theta, history


def lbfgs(func, initial_theta, alpha, num_iters=1000):
    pass
