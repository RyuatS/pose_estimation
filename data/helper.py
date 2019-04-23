# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

"""
Contains common utility functions and classes for building dataset.


"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


def create_heatmap(img, keypoint, sigma=1.0):
    """
    create heatmap.

    Args:
        keypoint: tuple or list. keypoint location (x, y).
        sigma: 2d-gauss distribution sigma.
            　　ヒートマップは、キーポイントがx, yが独立に生起すると仮定されている
    Return:
        heatmap
    """
    x = img.shape[1]
    y = img.shape[0]

    X = np.arange(0, x, 1)
    Y = np.arange(0, y, 1)

    # create 2d-map
    X, Y = np.meshgrid(X, Y)

    # define mean-vector and coeefficient matrix.
    mu = np.array(keypoint)
    Sigma = np.array([[sigma, 0],
                      [0, sigma]])

    # 2dガウス分布の係数を計算する。
    coe = (2*math.pi)**2
    coe *= np.linalg.det(Sigma)
    coe = math.sqrt(coe)

    # inverse matrix
    Sigma_inv = np.linalg.inv(Sigma)

    def f(x, y):
        x_c = np.array([x, y]) - mu
        return np.exp( - x_c.dot(Sigma_inv).dot(x_c[np.newaxis, :].T) / 2.0) / (coe)

    heatmap = np.vectorize(f)(X, Y)

    return heatmap
