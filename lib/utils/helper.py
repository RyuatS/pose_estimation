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
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalDiag


def create_heatmap_numpy(heatmap_shape, keypoint, sigma=1.0, is_norm=True):
    """
    create heatmap.

    Args:
        heatmap_shape: heatmap shape that you hope. (height, width)
        keypoint: tuple or list. keypoint location (x, y).
        sigma: 2d-gauss distribution sigma.
            　　ヒートマップは、キーポイントがx, yが独立に生起すると仮定されている
        is_norm: whether you normalize heatmap or not.
    Return:
        heatmap
    """
    x = heatmap_shape[1]
    y = heatmap_shape[0]

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
        ex = np.exp( - x_c.dot(Sigma_inv).dot(x_c[np.newaxis, :].T) / 2.0)

        if is_norm:
            return ex / coe
        else:
            return ex


    heatmap = np.vectorize(f)(X, Y)

    return heatmap

sess = tf.Session()
def create_heatmap(img, keypoint, sigma=1.0):
    """
    create heatmap.

    Args:
        img: image.
        keypoint: tuple or list. keypoint location(x, y).
        sigma: 2d-gauss distribution sigma.
    Return:
        heatmap
    """
    keypoint = np.array(keypoint)
    if type(sigma) == type(int(1)):
        sigma = float(sigma)
    mvn = MultivariateNormalDiag(
        loc=keypoint[::-1].astype(np.float32),
        scale_diag=[sigma, sigma])

    img_pixel_location = np.array([[[i, j] for j in range(img.shape[1])] for i in range(img.shape[0])])
    img_pixel_location = img_pixel_location.reshape(img.shape[1]*img.shape[0], 2)

    keypoint_probability = mvn.prob(img_pixel_location)
    result = sess.run(keypoint_probability)

    result = result.reshape((img.shape[0], img.shape[1]))
    return result
