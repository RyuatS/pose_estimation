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
import sys
import os
import cv2
from lib.core.config import _KEYPOINTS_LABEL, SKELETON

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from core.config import R_MEAN, G_MEAN, B_MEAN


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


def visualize_heatmaps(image, target=None, predict=None, is_separate=False):
    """
    visualize input image, target heatmaps and heatmaps.

    Args:
        image       : input image. [height, width, channel].
        target      : target heatmaps. [height, width, the number of keypoints]. The number of keypoints should be 17.
        predict    : predict heatmaps. [height, widht, the number of keypoints]. The number of keypoints hould be 17.
        is_separate : whether heatmaps separate or not.
    """

    plot_rows, plot_cols = 3, 6
    plt.figure(figsize=(12, 10))
    if target is not None:
        num_keys = target.shape[2]
        if is_separate:
            plt.subplot(plot_rows, plot_cols, 1)
            plt.imshow(image)
            plt.title('input image')
            image = image.astype(np.float32)
            for index in range(num_keys):
                plt.subplot(plot_rows, plot_cols, index+2)
                plt.imshow(target[:, :, index], cmap='gray')
                plt.title(_KEYPOINTS_LABEL[index])
        else:
            image = image.astype(np.float32)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if np.max(gray) > 1:
                gray = gray / 255
            _2d_heatmap = np.max(target, axis=2)
            gray = (gray + _2d_heatmap) / 2

            plt.imshow(gray, cmap='gray')
        plt.show()

    if predict is not None:
        num_keys = predict.shape[2]
        if is_separate:
            plt.subplot(plot_rows, plot_cols, 1)
            plt.imshow(image)
            plt.title('input image')
            image = image.astype(np.float32)
            for index in range(num_keys):
                plt.subplot(plot_rows, plot_cols, index+2)
                plt.imshow(predict[:, :, index], cmap='gray')
                plt.title(_KEYPOINTS_LABEL[index])
        else:
            image = image.astype(np.float32)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if np.max(gray) > 1:
                gray /= 255
            _2d_heatmap = np.max(predict, axis=2)
            gray  = (gray + _2d_heatmap) / 2

            plt.imshow(gray, cmap='gray')
        plt.show()

    if (predict is None) and (target is None):
        print('There is nothing to plot')


def visualize_keypoints(image, predict_heatmap=None):
    """
    visualize keypoints using line.

    Args:
        image: input image. [height, width, 3]
        predict_heatmap : heatmaps which is outputed from model.
    """

    keypoint_loc_list = []
    num_channel = predict_heatmap.shape[2]

    for i in range(num_channel):
        h = predict_heatmap[:, :, i]
        h_flatten = sorted(h.ravel())

        threshold = h_flatten[-20]
        h[h < threshold] = 0
        h[h > threshold] = 1

        width_sum = np.sum(h, axis=0).flatten()
        height_sum = np.sum(h, axis=1).flatten()
        loc_x = 0
        loc_y = 0
        count_x = 0
        count_y = 0
        for x in range(len(width_sum)):
            if width_sum[x] >= 1:
                loc_x += x
                count_x += 1

        for y in range(len(height_sum)):
            if height_sum[y] >= 1:
                loc_y += y
                count_y += 1

        key_x = loc_x // count_x
        key_y = loc_y // count_y
        keypoint_loc_list.append([key_x, key_y])

    for CON in SKELETON:
        index1, index2 = CON
        key1 = keypoint_loc_list[index1]
        key2 = keypoint_loc_list[index2]
        cv2.line(image, tuple(key1), tuple(key2), (255, 0, 0), 2)

    for i in range(len(keypoint_loc_list)):
        x, y = keypoint_loc_list[i]
        plt.scatter(x, y, label=_KEYPOINTS_LABEL[i])
    plt.imshow(image.astype(np.uint8))
    plt.legend()
    plt.show()
