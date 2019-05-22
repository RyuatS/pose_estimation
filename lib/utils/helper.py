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
from lib.core.config import KEYPOINTS_LABEL, SKELETON

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


def create_heatmap_for_sigmoid(heatmap_shape, keypoint, radius=10):
    """
    クロスエントロピー誤差用のヒートマップを作成する。
    作り方は、キーポイントの位置から半径がradiusの円の内部の値を１にする。
    それ以外は、0にする。

    Args:
        heatmap_shape: heatmap shape that you hope. (height, width)
        keypoint: tuple or list. keypoint location (x, y).
        radius: 半径

    Return:
        heatmap
    """
    height, width = heatmap_shape[0], heatmap_shape[1]
    heatmap = np.zeros((height, width))
    key_x, key_y = keypoint

    for height_loc in range(height):
        for width_loc in range(width):
            distance_to_key = abs(key_x - width_loc) ** 2 + abs(key_y - height_loc) ** 2
            distance_to_key = np.sqrt(distance_to_key)

            if distance_to_key <= radius:
                heatmap[height_loc, width_loc] = 1

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
    if image.dtype == np.float32 and np.max(image) >= 1:
        image = image.astype(np.uint8)


    plot_rows, plot_cols = 3, 6
    plt.figure(figsize=(12, 10))
    if target is not None:
        if image.shape[:2] != target.shape[:2]:
            target = cv2.resize(target, (image.shape[1], image.shape[0]))

        num_keys = target.shape[2]
        if is_separate:
            plt.subplot(plot_rows, plot_cols, 1)
            plt.imshow(image)
            plt.title('input image')
            for index in range(num_keys):
                plt.subplot(plot_rows, plot_cols, index+2)
                plt.imshow(target[:, :, index], cmap='viridis')
                plt.title(KEYPOINTS_LABEL[index])
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
        if image.shape[:2] != predict.shape[:2]:
            predict = cv2.resize(predict, (image.shape[1], image.shape[0]))

        num_keys = predict.shape[2]
        if is_separate:
            plt.subplot(plot_rows, plot_cols, 1)
            plt.imshow(image)
            plt.title('input image')
            plt.colorbar()
            for index in range(num_keys):
                plt.subplot(plot_rows, plot_cols, index+2)
                plt.imshow(predict[:, :, index], cmap='viridis')
                plt.title(KEYPOINTS_LABEL[index])
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


def visualize_keypoints(image, predict_heatmap=None, is_save=False):
    """
    visualize keypoints using line.

    Args:
        image : input image. [height, width, 3]
        predict_heatmap : heatmaps which is outputed from model.
        is_save : is save?

    Return:
        image which is written to keypoints.
    """

    keypoint_loc_list = []
    num_channel = predict_heatmap.shape[2]

    image_height, image_width = image.shape[:2]
    heat_height, heat_width = predict_heatmap.shape[:2]

    HEATMAP_THRESHOLD = 0.5
    for i in range(num_channel):
        h = predict_heatmap[:, :, i]

        if np.max(h) <= HEATMAP_THRESHOLD:
            keypoint_loc_list.append(None)
            continue
        max_index = np.unravel_index(np.argmax(h, axis=None), h.shape)
        key_y, key_x  = max_index

        key_x = (key_x*image_width)/(heat_width)
        key_y = (key_y*image_height)/(heat_height)

        keypoint_loc_list.append([int(key_x), int(key_y)])
        continue
        # h_flatten = sorted(h.ravel())
        #
        # threshold = h_flatten[-20]
        # h[h < threshold] = 0
        # h[h > threshold] = 1
        #
        # if threshold <= 0:
        #     keypoint_loc_list.append(None)
        #     continue
        #
        # width_sum = np.sum(h, axis=0).flatten()
        # height_sum = np.sum(h, axis=1).flatten()
        # loc_x = 0
        # loc_y = 0
        # count_x = 0
        # count_y = 0
        # for x in range(len(width_sum)):
        #     if width_sum[x] >= 1:
        #         loc_x += x
        #         count_x += 1
        #
        # for y in range(len(height_sum)):
        #     if height_sum[y] >= 1:
        #         loc_y += y
        #         count_y += 1
        # key_x = loc_x // count_x
        # key_y = loc_y // count_y
        # keypoint_loc_list.append([key_x, key_y])

    for CON in SKELETON:
        index1, index2, color = CON
        key1 = keypoint_loc_list[index1]
        key2 = keypoint_loc_list[index2]
        if (key1 is not None) and (key2 is not None):
            cv2.line(image, tuple(key1), tuple(key2), color, 2)


    for i in range(len(keypoint_loc_list)):
        if keypoint_loc_list[i] is not None:
            x, y = keypoint_loc_list[i]
            plt.scatter(x, y, label=KEYPOINTS_LABEL[i])


    plt.imshow(image.astype(np.uint8))
    plt.legend()
    if is_save:
        cv2.imwrite('vis_keypoint_result.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    plt.show()

    return image.astype(np.uint8)


def create_saver_and_restore(session, checkpoints_dir, backbone_name=None):
    """
    create saver and restore.

    Args:
        session        : tensorflow session
        checkpoints_dir: checkpoint directory.
        backbone_name  : backbone name. もしも、バックボーンを使用する場合に指定する。
                        　example) resnet_v1_101

    Return:
        saver, checkopint_path
    """
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
    if checkpoint:
        print('\n\n' + checkpoint.model_checkpoint_path)
        print('variables were restored from {}.'.format(checkpoint.model_checkpoint_path))
        saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        session.run(tf.global_variables_initializer())
        print('variables were initialized.')

        # load backbone weights
        if backbone_name is not None:
            backbone_vars = tf.contrib.framework.get_variables_to_restore(include=[backbone_name])
            backbone_saver = tf.train.Saver(var_list=backbone_vars)
            backbone_checkpoint_path = os.path.join('backbone_checkpoints',
                                                    '{}.ckpt'.format(backbone_name))
            backbone_saver.restore(session, backbone_checkpoint_path)
            print('{} weights were restored.'.format(backbone_name))

    # checkpoint_path
    checkpoint_path = os.path.join(checkpoints_dir, 'model.ckpt')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    return saver, checkpoint_path
