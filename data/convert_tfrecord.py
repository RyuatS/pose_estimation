# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2018-4-19
# Description:
#
# ===============================================

"""
This script contains utility functions and classes to converts dataset to
TFRecord file format following Example protos.

The Example proto contains the following fields:
  image/encoded: encoded image content.
  image/height: image height.
  image/width: image width.
  image/channels: image channles
  image/key_x_list: keypoint locations x list.
  image/key_y_list: keypoint locations y list.
  image/key_v_list: keypoint visible list.
"""

import tensorflow as tf
import sys
import collections
import numpy as np


def _int64_list_feature(values):
    """
    Returns a TF-Feature of int64_list.

    Args:
        A scalar or list of values.
    Returns:
        A TF-Feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
    """
    Returns a TF-Feature of bytes.

    Args:
        values: byte list.

    Returns:
        A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_list_feature(values):
    """
    Returns a TF-Feature of float.

    Args:
        values: float list.
    Returns:
        A TF-Feature.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_heatmap_to_tfexample(image_data, heatmap):
    """
    Converts one image/heatmap pair to tf example.

    Args:
        image_data: string of image data.
        heatmap: string of heatmap.

    Returns:
        tf example of one image/heatmap pair.
    """
    if image_data.shape[0] != heatmap.shape[0]:
        raise ValueError('image and heatmap height is not equal.')
    if image_data.shape[1] != heatmap.shape[1]:
        raise ValueError('image and heatmap widht is not equal.')
    if len(image_data.shape) != 3:
        raise ValueError('image_data has not 3-dimentions => image_data.shape: {}'.format(image_data.shape))

    height, width, channels = image_data.shape
    heatmap_channels = heatmap.shape[2]
    flatten_img = image_data.ravel().tostring()
    flatten_heatmap = heatmap.astype(np.float32).ravel()

    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_list_feature([flatten_img]),
        'image/height': _int64_list_feature([height]),
        'image/width': _int64_list_feature([width]),
        'image/channels': _int64_list_feature([channels]),
        'image/heatmap': _float_list_feature(flatten_heatmap),
        'image/heatmap/channels': _int64_list_feature([heatmap_channels])
    }))


def image_keypoint_to_tfexample(image_data, key_x_list, key_y_list, key_v_list):
    """
    Converts one image/keypoints pair to tf example.

    Args:
        image_data: image. numpy array.
        key_x_list: list. keypoint x location list. [x1, x2, ...].
                    x1 is correspond to y1. In other words, first keypoint location is (x1, y1).
        key_y_list: list. keypoint y location list. [y1, y2, ...].
        key_v_list: list. keypoint visible list. [v1, v2, ...].
    Return:
        tf example of one image/keypoints pair.
    """

    if len(image_data.shape) != 3:
        raise ValueError('image_data has not 3-dimentions => image_data.shape: {}'.format(image_data.shape))

    height, width, channels = image_data.shape
    flatten_img = image_data.ravel().tostring()

    key_x_list = np.array(key_x_list, dtype=np.int64)
    key_y_list = np.array(key_y_list, dtype=np.int64)
    key_v_list = np.array(key_v_list, dtype=np.int64)

    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_list_feature([flatten_img]),
        'image/height': _int64_list_feature([height]),
        'image/width': _int64_list_feature([width]),
        'image/channels': _int64_list_feature([channels]),
        'image/key_x_list':  _int64_list_feature(key_x_list),
        'image/key_y_list': _int64_list_feature(key_y_list),
        'image/key_v_list': _int64_list_feature(key_v_list)
    }))
