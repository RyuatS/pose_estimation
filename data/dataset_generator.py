# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================
"""
Wrapper for providing human pose estimation data.

The PoseEstimationDataset class provides both images and annotations (pose
heatmap) for Tensorflow.
"""

import tensorflow as tf
import os
from tensorflow.contrib.distributions import MultivariateNormalDiag
import numpy as np
import math

class Dataset(object):
    """Respresents input dataset for pose estimation models."""

    def __init__(self,
                 tfrecord_dir,
                 data_type,
                 batch_size,
                 img_size,
                 num_readers=1,
                 is_training=False,
                 should_shuffle=False,
                 should_repeat=False):
        """
        Initializes the dataset.

        Args:
            tfrecord_dir: The directory containing the tfrecord.
            data_type: train2017 or val2017.
            batch_size: Batch size.
            img_size: Use this to resize. [height, width]
            num_readers: Number of readers for data provider.
            is_training: Boolean, if dataset is for training or not.
            should_shuffle: Boolean, if should shuffle the input data.
            should_repeat: Boolean, if should repeat the input data.

        Raises:
            ValueError: Dataset name and split name are not supported.
        """

        if not os.path.exists(tfrecord_dir):
            raise ValueError('The tfrecord directory is not exists')

        self.tfrecord_dir = tfrecord_dir
        self.data_type = data_type
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_readers = num_readers
        self.is_training = is_training
        self.should_shuffle = should_shuffle
        self.should_repeat = should_repeat


    def _parse_function(self, example_proto):
        """
        Function to parse the example proto.

        Args:
            example_proto: Proto in the format of tf.Example.

        Returns:
            A dictionary with parsed image, height, width, channel and heatmap.

        Raises:
            ValueError: Label is of wrong shape.
        """

        features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/height':
                tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/width':
                tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/channels':
                tf.FixedLenFeature((), tf.int64, default_value=0),
            # 'image/key_x_list':
            #     tf.FixedLenFeature((17), tf.int64, default_value=[0 for i in range(17)]),
            # 'image/key_y_list':
            #     tf.FixedLenFeature((17), tf.int64, default_value=[0 for i in range(17)]),
            # 'image/key_v_list':
            #     tf.FixedLenFeature((17), tf.int64, default_value=[0 for i in range(17)])
            'image/heatmap':
                tf.FixedLenFeature((256*192*17), tf.float32, default_value=np.zeros(256*192*17)),
            'image/heatmap/channels':
                tf.FixedLenFeature((), tf.int64, default_value=0)

        }

        parsed_features = tf.parse_single_example(example_proto, features)
        width = parsed_features['image/width']
        height = parsed_features['image/height']
        channels = parsed_features['image/channels']

        image = tf.reshape(tf.decode_raw(parsed_features['image/encoded'], tf.uint8),
                           [height, width, 3])
        image = tf.image.resize_images(image, self.img_size)
        image = tf.cast(image, tf.uint8)

        heatmaps = tf.reshape(parsed_features['image/heatmap'], [256, 192, 17])
        sample = {
            'image': image,
            'height': height,
            'width': width,
            # 'key_x': parsed_features['image/key_x_list'],
            # 'key_v': parsed_features['image/key_v_list'],
            # 'key_y': parsed_features['image/key_y_list'],
            'heatmaps': heatmaps
        }


        return sample


    def get_one_shot_iterator(self):
        """
        Gets an iterator that iterates across the dataset once.

        Returns:
            An iterator of type tf.data.Iterator.
        """
        files = self._get_all_files()

        dataset = (
            tf.data.TFRecordDataset(files, num_parallel_reads=self.num_readers)
            .map(self._parse_function, num_parallel_calls=self.num_readers)
        )

        if self.should_shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.should_repeat:
            dataset = dataset.repeat() # Repeat forever for training.
        else:
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
        return dataset.make_one_shot_iterator()


    def _get_all_files(self):
        """
        Gets all the files to read data from.

        Returns:
            A list of input files.
        """
        file_pattern = os.path.join(self.tfrecord_dir,
                                    '%s-*' % self.data_type)

        return tf.gfile.Glob(file_pattern)


if __name__ == '__main__':
    """
    For testing this class.

    """
    import matplotlib.pyplot as plt
    import cv2
    tfrecord_dir = os.path.join('cocodevkit', 'tfrecord')
    data_type = 'val2017'
    batch_size = 32
    img_size = (256, 192)
    dataset = Dataset(tfrecord_dir, data_type, batch_size, img_size)

    files = dataset._get_all_files()

    print('tfrecord files in {}'.format(tfrecord_dir))
    for file in files:
        print(file)

    iterator = dataset.get_one_shot_iterator()

    sample = iterator.get_next()
    with tf.Session() as sess:
        result = sess.run(sample)

    batch_heat = result['heatmaps']
    for i, heats in enumerate(batch_heat):
        iter_num = math.ceil(heats.shape[2] / 3)
        img = result['image'][i]
        copy_img = np.copy(img)
        # for i in range(iter_num):
        #     heat = heats[..., i*3:i*3+3]
        #     plt.subplot(2, 3, i+1)
        #     heat = heat * 255 / np.max(heat)
        #     if heat.shape[2] < 3:
        #         dim = 3 - heat.shape[2]
        #         zero_dim_padding = np.zeros((heat.shape[0], heat.shape[1], dim))
        #         heat = np.dstack((heat, zero_dim_padding))
        #     plt.imshow(heat)
        for i in range(heats.shape[2]):
            heat = heats[..., i]
            heat = heat * 255 / np.max(heat) + 0.001
            heat = np.dstack([heat, heat, heat]).astype(np.uint8)
            copy_img = cv2.addWeighted(copy_img, 1, heat, 0.5, 1.0)
        plt.imshow(copy_img)
        plt.show()
