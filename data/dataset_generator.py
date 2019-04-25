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


class Dataset(object):
    """Respresents input dataset for pose estimation models."""

    def __init__(self,
                 tfrecord_dir,
                 data_type,
                 batch_size,
                 img_size,
                 is_training=False,
                 should_shuffle=False,
                 should_repeat=False):
        """
        Initializes the dataset.

        Args:
            tfrecord_dir: The directory containing the tfrecord.
            data_type: train2017 or val2017.
            batch_size: Batch size.
            img_size: Use this to resize.
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
                tf.FixedLenFeature((), tf.int64, default_value=''),
            'image/width':
                tf.FixedLenFeature((), tf.int64, default_value=''),
            'image/channels':
                tf.FixedLenFeature((), tf.int64, default_value=''),
            'image/heatmap':
                tf.FixedLenFeature((), tf.float32, default_value=''),
            'image/heatmap/channels':
                tf.FixedLenFeature((), tf.int64, default_value='')
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        # image = _decode_image(parsed_features['image/encoded'],
        #                       channels=parsed_features['image/channels'])


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
            .map(self._preprocess_image, num_parallel_calls=self.num_readers)
        )

        if self.should_shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.should_reqeat:
            dataset = dataset.repat() # Repeat forever for training.
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
    tfrecord_dir = os.path.join('cocodevkit', 'tfrecord')
    data_type = 'val2017'
    batch_size = 32
    img_size = (256, 256)
    dataset = Dataset(tfrecord_dir, data_type, batch_size, img_size)

    files = dataset._get_all_files()

    print('tfrecord files in {}'.format(tfrecord_dir))
    for file in files:
        print(file)