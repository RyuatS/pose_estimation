# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

# lib
import tensorflow as tf
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
import cv2
import numpy as np
from lib.core.config import BACKBONE_NAME_LIST

# user packages
from lib.models.hourglass import Hourglass
import lib.utils.helper as helper


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir',
                           './checkpoints',
                           'checkpoints directory')

tf.app.flags.DEFINE_string('image_path',
                           './data/demo/ski.png',
                           'image you want to predict keypoints')

tf.app.flags.DEFINE_boolean('is_separate',
                           False,
                           'whether you visualize heatmaps separately or not.')

tf.app.flags.DEFINE_boolean('is_save_image',
                            False,
                            'Do you save the keypoints estimation image?')


def main(argv):

    sess = tf.Session()

    input_size = (256, 192)
    image = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='input')

    model = Hourglass(is_use_bn=True, num_keypoints=17)
    logits, _ = model.build(image, 'Hourglass', is_training=False, visualize=True)

    logits = tf.nn.sigmoid(logits)

    load_image = plt.imread(FLAGS.image_path)
    load_image = cv2.resize(load_image, (192, 256))

    if np.max(load_image) <= 1:
        load_image = load_image * 255

    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    if checkpoint:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint.model_checkpoint_path)
        res = sess.run(logits, feed_dict={image: np.expand_dims(load_image, axis=0)})

        res = res[0]

        helper.visualize_heatmaps(load_image, predict=res, is_separate=FLAGS.is_separate)

        helper.visualize_keypoints(load_image, predict_heatmap=res, is_save=FLAGS.is_save_image)

    else:
        raise ValueError("'{}' does not exist".format(FLAGS.check_dir))


if __name__ == '__main__':
    tf.app.run()
