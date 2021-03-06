# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

# lib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

# user packages
from lib.models.reshourglass import ResHourglass
from lib.core.config import BACKBONE_NAME_LIST
import lib.utils.helper as helper
from lib.models.hourglass import Hourglass
from lib.models.stacked_hourglass import StackedHourglass

tf.logging.set_verbosity(tf.logging.FATAL)

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

tf.app.flags.DEFINE_boolean('is_save',
                            False,
                            'Do you save the keypoints estimation image?')
tf.app.flags.DEFINE_boolean('is_measure',
                            False,
                            'Do you measure the fps about model.')

tf.app.flags.DEFINE_enum('model_type',
                           'hourglass',
                           ['reshourglass', 'hourglass', 'stacked'],
                           'model type which should be defined ./lib/models/')


def main(argv):

    sess = tf.Session()

    input_size = (256, 192)
    image = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='input')

    if FLAGS.model_type == 'hourglass':
        model = Hourglass(is_use_bn=True, num_keypoints=17)

        resize = (128, 96)

    elif FLAGS.model_type == 'reshourglass':
        model = ResHourglass(is_use_bn=True, num_keypoints=17)
        resize = (64, 48)

    elif FLAGS.model_type == 'stacked':
        model = StackedHourglass(is_use_bn=True, num_keypoints=17)

        resize = (64, 48)


    logits   = model.build(image, 'Hourglass', is_training=False, visualize=True)

    logits = tf.nn.sigmoid(logits)


    load_image = plt.imread(FLAGS.image_path)
    load_image = cv2.resize(load_image, (192, 256))

    if np.max(load_image) <= 1:
        load_image = load_image * 255

    saver, checkpoint_path = helper.create_saver_and_restore(sess, FLAGS.checkpoint_dir)
    # checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    time_list = []
    for _ in range(100):
        start = time.perf_counter()
        res = sess.run(logits, feed_dict={image: np.expand_dims(load_image, axis=0)})
        end = time.perf_counter()
        time_list.append(end - start)

    if FLAGS.is_measure:
        plt.boxplot(time_list)
        plt.title('time')
        plt.savefig('time_result.png')
        plt.show()
    res = res[0]
    helper.visualize_heatmaps(load_image, predict=res, is_separate=FLAGS.is_separate, is_save=FLAGS.is_save)
    helper.visualize_keypoints(load_image, predict_heatmap=res, is_save=FLAGS.is_save)





if __name__ == '__main__':
    tf.app.run()
