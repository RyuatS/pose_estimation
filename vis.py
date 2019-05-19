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
from lib.core.config import BACKBONE_NAME_LIST, _KEYPOINTS_LABEL, CONNECT


# user packages
from lib.models.hourglass import Hourglass

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('check_dir',
                           './checkpoints_dir',
                           'checkpoints directory')

tf.app.flags.DEFINE_string('image_path',
                           './god.jpg',
                           '')

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

    checkpoint = tf.train.get_checkpoint_state(FLAGS.check_dir)

    if checkpoint:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint.model_checkpoint_path)
        res = sess.run(logits, feed_dict={image: np.expand_dims(load_image, axis=0)})

        res = res[0]
        res = cv2.resize(res, (192, 256))


        # load_image = load_image.astype(np.float32)
        # gray = cv2.cvtColor(load_image, cv2.COLOR_RGB2GRAY)
        # gray /= 255
        # res = np.max(res, axis=2)
        # result = (gray + res) / 2
        # plt.imshow(result, cmap='gray')
        # plt.title('result')
        # plt.show()

        # plt.figure(figsize=(12, 10))
        # plt.subplot(3, 6, 1)
        # plt.imshow(load_image.astype(np.uint8))
        # num_channel = res.shape[2]
        # for i in range(num_channel):
        #     plt.subplot(3, 6, i+2)
        #     plt.imshow(res[:,:,i], cmap='gray')
        #     plt.title(_KEYPOINTS_LABEL[i])
        # plt.show()

        keypoint_locs = []
        num_channel = res.shape[2]
        for i in range(num_channel):
            h = res[:,:, i]
            temp = h.ravel()
            temp = sorted(temp)
            max_10 = temp[-20]
            h[h < max_10] = 0
            h[h >= max_10] = 1
            plt.imshow(h)
            plt.title(_KEYPOINTS_LABEL[i])
            plt.show()

            width_sum = np.sum(h, axis=0).flatten()
            height_sum = np.sum(h, axis=1).flatten()
            mean_x = 0
            mean_y = 0
            count_x = 0
            count_y = 0
            for x in range(len(width_sum)):
                if width_sum[x] >= 1:
                    mean_x += x
                    count_x += 1
            for y in range(len(height_sum)):
                if height_sum[y] >= 1:
                    mean_y += y
                    count_y += 1

            key_x = mean_x / count_x
            key_y = mean_y / count_y
            keypoint_loc = [key_x, key_y]
            keypoint_locs.append(keypoint_loc)

        for CON in CONNECT:
            ind1, ind2 = CON
            key1 = keypoint_locs[ind1]
            key2 = keypoint_locs[ind2]
            key1 = np.array(key1).astype(np.int)
            key2 = np.array(key2).astype(np.int)
            load_image = cv2.line(load_image, tuple(key1),tuple(key2),(255,0,0),1)

        for i in range(len(keypoint_locs)):
            x, y = keypoint_locs[i]
            plt.scatter(x, y, label=_KEYPOINTS_LABEL[i])
        plt.imshow(load_image.astype(np.uint8))
        plt.legend()
        plt.show()



    else:
        raise ValueError("'{}' does not exist".format(FLAGS.check_dir))


if __name__ == '__main__':
    tf.app.run()
