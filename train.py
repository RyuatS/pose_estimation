# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description: training script.
#
# ===============================================

"""
training script for single human pose estimation.

"""

# packages
import time
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

# my script
from lib.utils import helper
from data.dataset_generator import Dataset
from lib.models.hourglass import Hourglass


FLAGS = tf.app.flags.FLAGS

# about dataset, checkpoint and log.
tf.app.flags.DEFINE_string('checkpoints_dir', './checkpoints', 'checkpoint directory for saving model')
tf.app.flags.DEFINE_string('tfrecord_dir', './data/cocodevkit/tfrecord', 'tfrecord directory')
tf.app.flags.DEFINE_enum('data_type', 'val2017', ['train2017', 'val2017'], 'train2017 or val2017')
tf.app.flags.DEFINE_string('logdir', './logdir', 'tensorboard log directory')

# Hyper parameters.
tf.app.flags.DEFINE_enum('backbone', 'resnet_101', ['resnet_101', 'resnet_50', 'vgg16'],
                         'Backbone network')
tf.app.flags.DEFINE_integer('steps', 1000, 'steps')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch_size')
tf.app.flags.DEFINE_float('base_learning_rate', .0001,
                          'The base learning rate for model training.')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          'The rate to decay the base learning rate.')

tf.app.flags.DEFINE_integer('learning_rate_decay_step', 2000,
                            'Decay the base learning rate at a fixed step.')

tf.app.flags.DEFINE_float('weight_decay', 0.00005,
                          'The value of the weight decay for training')


def create_heatmaps(heatmap_shape, key_x_list, key_y_list, key_v_list):
    """
    create heatmaps.

    Args:
        heamap_shape: heatmap shape. It should be equal to input_size. (height, width)
        key_x_list: keypoint locations x. [x1, x2, ...].
        key_y_list: keypoint locations y. [y1, y2, ...].
        key_v_list: keypoint visible or not. [v1, v2, ...].
        key x1 is correspond to y1 and v1. If v1 is 0, there is not label1 keypoint in image. => create all zeros heatmap.
                                           If v1 is 1, there is label1 keypoint in image. => create heatmap.
    Returns:
        heatmaps. [height, width, num_keypoints].
        If you want to detect all body, num_keypoints is 17.
        If you want to detect upper body, num_keypoints is 13.
    """

    heatmaps = np.zeros(heatmap_shape)
    for key_index in range(len(key_x_list)):
        key_x = key_x_list[key_index]
        key_y = key_y_list[key_index]
        key_v = key_v_list[key_index]

        if key_v == 0:
            heatmap = np.zeros(heatmap_shape)
        else:
            heatmap = helper.create_heatmap_numpy(heatmap_shape,
                                                  (key_x, key_y),
                                                  sigma=10.0,
                                                  is_norm=False)
        heatmaps = np.dstack((heatmaps, heatmap))

    return heatmaps[... ,1:]


def get_imgs_and_heatmap(sess, mini_batch, input_size):
    mini_batch_data = sess.run(mini_batch)
    mini_batch_imgs = mini_batch_data['image']

    mini_batch_heatmaps = []

    mini_batch_key_x = mini_batch_data['key_x']
    mini_batch_key_y = mini_batch_data['key_y']
    mini_batch_key_v = mini_batch_data['key_v']

    for ind in range(len(mini_batch_key_x)):
        key_x_list = mini_batch_key_x[ind]
        key_y_list = mini_batch_key_y[ind]
        key_v_list = mini_batch_key_v[ind]

        heatmaps = create_heatmaps(input_size, key_x_list, key_y_list, key_v_list)

        mini_batch_heatmaps.append(heatmaps)


    return mini_batch_imgs, np.array(mini_batch_heatmaps)

def main(unused_argv):
    batch_size = FLAGS.batch_size

    print('-' * 40)
    for key in FLAGS.__flags.keys():
        if key == 'h' or key == 'help' or key == 'helpfull' or key =='helpshort':
            pass
        else:
            print('{:20} : {}'.format(key, FLAGS[key].value))
    print('-' * 40)

    input_size = (256, 192)
    dataset = Dataset(FLAGS.tfrecord_dir,
                      FLAGS.data_type,
                      batch_size,
                      input_size,
                      should_repeat=True,
                      should_shuffle=True)

    iterator = dataset.get_one_shot_iterator()

    mini_batch = iterator.get_next()
    image = mini_batch['image']
    heatmaps = mini_batch['heatmaps']
    image = tf.to_float(image)
    model = Hourglass(is_use_bn=True, num_keypoints=17)
    logits, savers = model.build(image, 'Hourglass', is_training=True, visualize=True)

    # get loss and train_operater
    loss, train_op = model.get_train_op(logits,
                                        heatmaps,
                                        scope='Hourglass',
                                        learning_rate=FLAGS.base_learning_rate,
                                        decay_rate=FLAGS.weight_decay
                                        )

    # global step holder
    global_step = tf.Variable(0, name='global_step')
    global_step_holder = tf.placeholder(tf.int32)
    global_step_op = global_step.assign(global_step_holder)


    global_saver = tf.train.Saver()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:

        ####################### setting saver ###########################
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoints_dir)
        if checkpoint:
            print('\n\n' + checkpoint.model_checkpoint_path)
            print('variables were restored.')
            global_saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            print('variables were initialized.')

            # load backbone weights
            for key in savers.keys():
                saver = savers[key]['saver']
                checkpoint_path = savers[key]['checkpoint_path']
                saver.restore(sess, checkpoint_path)
                print('{} weights were restored.'.format(key))
            print('model_checkpoints {}'.format(FLAGS.checkpoints_dir))

        # checkpoint_path
        checkpoint_path = os.path.join(FLAGS.checkpoints_dir, 'model.ckpt')
        if not os.path.exists(FLAGS.checkpoints_dir):
            os.makedirs(checkpoints_dir)
        step = global_step.eval()
        #################################################################

        loss_summary = tf.summary.scalar('loss', loss)
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        merged = tf.summary.merge_all()

        loss_list = []
        try:
            for train_count in range(FLAGS.steps):
                step += 1
                l, _ = sess.run([loss, train_op])

                loss_list.append(l)
                disp_string = '\rstep {train_step:{len}}: loss - {loss:.2f}'.format(train_step=train_count+1,
                                                                    len=len(str(FLAGS.steps)),
                                                                    loss=l)
                sys.stdout.write(disp_string)
                if train_count % 100 == 0:
                    print(' - mean loss: {}'.format(np.mean(loss_list)))
                # print('step {train_step:{len}}: loss - {loss:.2f}'.format(train_step=train_count+1,
                #                                                           len=len(str(FLAGS.steps)),
                #                                                           loss=l))

                if step % 5 == 0:
                    # record summary
                    summary_str = sess.run(merged)
                    writer.add_summary(summary_str, step)

                if step % 1000 == 0:
                    sess.run(global_step_op, feed_dict={global_step_holder: step})
                    save_path = global_saver.save(sess, checkpoint_path, global_step=step)
                    print('\nModel saved in path: %s' % save_path)


            save_path = global_saver.save(sess, checkpoint_path, global_step=step)
            print('\nModel saved in path %s' % save_path)

        except KeyboardInterrupt:
            print('\ncatch keyboard interrupt.')
        finally:
            # save
            sess.run(global_step_op, feed_dict={global_step_holder: step})
            save_path = global_saver.save(sess, checkpoint_path, global_step=step)
            print('\nModel saved in path: %s' % save_path)

if __name__ == '__main__':
    tf.app.run()
