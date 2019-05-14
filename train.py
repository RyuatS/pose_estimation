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

# lib
import time
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

# user packages
from lib.utils import helper
from data.dataset_generator import Dataset
from lib.models.hourglass import Hourglass


FLAGS = tf.app.flags.FLAGS

# about dataset, checkpoint and log.
tf.app.flags.DEFINE_string('checkpoints_dir', './checkpoints', 'checkpoint directory for saving model')
tf.app.flags.DEFINE_string('tfrecord_dir', './data/cocodevkit/tfrecord', 'tfrecord directory')
tf.app.flags.DEFINE_enum('data_type', 'val2017', ['train2017', 'val2017'], 'train2017 or val2017')
tf.app.flags.DEFINE_string('logdir', './logdir', 'tensorboard log directory')

# saver
tf.app.flags.DEFINE_integer('save_interval',
                            100,
                            '保存する間隔')
tf.app.flags.DEFINE_integer('eval_interval',
                            100,
                            'evaluate interval (not yet)')

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
                          'The value of the weight decay for training with weight l2 loss')


def visualize_flags():
    """
    visualize flags
    """
    print('-' * 40)
    for key in FLAGS.__flags.keys():
        if key in ['h', 'help', 'helpfull', 'helpshort']:
            pass
        else:
            print('{:20} : {}'.format(key, FLAGS[key].value))
    print('-' * 40)

def main(unused_argv):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config = config)

    visualize_flags()

    batch_size = FLAGS.batch_size
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

    # global step holder
    global_step = tf.Variable(0, name='global_step')
    global_step_holder = tf.placeholder(tf.int32)
    global_step_op = global_step.assign(global_step_holder)

    learning_rate = tf.train.exponential_decay(FLAGS.base_learning_rate,
                                               global_step=global_step,
                                               decay_steps=FLAGS.learning_rate_decay_step,
                                               decay_rate =FLAGS.learning_rate_decay_factor,
                                               staircase=True)
    # get loss and train_operater
    loss, train_op = model.get_train_op(logits,
                                        heatmaps,
                                        scope='Hourglass',
                                        learning_rate=learning_rate,
                                        decay_rate=FLAGS.weight_decay)



    global_saver = tf.train.Saver()
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
    step = sess.run(global_step)
    #################################################################

    ################### setting summary writer ######################
    writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    writer_op = tf.summary.merge_all()
    #################################################################

    loss_list = []
    try:
        for _ in range(FLAGS.steps):
            step += 1
            l, _, summary_str = sess.run([loss, train_op, writer_op])

            loss_list.append(l)
            writer.add_summary(summary_str, global_step=step)
            writer.flush()

            print('=> STEP %10d [TRAIN]:\tloss:%7.4f ' %(step, l))

            if step % FLAGS.eval_interval == 0:
                print(' - mean loss: %7.4f' %(np.mean(loss_list)))
                loss_list = []

            if step % FLAGS.save_interval == 0:
                sess.run(global_step_op, feed_dict={global_step_holder: step})
                save_path = global_saver.save(sess, checkpoint_path, global_step=step)
                print('\nModel saved in path: %s' % save_path)

    except KeyboardInterrupt:
        print('\ncatch keyboard interrupt.')
    finally:
        # save
        sess.run(global_step_op, feed_dict={global_step_holder: step})
        save_path = global_saver.save(sess, checkpoint_path, global_step=step)
        print('\nModel saved in path: %s' % save_path)
        sess.close()

    ####################### summary #################################
    # tf.summary.scalar('loss', loss)
    # write_op     = tf.summary.merged_all()
    # writer_train = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'train'), graph=sess.graph)
    # writer_val   = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'val'))
    #################################################################



if __name__ == '__main__':
    tf.app.run()
