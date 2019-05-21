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
from lib.models.stacked_hourglass import StackedHourglass
from lib.core.config import BACKBONE_NAME_LIST


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
# tf.app.flags.DEFINE_enum('backbone', 'resnet_v1_101', BACKBONE_NAME_LIST,
#                          'Backbone network')
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

tf.app.flags.DEFINE_boolean('show',
                            False,
                            '')

tf.app.flags.DEFINE_enum('model_type',
                           'hourglass',
                           ['hourglass', 'stacked'],
                           'model type which should be defined ./lib/models/')

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

    if FLAGS.model_type == 'hourglass':
        model = Hourglass(is_use_bn=True, num_keypoints=17)

        resize = (128, 96)

    elif FLAGS.model_type == 'stacked':
        model = StackedHourglass(is_use_bn=True)

        resize = (64, 48)

    batch_size = FLAGS.batch_size
    input_size = (256, 192)
    dataset = Dataset(FLAGS.tfrecord_dir,
                      FLAGS.data_type,
                      batch_size,
                      input_size,
                      resize=resize,
                      should_repeat=True,
                      should_shuffle=True)

    iterator = dataset.get_one_shot_iterator()

    mini_batch = iterator.get_next()
    image = mini_batch['image']
    heatmaps = mini_batch['heatmaps']
    image = tf.cast(image, tf.float32)
    logits, savers = model.build(image, 'Model', is_training=True, visualize=True)
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

    logits = tf.nn.sigmoid(logits)

    ####################### setting saver ###########################
    global_saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoints_dir)
    if checkpoint:
        print('\n\n' + checkpoint.model_checkpoint_path)
        print('variables were restored.')
        global_saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        print('variables were initialized.')

        # load backbone weights
        if 'backbone' in model.model_structure.keys():
            pretrained_name = model.model_structure['backbone']['net']
            backbone_vars = tf.contrib.framework.get_variables_to_restore(include=[pretrained_name])
            pretrained_saver = tf.train.Saver(var_list=backbone_vars)
            pretrained_checkpoint = os.path.join('.',
                                                 'backbone_checkpoints',
                                                 '{}.ckpt'.format(pretrained_name))
            pretrained_saver.restore(sess, pretrained_checkpoint)
            print('{} weights were restored.'.format(pretrained_name))

    # checkpoint_path
    checkpoint_path = os.path.join(FLAGS.checkpoints_dir, 'model.ckpt')
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
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
            # l, _, summary_str, lr = sess.run([loss, train_op, writer_op, learning_rate])
            l, _, summary_str, lr, img, target, pred = sess.run([loss, train_op, writer_op, learning_rate, image, heatmaps, logits])
            # print('img.shape: {}, hm.shape: {}'.format(img.shape, hm.shape))
            # print('pred.shape: {}'.format(pred.shape))

            img = img[0]
            target = target[0]
            pred = pred[0]
            target = cv2.resize(target, (192, 256))
            pred = cv2.resize(pred, (192, 256))

            if FLAGS.show:
                helper.visualize_heatmaps(img, predict=pred, is_separate=True)

            loss_list.append(l)
            writer.add_summary(summary_str, global_step=step)
            # writer.flush()

            print('=> STEP %10d [TRAIN]:\tloss:%7.4f\t lr: %.4f' %(step, l, lr))

            if step % FLAGS.eval_interval == 0:
                print('==> mean loss: %7.4f' %(np.mean(loss_list)))
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
