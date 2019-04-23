# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

"""
Converts coco data to TFReacord file format with Example protos.

This dataset is expected to have the following directory structure:

  + data
    - convert_tfrecord.py
    - build_coco_data.py (current working directory).
    + cocodevkit
      + tfrecord
      - coco_helper.py
    - helper.py


Image Directory:
  /home/user/coco

Annotation file:
  /home/user/coco/annotations/person_keypoints_{}.json
  {}: train2017 or val2017 or test2017

This script converts data into shrded data files and save at tfrecord folder.

The Example proto contains the following fields:
  image/filename: image filename.
  image/encoded: encoded image content.
  image/height: image height.
  image/width: image width.
  image/channels: image channles
  image/part/x: keypoint locations x
  image/part/y: keypoint locations y
  image/part/v: keypoint locations visible
  image/heatmap: keypoint heatmap
"""

import tensorflow as tf
import sys
import math
import convert_tfrecord
import helper
from cocodevkit.coco_helper import CocoLoader
import convert_tfrecord
import numpy as np
import convert_tfrecord
import matplotlib.pyplot as plt

import os
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir',
                           '/home/ryutashitomi/coco/val2017',
                           'Directory containing images.')

tf.app.flags.DEFINE_string('annotation_file',
                           '/home/ryutashitomi/coco/annotations/person_keypoints_val2017.json',
                           'annotation file. train2017 or val2017')


tf.app.flags.DEFINE_string('output_dir',
                           './cocodevkit/tfrecord',
                           'Path to save converted SSTable of TensorFlow examples.')

tf.app.flags.DEFINE_enum('include_keypoint',
                           'all', ['all', 'upper'],
                           'which do you include keypoint label. all or upper.')


# dataset
_NUM_SHARDS = 10

# キーポイントが上半身の場合の含むクラスの記述
_KEYPOINTS_LABEL = [
    'nose',             # 0
    'left_eye',         # 1
    'right_eye',        # 2
    'left_ear',         # 3
    'right_ear',        # 4
    'left_shoulder',    # 5
    'right_shoulder',   # 6
    'left_elbow',       # 7
    'right_elbow',      # 8
    'left_wrist',       # 9
    'right_wrist',      # 10
    'left_hip',         # 11
    'right_hip',        # 12
    'left_knee',        # 13
    'right_knee',       # 14
    'left_ankle',       # 15
    'right_ankle'       # 16
]

# this indexes correspond to up keypoint_label.
_ALL_BODY_LABEL = [i for i in range(17)]
_UPPER_BODY_LABEL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def _convert_dataset(loader):
    """
    Converts the specified dataset split to TFRecord format.

    Args:
        loader: coco dataset loader class. this class defined in coco_helper.py .

    Raises:
        RuntimeError: If keypoint_location is out of image height or image width.
    """

    dataset = os.path.basename(FLAGS.image_dir).split('.')[0]
    sys.stdout.write('Processing ' + dataset + '\n')
    num_images = len(loader)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    # どこまでのキーポイントを含むかで、キーポイントのラベルをヒートマップに変換する数を変える
    # all-> すべてのキーポイントをヒートマップに変換する
    # upper -> right_wristまでをヒートマップに変換する
    if FLAGS.include_keypoint == 'all':
        label_to_heatmap = _ALL_BODY_LABEL
    else:
        label_to_heatmap = _UPPER_BODY_LABEL

    keypoint_anns = loader.keypoint_anns
    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)

            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Convering image %d/%d shard %d' % (
                    i + 1, num_images, shard_id
                ))
                sys.stdout.flush()

                # get the img information from ann['image_id']
                ann = keypoint_anns[i]
                img_id = ann['image_id']
                img_information = loader.get_img_inf(img_id)

                img_path = os.path.join(
                    FLAGS.image_dir, img_information['file_name'])
                keypoint = ann['keypoints']
                keypoint_list = [keypoint[offset:offset+3] for offset in range(0, len(keypoint), 3)]
                bbox = ann['bbox']

                cropped_img = loader.decode_image_and_crop(img_path, bbox)

                heatmaps = np.zeros((cropped_img.shape[0], cropped_img.shape[1]))

                # create heatmap
                for key_index in label_to_heatmap:
                    key = keypoint_list[key_index]
                    if key[2] == 0:
                        heatmap = np.zeros((cropped_img.shape[0], cropped_img.shape[1]))
                    else:
                        heatmap = helper.create_heatmap(cropped_img, key[:2], sigma=1)
                    heatmaps = np.dstack((heatmaps, heatmap))

                    example = convert_tfrecord.image_heatmap_to_tfexample(
                        cropped_img, heatmaps
                    )
                    tfrecord_writer.write(example.SerializeToString())
                # ======================= below incomplete =============================== #

        sys.stdout.write('\nComplete!!')
        sys.stdout.flush()


def main(unused_argv):
    loader = CocoLoader(FLAGS.image_dir, FLAGS.annotation_file)
    print('length of coco single person image : %d' % len(loader))
    _convert_dataset(loader)


if __name__ == '__main__':
    tf.app.run()
