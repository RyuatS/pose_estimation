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
      + dataset
        + val2017 or train2017
        + annotations
    - helper.py


Image Directory:
  ./cocodevkit/dataset/{}
  {}: train2017 or val2017
  ./cocodevkit/dataset/train2017

Annotation file:
  ./cocodevkit/dataset/annotations/person_keypoints_{}.json
  {}: train2017 or val2017 or test2017

This script converts data into shrded data files and save at tfrecord folder.

The Example proto contains the following fields:
  image/filename: image filename.
  image/encoded: encoded image content.
  image/height: image height.
  image/width: image width.
  image/channels: image channles
  image/key_x_list: keypoint locations x
  image/key_y_list: keypoint locations y
  image/key_v_list: keypoint locations visible
"""

# lib
import tensorflow as tf
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# user packages
import helper
import convert_tfrecord
from cocodevkit.coco_helper import CocoLoader

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir',
                           './cocodevkit/dataset/val2017',
                           'Directory containing images.')

tf.app.flags.DEFINE_string('annotation_file',
                           './cocodevkit/dataset/annotations/person_keypoints_val2017.json',
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

_SMALL_RADIUS_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear'
]

# this indexes correspond to up keypoint_label.
_ALL_BODY_LABEL = [i for i in range(17)]
_NUM_ALL_BODY_KEYPOINTS = 17
_UPPER_BODY_LABEL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_NUM_UPPER_BODY_KEYPOINTS = 13
_RESIZE_SHAPE = (256, 192) # (HEIGHT, WIDTH)

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

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)


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
                bbox_x = bbox[0]
                bbox_y = bbox[1]

                cropped_img = loader.decode_image_and_crop(img_path, bbox)

                heatmaps = np.zeros(_RESIZE_SHAPE)

                # create heatmap
                for key_index in label_to_heatmap:
                    keypoint_name = _KEYPOINTS_LABEL[key_index]
                    key = keypoint_list[key_index]
                    if key[2] == 0:
                        heatmap = np.zeros(_RESIZE_SHAPE)
                    else:
                        # After cropping, fix keypoint location
                        key[0] = key[0] - bbox_x
                        key[1] = key[1] - bbox_y
                        key_resized_x = key[0]*_RESIZE_SHAPE[1]/cropped_img.shape[1]
                        key_resized_y = key[1]*_RESIZE_SHAPE[0]/cropped_img.shape[0]
                        # heatmap = helper.create_heatmap_numpy_gaussian(cropped_img.shape, (key_resized_x, key_resized_y), sigma=10, is_norm=True)
                        if keypoint_name in _SMALL_RADIUS_KEYPOINTS:
                            radius = 3
                        else:
                            radius = 10
                        heatmap = helper.create_heatmap_for_sigmoid(_RESIZE_SHAPE, (key_resized_x, key_resized_y), radius=radius)

                    heatmaps = np.dstack((heatmaps, heatmap))
                heatmaps = heatmaps[: ,:, 1:]
                resized_img = cv2.resize(cropped_img, _RESIZE_SHAPE[::-1])

                # test_visualize(resized_img, heatmaps)
                example = convert_tfrecord.image_heatmap_to_tfexample(
                    resized_img, heatmaps
                )
                tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\nComplete!!\n')
    sys.stdout.flush()

def test_visualize(resized_img, heatmaps):
    """
    make sure that heatmaps is created correctly.

    Args:
        resized_img: raw resized image. [height, width, 3].
        heatmaps: heatmaps. [height, width, keypoint_num].
                  heatmap's height and width should be same as image's height and width.

    """
    heat_channel = heatmaps.shape[2]
    temp = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY).astype(np.float64) * (heat_channel + 1)

    for c in range(heat_channel):
        h = heatmaps[:,:,c]
        temp += h * 255 * (heat_channel +1)
    temp /= (heat_channel + 1)
    plt.imshow(temp/255)
    plt.savefig('keypoint_vis.jpg')


def main(unused_argv):
    loader = CocoLoader(FLAGS.image_dir, FLAGS.annotation_file)
    print('length of coco single person image : %d' % len(loader))
    _convert_dataset(loader)


if __name__ == '__main__':
    tf.app.run()
