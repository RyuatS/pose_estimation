# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-29T09:02:57.278Z
# Description:
#
# ===============================================

from .ParentFCN import ParentFCN
from collections import OrderedDict

class Hourglass(ParentFCN):
    def __init__(self, is_use_bn, num_keypoints=17):
        super().__init__(is_use_bn=True)
        self._model_type = 'Hour Glass Net'

        self.model_structure = OrderedDict([
            # input > 256x192
            # backbone net 8x6x1024
            ('backbone1', {'net': 'resnet', 'checkpoint_path': 'backbone_checkpoints/resnet_v1_101.ckpt'}),

            # 8x6x1024 => 16x12x512
            ('deconv1', {'filter_shape': [3, 3, 512, 1024], 'strides': [1, 2, 2, 1], 'output_shape': [None, 16, 12, 512]}),
            ('conv1_1', {'filter_shape': [3, 3, 512, 512], 'strides': [1, 1, 1, 1]}),
            ('conv1_2', {'filter_shape': [3, 3, 512, 512], 'strides': [1, 1, 1, 1]}),

            # 16x12x512 => 32x24x256
            ('deconv2', {'filter_shape': [3, 3, 256, 512], 'strides': [1, 2, 2, 1], 'output_shape': [None, 32, 24, 256]}),
            ('conv2_1', {'filter_shape': [3, 3, 256, 256], 'strides': [1, 1, 1, 1]}),
            ('conv2_2', {'filter_shape': [3, 3, 256, 256], 'strides': [1, 1, 1, 1]}),

            # 32x24x256 => 64x48x128
            ('deconv3', {'filter_shape': [3, 3, 128, 256], 'strides': [1, 2, 2, 1], 'output_shape': [None, 64, 48, 128]}),
            ('conv3_1', {'filter_shape': [3, 3, 128, 128], 'strides': [1, 1, 1, 1]}),
            ('conv3_2', {'filter_shape': [3, 3, 128, 128], 'strides': [1, 1, 1, 1]}),

            # 64x48x128 => 128x96x64
            ('deconv4', {'filter_shape': [3, 3, 64, 128], 'strides': [1, 2, 2, 1], 'output_shape': [None, 128, 96, 64]}),
            ('conv4_1', {'filter_shape': [3, 3, 64,  64], 'strides': [1, 1, 1, 1]}),
            ('conv4_2', {'filter_shape': [3, 3, 64,  64], 'strides': [1, 1, 1, 1]}),

            # 128x96x32 => 256x192xK
            ('deconv5', {'filter_shape': [3, 3, 32, 64], 'strides': [1, 2, 2, 1], 'output_shape': [None, 256, 192, 32]}),
            ('conv5_1', {'filter_shape': [3, 3, 32, 32], 'strides': [1, 1, 1, 1]}),
            ('conv5_2', {'filter_shape': [1, 1, 32, num_keypoints], 'strides': [1, 1, 1, 1]})
        ])
