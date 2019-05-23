# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-29T09:02:57.278Z
# Description:
#
# ===============================================

from .ParentFCN import ParentFCN
from collections import OrderedDict

class StackedHourglass(ParentFCN):
    def __init__(self, is_use_bn, num_keypoints=17):
        super().__init__(is_use_bn=True)
        self._model_type = 'Stacked Hour Glass Net'

        self.layer_funcs['preprocess'] = self.preprocess

        self.model_structure = OrderedDict([
            # input > 256x192
            # 256x192x3 => 128x96x256
            ('preprocess', {}),
            ('conv1', {'filter_shape': [7, 7, 3, 256], 'strides': [1, 2, 2, 1]}),
            # 128x96x256 => 64x48x256
            ('pool1', {'strides': [1, 2, 2, 1], 'pool_type': 'max'}),

            # residual block1 64x48x256 => 32x24x256
            ('conv2_bottleneck/hourglass1', {'filter_shape': [1, 1, 256, 128], 'strides': [1, 1, 1, 1]}),
            ('conv2_1/hourglass1', {'filter_shape': [3, 3, 128, 128], 'strides': [1, 1, 1, 1]}),
            ('conv2_2/hourglass1', {'filter_shape': [3, 3, 128, 256], 'strides': [1, 1, 1, 1]}),
            ('residual1/hourglass1', {'residual': 'pool1'}),
            ('pool2/hourglass1',   {'strides': [1, 2, 2, 1], 'pool_type': 'max'}),

            # residual block2 32x24x256 => 16x12x256
            ('conv3_bottleneck/hourglass1', {'filter_shape': [1, 1, 256, 128], 'strides': [1, 1, 1, 1]}),
            ('conv3_1/hourglass1', {'filter_shape': [3, 3, 128, 128], 'strides': [1, 1, 1, 1]}),
            ('conv3_2/hourglass1', {'filter_shape': [3, 3, 128, 256], 'strides': [1, 1, 1, 1]}),
            ('residual2/hourglass1', {'residual': 'pool2/hourglass1'}),
            ('pool3/hourglass1',   {'strides': [1, 2, 2, 1], 'pool_type': 'max'}),

            # residual block3 16x12x256 => 8x6x256
            ('conv4_bottleneck/hourglass1', {'filter_shape': [1, 1, 256, 128], 'strides': [1, 1, 1, 1]}),
            ('conv4_1/hourglass1', {'filter_shape': [3, 3, 128, 128], 'strides': [1, 1, 1, 1]}),
            ('conv4_2/hourglass1', {'filter_shape': [3, 3, 128, 256], 'strides': [1, 1, 1, 1]}),
            ('residual3/hourglass1', {'residual': 'pool3/hourglass1'}),
            ('pool4/hourglass1',   {'strides': [1, 2, 2, 1], 'pool_type': 'max'}),


            # deconv5   8x6x256 => 16x12x256
            ('deconv5/hourglass1', {'filter_shape': [3, 3, 256, 256], 'strides': [1, 2, 2, 1], 'output_shape': [None, 16, 12, 256]}),
            ('conv5_1/hourglass1', {'filter_shape': [1, 1, 256,  128], 'strides': [1, 1, 1, 1]}),
            ('conv5_2/hourglass1', {'filter_shape': [3, 3,  128,  128], 'strides': [1, 1, 1, 1]}),
            ('conv5_3/hourglass1', {'filter_shape': [1, 1,  128,  256],'strides': [1, 1, 1, 1]}),
            ('residual4/hourglass1', {'residual': 'pool3/hourglass1'}),


            # deconv6  16x12x256 => 32x24x256
            ('deconv6/hourglass1', {'filter_shape': [3, 3, 256, 256], 'strides': [1, 2, 2, 1], 'output_shape': [None, 32, 24, 256]}),
            ('conv6_1/hourglass1', {'filter_shape': [1, 1, 256,  128], 'strides': [1, 1, 1, 1]}),
            ('conv6_2/hourglass1', {'filter_shape': [3, 3,  128,  128], 'strides': [1, 1, 1, 1]}),
            ('conv6_3/hourglass1', {'filter_shape': [1, 1,  128,  256],'strides': [1, 1, 1, 1]}),
            ('residual5/hourglass1', {'residual': 'pool2/hourglass1'}),

            # deconv7 32x24x256 => 64x48x256
            ('deconv7/hourglass1', {'filter_shape': [3, 3, 256, 256], 'strides': [1, 2, 2, 1], 'output_shape': [None, 64, 48, 256]}),
            ('conv7_1/hourglass1', {'filter_shape': [1, 1, 256,  128], 'strides': [1, 1, 1, 1]}),
            ('conv7_2/hourglass1', {'filter_shape': [3, 3,  128,  128], 'strides': [1, 1, 1, 1]}),
            ('conv7_3/hourglass1', {'filter_shape': [1, 1,  128,  256],'strides': [1, 1, 1, 1]}),
            ('residual6/hourglass1', {'residual': 'pool1'}),

            ('conv_supervision/hourglass1', {'filter_shape': [1, 1, 256, num_keypoints], 'strides': [1, 1, 1, 1], 'activation': 'no'}),


            # HourGlass2
            # residual block1 64x48x256 => 32x24x256
            ('conv2_bottleneck/hourglass2', {'filter_shape': [1, 1, 256, 128], 'strides': [1, 1, 1, 1], 'input': 'residual6/hourglass1'}),
            ('conv2_1/hourglass2', {'filter_shape': [3, 3, 128, 128], 'strides': [1, 1, 1, 1]}),
            ('conv2_2/hourglass2', {'filter_shape': [3, 3, 128, 256], 'strides': [1, 1, 1, 1]}),
            ('residual1/hourglass2', {'residual': 'pool1'}),
            ('pool2/hourglass2',   {'strides': [1, 2, 2, 1], 'pool_type': 'max'}),

            # residual block2 32x24x256 => 16x12x256
            ('conv3_bottleneck/hourglass2', {'filter_shape': [1, 1, 256, 128], 'strides': [1, 1, 1, 1]}),
            ('conv3_1/hourglass2', {'filter_shape': [3, 3, 128, 128], 'strides': [1, 1, 1, 1]}),
            ('conv3_2/hourglass2', {'filter_shape': [3, 3, 128, 256], 'strides': [1, 1, 1, 1]}),
            ('residual2/hourglass2', {'residual': 'pool2/hourglass2'}),
            ('pool3/hourglass2',   {'strides': [1, 2, 2, 1], 'pool_type': 'max'}),

            # residual block3 16x12x256 => 8x6x256
            ('conv4_bottleneck/hourglass2', {'filter_shape': [1, 1, 256, 128], 'strides': [1, 1, 1, 1]}),
            ('conv4_1/hourglass2', {'filter_shape': [3, 3, 128, 128], 'strides': [1, 1, 1, 1]}),
            ('conv4_2/hourglass2', {'filter_shape': [3, 3, 128, 256], 'strides': [1, 1, 1, 1]}),
            ('residual3/hourglass2', {'residual': 'pool3/hourglass2'}),
            ('pool4/hourglass2',   {'strides': [1, 2, 2, 1], 'pool_type': 'max'}),


            # deconv5   8x6x256 => 16x12x256
            ('deconv5/hourglass2', {'filter_shape': [3, 3, 256, 256], 'strides': [1, 2, 2, 1], 'output_shape': [None, 16, 12, 256]}),
            ('conv5_1/hourglass2', {'filter_shape': [1, 1, 256,  128], 'strides': [1, 1, 1, 1]}),
            ('conv5_2/hourglass2', {'filter_shape': [3, 3,  128,  128], 'strides': [1, 1, 1, 1]}),
            ('conv5_3/hourglass2', {'filter_shape': [1, 1,  128,  256],'strides': [1, 1, 1, 1]}),
            ('residual4/hourglass2', {'residual': 'pool3/hourglass2'}),


            # deconv6  16x12x256 => 32x24x256
            ('deconv6/hourglass2', {'filter_shape': [3, 3, 256, 256], 'strides': [1, 2, 2, 1], 'output_shape': [None, 32, 24, 256]}),
            ('conv6_1/hourglass2', {'filter_shape': [1, 1, 256,  128], 'strides': [1, 1, 1, 1]}),
            ('conv6_2/hourglass2', {'filter_shape': [3, 3,  128,  128], 'strides': [1, 1, 1, 1]}),
            ('conv6_3/hourglass2', {'filter_shape': [1, 1,  128,  256],'strides': [1, 1, 1, 1]}),
            ('residual5/hourglass2', {'residual': 'pool2/hourglass2'}),

            # deconv7 32x24x256 => 64x48x256
            ('deconv7/hourglass2', {'filter_shape': [3, 3, 256, 256], 'strides': [1, 2, 2, 1], 'output_shape': [None, 64, 48, 256]}),
            ('conv7_1/hourglass2', {'filter_shape': [1, 1, 256,  128], 'strides': [1, 1, 1, 1]}),
            ('conv7_2/hourglass2', {'filter_shape': [3, 3,  128,  128], 'strides': [1, 1, 1, 1]}),
            ('conv7_3/hourglass2', {'filter_shape': [1, 1,  128,  256],'strides': [1, 1, 1, 1]}),
            ('residual6/hourglass2', {'residual': 'pool1'}),

            ('conv_out', {'filter_shape': [1, 1, 256, num_keypoints], 'strides': [1, 1, 1, 1], 'activation': 'no'})




        ])


    def preprocess(self, input, key, param_dict, is_training):

        return input / 255
