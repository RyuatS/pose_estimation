# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-05-15T03:00:51.267Z
# Description:
#
# ===============================================

# 実装されているバックボーンの名前
# already implemented backbone name
# If you use backbone network, output width and height is 1/16
# exapmle) input shape = [None, 256, 192, 3], output shape = [None, 16, 12, 2048]
BACKBONE_NAME_LIST = ['resnet_v1_101', 'resnet_v1_50']

R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94

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

SKELETON = [[15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6]]
