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

KEYPOINTS_LABEL = [
    'nose',             # 0 Upper
    'left_eye',         # 1 Upper
    'right_eye',        # 2 Upper
    'left_ear',         # 3 Upper
    'right_ear',        # 4 Upper
    'left_shoulder',    # 5 Upper
    'right_shoulder',   # 6 Upper
    'left_elbow',       # 7 Upper
    'right_elbow',      # 8 Upper
    'left_wrist',       # 9 Upper
    'right_wrist',      # 10 Upper
    'left_hip',         # 11 Upper
    'right_hip',        # 12 Upper
    'left_knee',        # 13
    'right_knee',       # 14
    'left_ankle',       # 15
    'right_ankle'       # 16
]

SKELETON = [
        [15, 13, (255, 0, 0)],   # left_ankle     - left_knee
        [13, 11, (255, 0, 0)],   # left_knee      - left_hip
        [5,  11, (255, 0, 0)],   # left_shoulder  - left_hip
        [5,   7, (255, 0, 0)],   # left_shoulder  - left_elbow
        [7,   9, (255, 0, 0)],   # left_elbow     - left_wrist
        [16, 14, (0, 0, 255)],   # right_ankle    - right_knee
        [14, 12, (0, 0, 255)],   # right_knee     - right_hip
        [6,  12, (0, 0, 255)],   # right_shoulder - right_hip
        [6,   8, (0, 0, 255)],   # right_shoulder - right_elbow
        [8,  10, (0, 0, 255)],   # right_elbow    - right_wrist
        [5,   6, (255, 0, 255)], # left_shoulder  - right_shoulder
        [1,   2, (0, 255, 0)],   # left_eye       - right_eye
        [0,   1, (0, 255, 0)],   # nose           - left_eye
        [0,   2, (0, 255, 0)],   # nose           - right_eye
        [1,   3, (0, 255, 0)],   # left_eye       - left_ear
        [2,   4, (0, 255, 0)]    # right_eye      - right_ear
]
        # [11, 12, (255, 0, 255)],   # left_hip       - right_hip
        # [3, 5],     # left_ear       - left_shoulder
        # [4, 6]]     # right_ear      - right_shoulder

SMALL_RADIUS_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear'
]

# this indexes correspond to up keypoint_label.
ALL_BODY_LABEL = [i for i in range(17)]
UPPER_BODY_LABEL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
RESIZE_SHAPE = (256, 192) # (HEIGHT, WIDTH)
