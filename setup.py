# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description: training script.
#
# ===============================================

"""
Setup script. This script do following steps.
1) Make directory 'checkpoints', 'backbone_checkpoints' and 'data/cocodevkit/dataset'.
   The checkpoints directory will contain trained models checkpoint files.
   The backbone_checkpoints will contain pre-trained models(resnet, vgg) checkpoint files.
   The data/cocodevkit/dataset will contain ms-coco dataset. (image and annotations).
2) Download pre-trained model checkpoint files and ms-coco dataset.
3) Unzip downloaded files.
"""

import os
import sys
import argparse


parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        print("directory '{}' is already exist".format(dir_name))



make_dir('checkpoints')
make_dir('backbone_checkpoints')
make_dir(os.path.join('data', 'cocodevkit', 'dataset'))
