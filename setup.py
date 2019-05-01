# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

"""
Setup script. This script do following steps.
1) Make directory 'checkpoints', 'backbone_checkpoints' and 'data/cocodevkit/dataset'.
   The checkpoints directory will contain models checkpoint files you train.
   The backbone_checkpoints will contain pre-trained models(resnet, vgg) checkpoint files.
   The data/cocodevkit/dataset will contain ms-coco dataset. (image and annotations).
2) Download pre-trained model checkpoint files and ms-coco dataset.
3) Unzip downloaded files.
"""
import os
import sys
import urllib.request, shutil, tarfile, zipfile
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum('pre_trained', 'resnet_v1_101',
                         ['vgg16', 'resnet_v1_101', 'resnet_v1_50'],
                         'Pre-trained model name you wanna download (vgg16, resnet_v1_101, resnet_v1_50)')

tf.app.flags.DEFINE_enum('dataset_type', 'val', ['val', 'train'],
                         'coco dataset type (val or train)')

PRE_TRAINED_URL_TABLE = {
    'vgg16': 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
    'resnet_v1_101': 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
    'resnet_v1_50': 'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
}

COCODATA_URL_TABLE = {
   'train': 'http://images.cocodataset.org/zips/train2017.zip',
   'val'  : 'http://images.cocodataset.org/zips/val2017.zip',
   'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}



def make_dir(dir_name):
    """
    make directory.

    Args:
        dir_name: directory name you want to make.

    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("make directory '{}'".format(dir_name))
    else:
        print("directory '{}' is already exist".format(dir_name))


def download_pretrained_model(url, save_dir):
    """
    Download pretrained model.

    Args:
        url: string. tensorflow checkpoint url.
        save_dir: string. directory you save.

    """
    save_path = os.path.join(save_dir, url.split('/')[-1])

    with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        print('download {}'.format(url))

        with tarfile.open(save_path, 'r:gz') as tar:
            tar_members = tar.getmembers()

            for member in tar_members:
                tar.extract(member=member, path=save_dir)
                print('extracting {} succeed!'.format(member.name))


def download_dataset(img_url, ann_url, save_dir):
    """
    """

    save_img_path = os.path.join(save_dir, img_url.split('/')[-1])
    save_ann_path = os.path.join(save_dir, ann_url.split('/')[-1])

    if os.path.exists(save_img_path):
        with zipfile.ZipFile(save_img_path) as zip:
            name_list = zip.namelist()
            num_files = len(name_list)
            for i, name in enumerate(name_list, start=1):
                zipinfo = zip.getinfo(name)
                zip.extract(zipinfo, path=save_dir)
                sys.stdout.write('\rextract {}/{} name: {}'.format(i, num_files, name))
            sys.stdout.write('\nextract all done!\n')
    else:
        with urllib.request.urlopen(img_url) as response, open(save_img_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            print('download {}'.format(img_url))

            with zipfile.ZipFile(save_img_path) as zip:
                zipinfo = zip.getinfo()

    if os.path.exists(save_ann_path):
        with zipfile.ZipFile(save_ann_path) as zip


def extract_zip(file_path, save_dir=None):

    with zipfile.ZipFile(file_path) as zip:
        name_list = zip.namelist()
        num_files = len(name_list)
        for i, name in enumerate(name_list, start=1):
            zipinfo = 


def main(argv):

    backbone_dir = 'backbone_checkpoints'
    cocodata_dir = os.path.join('data', 'cocodevkit', 'dataset')

    make_dir('checkpoints')
    make_dir(backbone_dir)
    make_dir(cocodata_dir)

    pre_trained_model_url = PRE_TRAINED_URL_TABLE[FLAGS.pre_trained]
    # download_pretrained_model(pre_trained_model_url, backbone_dir)

    coco_img_url = COCODATA_URL_TABLE[FLAGS.dataset_type]
    coco_ann_url = COCODATA_URL_TABLE['annotations']
    download_dataset(coco_img_url, coco_ann_url, cocodata_dir)



if __name__ == '__main__':
    tf.app.run()
