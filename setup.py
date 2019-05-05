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
import threading
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum('pre_trained', 'resnet_v1_101',
                         ['vgg16', 'resnet_v1_101', 'resnet_v1_50'],
                         'Pre-trained model name you wanna download (vgg16, resnet_v1_101, resnet_v1_50)')

tf.app.flags.DEFINE_enum('dataset_type', 'val', ['val', 'train'],
                         'coco dataset type (val or train)')

PRE_TRAINED_URL_TABLE = {
    'vgg16': 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
    'resnet_v1_50': 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
    'resnet_v1_101': 'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
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
    print('\ndownload pretrained model.')
    if os.path.exists(save_path):
        print('{} is already exist.'.format(save_path))
        extract_tar(save_path, save_dir)

    else:
        print('downloading tar file in {}'.format(url))
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            print('Complete downloading tar file!!')

            extract_tar(save_path, save_dir)




def download_dataset(img_url, ann_url, save_dir):
    """
    download cocodataset.

    Args:
        img_url: image dataset url.
        ann_url: annotation files url.
        save_dir: save directory to which you want to save.

    """

    save_img_path = os.path.join(save_dir, img_url.split('/')[-1])
    save_ann_path = os.path.join(save_dir, ann_url.split('/')[-1])

    print('\ndownload cocodataset')
    print('download images')

    if os.path.exists(save_img_path):
        # if zip file already exist.
        print(save_img_path)
        extract_zip(save_img_path, save_dir)
    else:
        print('downloading zip from {}'.format(img_url))
        with urllib.request.urlopen(img_url) as response, open(save_img_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            print('complete downloading!')

        extract_zip(save_img_path, save_dir)

    print('download annotations')
    if os.path.exists(save_ann_path):
        " if zip file already exist. "
        extract_zip(save_ann_path, save_dir)
    else:
        print('downloading zip from {}'.format(ann_url))
        with urllib.request.urlopen(ann_url) as response, open(save_ann_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            print('complete downloading!')

        extract_zip(save_ann_path, save_dir)


def extract_tar(file_path, save_dir=None):
    """
    extract files in tar file.

    Args:
        file_path: path of tar file.
        save_dir: directory where you want to save extracted files.
    """
    if save_dir is None:
        save_dir = os.getcwd()

    with tarfile.open(file_path, 'r:gz') as tar:
        tar_members = tar.getmembers()

        for member in tar_members:
            tar.extract(member=member, path=save_dir)
            print('extractiong {} succeed!'.format(member.name))


def extract_zip(file_path, save_dir=None):
    """
    extract files in zip file.

    Args:
        file_path: path of zip file.
        save_dir: directory where you want to save extracted files.

    """
    if save_dir is None:
        save_dir = os.getcwd()

    # oepn zip file as zip.
    with zipfile.ZipFile(file_path) as zip:
        # get files name in zip file.
        name_list = zip.namelist()
        num_files = len(name_list)
        for i, name in enumerate(name_list, start=1):
            if os.path.exists(os.path.join(save_dir, name)):
                sys.stdout.write('\rextract       name: {}'.format(name))
            else:
                zipinfo = zip.getinfo(name)
                zip.extract(zipinfo, path=save_dir)
                sys.stdout.write('\rextract {}/{} name: {}'.format(i, num_files, name))
        sys.stdout.write('\nextract all done!\n')


def main(argv):
    """
    main function.
    execute at first.
    """

    backbone_dir = 'backbone_checkpoints'
    cocodata_dir = os.path.join('data', 'cocodevkit', 'dataset')

    make_dir('checkpoints')
    make_dir(backbone_dir)
    make_dir(cocodata_dir)

    pre_trained_model_url = PRE_TRAINED_URL_TABLE[FLAGS.pre_trained]
    download_pretrained_model(pre_trained_model_url, backbone_dir)

    coco_img_url = COCODATA_URL_TABLE[FLAGS.dataset_type]
    coco_ann_url = COCODATA_URL_TABLE['annotations']
    download_dataset(coco_img_url, coco_ann_url, cocodata_dir)

    print('\nall done!!!!')

if __name__ == '__main__':
    tf.app.run()
