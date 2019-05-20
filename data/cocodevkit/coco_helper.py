import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
import os
import cv2
import sys
import argparse
import gc
import tensorflow as tf



def draw_bbox(img, bbox):
    """
    draw bounding box.

    Args:
        img: image.
        bbox: bounding box. format => [x, y, width, height]
    Return:
        image drawed bounding box.
    """
    bbox = np.array(bbox, dtype='int32')
    x, y, width, height = bbox
    up_left = (x, y)
    bottom_right = (x + width, y + height)
    cv2.rectangle(img, up_left, bottom_right, (255, 0, 0), 2)

    return img

def draw_keypoints(img, keypoints):
    """
    draw keypoints.

    Args:
        img: image.
        keypoints: keypoints list. format => [x1, y1, v1, ...]
                   x -> x location.
                   y -> y location.
                   v -> if v is 0, this key has no location and no visible.
                        if v is 1, this key has location but no visible.
                        if v is 2, this key has location and visible.

    Return:
        image drawed keypoints.
    """
    key_list = [keypoints[offset:offset+3] for offset in range(0, len(keypoints), 3)]

    for key in key_list:
        if key[2] == 2:
            cv2.circle(img, (key[0], key[1]), 3, (0, 0, 255), -1)

    return img

def crop_and_relocate(img, bbox, keypoints):
    """
    crop and relocate keypoints.

    Args:
        img: image.
        bbox: bounding box. crop image based on this.
        keypoints: image keypoints locations.

    Return:
        cropped image and relocate keypoints.
    """
    relocate_keypoints = []
    x, y, width, height = np.array(bbox, dtype='int32')
    cropped_img = img[y:y+height, x:x+width]

    key_list = [keypoints[offset:offset+3] for offset in range(0, len(keypoints), 3)]

    for key in key_list:
        key_x, key_y, key_v = key
        relocate_key_x = key_x - x
        relocate_key_y = key_y - y

        relocate_keypoints.append(relocate_key_x)
        relocate_keypoints.append(relocate_key_y)
        relocate_keypoints.append(key_v)


    return cropped_img, relocate_keypoints

class CocoLoader():
    """
    for loading data from coco dataset.
    for create single person pose estimation label.

    Attributes:
        img_dir: image directory
        ann_file: annotation file path
        category: annotation category => for estimate keypoints, you should set ['person']
        KEYPOINTS_LABEL: keypoint label name. example) shoulder, elbow, nose ...
        SKELETON: connectivity list. example) [15, 5] mean that label 15 and label 5 is connected.
        cat_ids: category ids.
        img_ids: image ids. this image containing the category ids.
                example) when category ids = ['person'], image which related to img_ids has person.
        imgs: images list. each element has image and annotation.
        crop_imgs cropped images list. each element has image and keypoint.
    """
    def __init__(self, img_dir, ann_file, category=['person'], ):
        """
        constructor

        Args:
            data_dir : coco image directory.
            ann_file: annotation directory.
            category: category which is contained img you want to load.
                      for example)
                      If you want images containing 'person', 'dog', and 'sketeboard',
                      you must set ['person', 'dog', 'sketeboard'].
        """
        self._img_dir = img_dir
        self._ann_file = ann_file
        self._category = category

        # initialize COCO api from ann_file.
        self.coco = COCO(self._ann_file)

        # get keypoints label
        person_category = self.coco.loadCats(self.coco.getCatIds(['person']))[0]
        self._KEYPOINTS_LABEL = person_category['keypoints']
        self._SKELETON = person_category['skeleton']

        # load category id from variable 'category'
        self._cat_ids = sorted(self.coco.getCatIds(catNms=self._category))
        self._img_ids = sorted(self.coco.getImgIds(catIds=self._cat_ids))

        self._single_person_ann_ids = self.coco.getAnnIds(catIds=self._cat_ids, iscrowd=None)
        # 人のアノテーションを取り出す。
        self._single_person_anns = self.coco.loadAnns(self._single_person_ann_ids)

        self.keypoint_anns = []
        # キーポイントのアノテーションがされているのを取り出す
        for ann in self._single_person_anns:
            if ann['num_keypoints'] > 13:
                bbox = ann['bbox']
                if bbox[2] < 100 or bbox[3] < 100:
                    continue
                self.keypoint_anns.append(ann)
        print(len(self.keypoint_anns))

        # create image reader
        self.create_img_reader()





        # load image information and annotations from img ids.
        # print('load image information and annotations...')
        # self._load_imgs_and_anns(coco)

        # parse annotation and crop to single person image.
        # print('parse annotations and crop images...')
        # self._parse_and_crop()
        gc.collect()

        print('\ncomplete!!')
        print('length of single person images: {}'.format(len(self.keypoint_anns)))


    def __len__(self):
        return len(self.keypoint_anns)


    def get_img_inf(self, img_id):
        """
        get image information from img_id.

        Args:
            img_id: image id.
        Return:
            image information.
        """
        img_information = self.coco.loadImgs(img_id)[0]

        return img_information


    def create_img_reader(self):
        with tf.Graph().as_default():
            self._decode_path = tf.placeholder(dtype=tf.string)
            self._jpeg_file = tf.read_file(self._decode_path)
            self._session = tf.Session()

            self._decode = tf.image.decode_jpeg(self._jpeg_file,
                                                channels=3)

            # bbox => [top_left_x, top_left_y, width, height]
            self._bbox = tf.placeholder(dtype=tf.int32, shape=(4))

            self._cropped_img = tf.image.crop_to_bounding_box(
                image=self._decode,
                offset_height=self._bbox[1],
                offset_width=self._bbox[0],
                target_height=self._bbox[3],
                target_width=self._bbox[2]
            )


    def decode_image_and_crop(self, image_data, bbox):
        """
        Decodes the image data string.

        Args:
            image_data: string of image data.
            bbox: bounding box. [top_left_x, top_left_y, width, height].

        Return:
            Cropped image.

        Raises:
            ValueError:  value of image channels not supperted.
        """
        image = self._session.run(self._cropped_img,
                                  feed_dict={self._decode_path: image_data,
                                             self._bbox: bbox})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('the image channels not supported')

        return image


    def _load_imgs_and_anns(self, coco):
        """
        load images and annotations and save self._imgs.

        Args:
            coco: COCO api.

        Return:
            none.
        """
        self.imgs = []
        self.num_ann = 0
        # load image one by one from img ids.
        # and append to self._imgs.
        for i, img_id in enumerate(self._img_ids):
            img = coco.loadImgs(img_id)[0]
            # その画像内のpersonのアノテーションのみを取り出す
            # get annotations is specified category ids.
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self._cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            # keypointがあるアノテーションだけを入れるリスト
            have_key_anns = []
            for ann in anns:
                if ann['num_keypoints'] > 0:
                    have_key_anns.append(ann)
                    self.num_ann += 1

            temp_dict = {}
            temp_dict['img'] = img
            temp_dict['anns'] = have_key_anns
            self.imgs.append(temp_dict)
        print('num_ann: {}'.format(self.num_ann))
        exit()

    def _parse_and_crop(self):
        """
        parse annotation and crop to single person image.
        append self._crop_imgs.

        """
        self._crop_imgs = []
        num_imgs = len(self._imgs)
        for i, img in enumerate(self._imgs, start=1):
            img_path = img['img']['file_name']
            I = plt.imread(os.path.join(self._img_dir, img_path))
            # その画像のアノテーションを取得.
            # アノテーションは、その画像に複数のオブジェクトがあるので、
            # アノテーションも複数となる.
            anns = img['anns']
            num_anns = len(anns)
            # 一つ一つ、アノテーションを取り出し、キーポイントのアノテーションが
            # あるものに対しては、クロップしてキーポイントのロケーションを直す
            for j, ann in enumerate(anns, start=1):
                sys.stdout.write('\r{:5d}/{:2d}, img{:7d}/{:7d}'.format(j, num_anns, i, num_imgs))
                if ann['num_keypoints'] > 0:
                    bbox = ann['bbox']
                    keypoints = ann['keypoints']
                    tmp = np.copy(I)
                    cropped_img, relocate_keypoints = crop_and_relocate(tmp, bbox, keypoints)
                    temp_dict = {}
                    temp_dict['img'] = cropped_img
                    temp_dict['keypoint'] = relocate_keypoints
                    self._crop_imgs.append(temp_dict)
                    # For test
                    # res = self.draw_keypoints(cropped_img, relocate_keypoints)
                    # plt.imshow(res)
                    # plt.show()
        self._imgs = None


    def _draw_keypoints(self, img, keypoints):
        """
        draw keypoints.

        Args:
            img: image.
            keypoints: keypoints list. format => [x1, y1, v1, ...]
                       x -> x location.
                       y -> y location.
                       v -> if v is 0, this key has no location and no visible.
                            if v is 1, this key has location but no visible.
                            if v is 2, this key has location and visible.

        Return:
            image drawed keypoints.
        """
        result_img = np.copy(img)
        key_list = [keypoints[offset:offset+3] for offset in range(0, len(keypoints), 3)]
        print(len(key_list))

        # それぞれのキーポイントのつながりを可視化
        for connection in self._SKELETON:
            keypoint1 = key_list[connection[0]-1]
            keypoint2 = key_list[connection[1]-1]

            # if both keypoints has visible
            if keypoint1[2] == 2 and keypoint2[2] == 2:
                cv2.circle(result_img, (keypoint1[0], keypoint1[1]), 3, (0, 0, 255), -1)
                cv2.circle(result_img, (keypoint2[0], keypoint2[1]), 3, (0, 0, 255), -1)
                cv2.line(result_img, (keypoint1[0], keypoint1[1]), (keypoint2[0], keypoint2[1]), (0, 0, 255), 2)

        return result_img




if __name__ == '__main__':
    # define the dataset directory and coco dataset type.
    dataDir = './dataset'
    parser = argparse.ArgumentParser(description='This file contains coco dataset loader for single person pose estimation')
    parser.add_argument('--data_type',
                        help='coco datatype example) val2017',
                        default='val2017',
                        type=str)
    args = parser.parse_args()
    dataType = args.data_type
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
    loader = CocoLoader(os.path.join(dataDir, dataType), annFile)

    anns = loader.keypoint_anns

    for ann in anns:
        img_id = ann['image_id']
        img_information = loader.get_img_inf(img_id)

        img_path = os.path.join(
            dataDir, 'val2017', img_information['file_name']
        )

        img = plt.imread(img_path)

        keypoint = ann['keypoints']
        res = loader._draw_keypoints(img, keypoint)
        plt.imshow(res)
        plt.show()
