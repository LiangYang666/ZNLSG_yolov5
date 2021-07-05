#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :step1_crop_image.py
# @Time      :2021/6/18 上午8:52
# @Author    :Yangliang
import os
import json
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from tqdm import tqdm
import re

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle


from znlsg_tolls.tool2_watch_img import ZNLSG_COCO
from path import data_dir, cs_train_dir, train_meta_learning_dir

crop_dir = os.path.join(train_meta_learning_dir, 'train_crop_images')
crop_all_dir = os.path.join(train_meta_learning_dir, 'train_crop_images', 'b_emebeddings_all')

if __name__ == "__main__":
    train_a_annotations_file = os.path.join(data_dir, "cssjj/train/b_annotations.json")
    train_a_coco = ZNLSG_COCO(train_a_annotations_file)
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    if not os.path.exists(crop_all_dir):
        os.makedirs(crop_all_dir)
    for imageid in tqdm(train_a_coco.getImgIds()):
        annotationIds = train_a_coco.getAnnIds(imgIds=imageid)
        anns = train_a_coco.loadAnns(annotationIds)
        imgInfo = train_a_coco.loadImgs(imageid)[0]
        imgName = imgInfo['file_name']
        imgFile = os.path.join(train_a_coco.imgs_dir, imgInfo['file_name'])
        img = cv2.imread(imgFile)
        if imgName.split('.').__len__() != 2:
            continue

        for ann in anns:
            category_id = ann['category_id']
            category_info = train_a_coco.loadCats(category_id)[0]
            category_name = category_info['name']
            [bbox_x1, bbox_y1, bbox_w, bbox_h] = ann['bbox']
            a = re.findall('[a-zA-Z0-9]+', category_name)
            category_name_p = ''.join(a)
            crop_name = f'catid_{category_id}_cat_{category_name_p}_bbox_{bbox_x1}_{bbox_y1}_{bbox_w}_{bbox_h}'+imgName
            cat_dir = os.path.join(crop_all_dir, f'cat{category_id}_{category_name_p}')
            if not os.path.exists(cat_dir):
                os.mkdir(cat_dir)
            crop = img[bbox_y1:bbox_h+bbox_y1+1, bbox_x1:bbox_w+bbox_x1+1]
            crop_path = os.path.join(cat_dir, crop_name)
            cv2.imwrite(crop_path, crop)



        



