#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool6_watchimg_yolo.py
# @Time      :2021/6/20 上午9:59
# @Author    :Yangliang

import os
import json
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from measure.datasets import ZNLSG_eval_imgs_dataset


data_dir = "../../../data/cssjj/test"

if __name__ == "__main__":
    labels_dir = os.path.join(data_dir, 'a_labels_yolov5x_40')
    datasets = ZNLSG_eval_imgs_dataset(labels_dir)
    for imgname in datasets.imgs_names:
        imgFile = os.path.join(datasets.imgs_dir, imgname)
        img = plt.imread(imgFile)
        labels = datasets.get_one_image_bboxes(imgname)
        plt.imshow(img)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        width = img.shape[1]
        height = img.shape[0]
        for label in labels:
            img_name, x_f, y_f, w_f, h_f, conf = label
            x, w = x_f * width, w_f * width
            y, h = y_f * height, h_f * height
            x1, x2 = x - w / 2, x + w / 2
            y1, y2 = y - h / 2, y + h / 2
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2 + 0.8), int(y2 + 0.8)
            [bbox_x1, bbox_y1, bbox_w, bbox_h] = x1, y1, w, h
            c = 'r'
            ax.add_patch(
                plt.Rectangle((bbox_x1, bbox_y1), bbox_w, bbox_h, color=c, fill=False, linewidth=2))
            ax.text(bbox_x1, bbox_y1, str(conf), fontsize=10, color='white',
                    bbox={'facecolor': c, 'alpha': 0.5})
            # ax.text(bbox_x, bbox_y-3, category_name, fontsize=16, color=c)

        plt.pause(0.01)
        key_press = 0
        while not key_press:
            key_press = plt.waitforbuttonpress()
        # plt.waitforbuttonpress()
        plt.cla()




