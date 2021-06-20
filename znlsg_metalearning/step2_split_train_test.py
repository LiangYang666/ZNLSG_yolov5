#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :step2_split_train_val.py
# @Time      :2021/6/18 下午1:44
# @Author    :Yangliang
import os
import shutil
import random
from znlsg_metalearning.step1_crop_image import crop_dir, crop_all_dir

if __name__ == "__main__":
    crop_train_dir = os.path.join(crop_dir, 'train')
    crop_test_dir = os.path.join(crop_dir, 'test')
    if not os.path.exists(crop_train_dir):
        os.mkdir(crop_train_dir)
    if not os.path.exists(crop_test_dir):
        os.mkdir(crop_test_dir)
    shutil.rmtree(crop_train_dir)
    shutil.rmtree(crop_test_dir)

    all_cat_dir = os.listdir(crop_all_dir)
    random.shuffle(all_cat_dir)
    ratio = 0.7
    n = int(ratio*len(all_cat_dir))
    train_cat_dir = all_cat_dir[:n]
    test_cat_dir = all_cat_dir[n:]
    for dir in train_cat_dir:
        shutil.copytree(os.path.join(crop_all_dir, dir), os.path.join(crop_train_dir, dir))

    for dir in test_cat_dir:
        shutil.copytree(os.path.join(crop_all_dir, dir), os.path.join(crop_test_dir, dir))
