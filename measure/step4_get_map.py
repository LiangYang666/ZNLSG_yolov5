#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :step4_get_map.py  calculate the map
# @Time      :2021/6/6 下午9:27
# @Author    :Yangliang
import json
import sys
sys.path.append('..')
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from measure.step3_search_k_2pred import K

import os

if __name__ == "__main__":
    data_dir = "../../../data/cssjj/train"
    eval_annotations_file = os.path.join(data_dir, "a_annotations.json")
    pred_annotations_file = os.path.join(data_dir, f"pred_a_annotations_top{K}.json")
    print(f'Eval the {pred_annotations_file}')
    with open(pred_annotations_file, 'r') as f:
        pred_annotations = json.load(f)
    eval_coco = COCO(eval_annotations_file)
    # pred_coco = eval_coco.loadRes(pred_annotations['annotations'])
    pred_coco = COCO(pred_annotations_file)
    eval = COCOeval(eval_coco, pred_coco, iouType='bbox')
    eval.evaluate()
    eval.accumulate()
    print('----------------------------------------')
    eval.summarize()