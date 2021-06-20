#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :step3_mesure_dist.py
# @Time      :2021/6/6 下午1:01
# @Author    :Yangliang
import json
import types
import sys
sys.path.append('..')
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import ipdb
import torchvision
from torch.autograd import Variable
from Network_generate_embedding import resnet18_embedding
from datasets import ZNLSG_library_imgs_dataset, ZNLSG_eval_imgs_dataset, ZNLSG_bboxes_embeddings_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from znlsg_tolls.tool2_watch_img import ZNLSG_COCO
from measure.step1_generate_library_embedding import data_dir
import re


def get_with_cat_dist_mean(a, b, n=30):
    # return torch.sort(euclidean_dist(a, b), dim=1, descending=True).values[:, :n].mean(dim=1)
    return (euclidean_dist(a, b).mean(dim=1))


def euclidean_dist(a, b):
    n = a.shape[0]
    m = b.shape[0]
    '''while  n_way=30 support=1 query=15
            m -> 30
            n -> 450
            !a.shape -> torch.Size([450, 1600])
            !b.shape -> torch.Size([30, 1600])
    '''
    a = a.unsqueeze(1).expand(n, m, -1)     # It just create a new view on existing tensor, not copy tensor
    b = b.unsqueeze(0).expand(n, m, -1)

    '''while  n_way=30 support=1 query=15
            !a.shape -> torch.Size([450, 30, 1600])
            !b.shape -> torch.Size([450, 30, 1600])
            logits.shape -> torch.Size([450, 30])
    '''
    logits = ((a - b)**2).sum(dim=2)

    return logits


if __name__ == "__main__":
    if len(sys.argv) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Step3 *-*-*-*-*-*-*-*-*-*')
    print('\tMeasure and predict the categories')
    eval_bboxes_embeddings_dir = os.path.join(data_dir, 'a_embeddings', 'bboxs')
    library_cats_embeddings_dir = os.path.join(data_dir, 'b_embeddings', 'cats')
    library_annotations_file = os.path.join(data_dir, "b_annotations.json")
    eval_annotations_file = os.path.join(data_dir, "a_annotations.json")
    pred_annotations_file = os.path.join(data_dir, "pred_a_annotations.json")
    library_coco = ZNLSG_COCO(library_annotations_file)
    library_cats_embeddings_dic = {}
    library_cats_embeddings_files = sorted(os.listdir(library_cats_embeddings_dir), key=lambda x: int(x.split('.')[0]))
    for file in library_cats_embeddings_files:
        cat_embeddings = torch.load(os.path.join(library_cats_embeddings_dir, file)).to(device)
        library_cats_embeddings_dic[int(file.split('.')[0])] = cat_embeddings
    datasets = ZNLSG_bboxes_embeddings_datasets(eval_bboxes_embeddings_dir)
    batch_size = 100
    dataloader = DataLoader(datasets, batch_size, shuffle=False, num_workers=10)
    pbar = tqdm(range(len(datasets)))
    cats_ids = sorted(list(library_cats_embeddings_dic.keys()))
    cats_ids_t = torch.tensor(cats_ids)
    imgs_catid_cat_dic = {}      # Stores the corresponding catid and cat name of all small slice bboxes images

    for datas, img_names in dataloader:
        datas = datas.to(device)
        pbar.update(datas.shape[0])
        datas_cats_dist_l = []
        for key in cats_ids:
            cat_embeddings = library_cats_embeddings_dic[key]
            cat_dist = get_with_cat_dist_mean(datas, cat_embeddings)
            datas_cats_dist_l.append(cat_dist)
        datas_cats_dist_t = torch.stack(datas_cats_dist_l, dim=1)
        min_cat_dist_lindex = torch.argmin(datas_cats_dist_t, dim=1)
        min_cat_dist_catids = cats_ids_t[min_cat_dist_lindex]
        library_coco.loadCats(library_coco.getCatIds())
        for i in range(datas.shape[0]):
            img_name = img_names[i]
            catid = min_cat_dist_catids[i].item()
            cat = library_coco.loadCats(catid)[0]['name']
            imgs_catid_cat_dic[img_name] = (catid, cat) # Store the corresponding catid and catName  of the small sliced image, the key is small sliced image embedding name
    pbar.close()
    eval_coco = ZNLSG_COCO(eval_annotations_file)
    pred_annotations_json = {'images': list(eval_coco.imgs.values()), 'categories': list(eval_coco.cats.values()), 'annotations': []}
    # pred_annotations_json = {'images': list(eval_coco.imgs.values()), 'annotations': []}
    bboxes2imgs_dic = {}            # Store the corresponding slice bboxes images of all original images
    for bboxes_img in imgs_catid_cat_dic.keys():   # Store the corresponding small slice bboxes images of all original images
        img_original_name_pre = bboxes_img.split('_bbox')[0]
        if img_original_name_pre not in bboxes2imgs_dic.keys():
            bboxes2imgs_dic[img_original_name_pre] = [bboxes_img]
        else:
            bboxes2imgs_dic[img_original_name_pre].append(bboxes_img)
    ann_id = 0
    for img_info in pred_annotations_json['images']:
        img_name = img_info['file_name']
        image_id = img_info['id']
        img_name_pre = img_name.split('.')[0]
        for small_bbox_img_emb_name in bboxes2imgs_dic[img_name_pre]:    # Process the bbox information of each small bboxes img in a complete img
            category_id, category_name = imgs_catid_cat_dic[small_bbox_img_emb_name]
            # _, xxyy = small_bbox_img_emb_name.split('_bbox_')
            try:
                _, xxyy, conf, pt = re.split('_bbox_|_conf_|\.', small_bbox_img_emb_name)
            except:
                print('Warning !\n'*3)
                print('File name wrong', small_bbox_img_emb_name)
                print('Warning !\n'*3)
                continue
            xxyy = xxyy.split('_')
            xxyy = [int(x) for x in xxyy]
            conf = int(conf)/100
            x1, y1, x2, y2 = xxyy
            w = x2-x1
            h = y2-y1
            annotation_info = {'image_id': image_id, 'id': ann_id, 'bbox': [x1, y1, w, h], 'category_id': category_id, 'score': conf, "area": w*h, "iscrowd": 0}
            # annotation_info = {'image_id': image_id, 'bbox': [x1, y1, w, h], 'category_id': category_id, 'score': 0.9}
            ann_id += 1
            pred_annotations_json['annotations'].append(annotation_info)
    with open(pred_annotations_file, 'w') as f:
        json.dump(pred_annotations_json, f)
        print(f"Writting to {pred_annotations_file}")
