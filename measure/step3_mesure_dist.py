#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :step3_mesure_dist.py
# @Time      :2021/6/6 下午1:01
# @Author    :Yangliang
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
from datasets import ZNLSG_library_imgs_dataset, ZNLSG_eval_imgs_dataset, ZNLSG_bboxes_emebeddings_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def euclidean_dist(a, b):
    ipdb.set_trace()
    n = a.shape[0]
    m = b.shape[0]

    '''while  n_way=30 support=1 query=15
            m -> 30
            n -> 450
            !a.shape -> torch.Size([450, 1600])
            !b.shape -> torch.Size([30, 1600])
    '''
    a = a.unsqueeze(1).expand(n, m, -1) # It just create a new view on existing tensor, not copy tensor
    b = b.unsqueeze(0).expand(n, m, -1)

    '''while  n_way=30 support=1 query=15
            !a.shape -> torch.Size([450, 30, 1600])
            !b.shape -> torch.Size([450, 30, 1600])
            logits.shape -> torch.Size([450, 30])
    '''
    logits = -((a - b)**2).sum(dim=2)

    return logits


if __name__ == "__main__":
    if len(sys.argv) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = "../../../data/cssjj/train"
    eval_bboxes_embeddings_dir = os.path.join(data_dir, 'a_embeddings', 'bboxs')
    library_cats_embeddings_dir = os.path.join(data_dir, 'b_embeddings', 'cats')
    library_cats_embeddings = []
    library_cats_embeddings_files = sorted(os.listdir(library_cats_embeddings_dir), key=lambda x: int(x.split('.')[0]))
    for file in library_cats_embeddings_files:
        cat_embeddings = torch.load(os.path.join(library_cats_embeddings_dir, file)).to(device)
        library_cats_embeddings.append(cat_embeddings)
    datasets = ZNLSG_bboxes_emebeddings_datasets(eval_bboxes_embeddings_dir)
    batch_size = 100
    dataloader = DataLoader(datasets, batch_size, shuffle=False, num_workers=10)
    for data, img_name in tqdm(dataloader):




