import types

import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import ipdb
import torchvision
from torch.autograd import Variable
from Network_generate_embedding import resnet18_embedding, prototypical_net
from datasets import ZNLSG_library_imgs_dataset
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import shutil

batch_size = 20
data_dir = "../../../data/cssjj/train"

if __name__ =="__main__":
    if len(sys.argv) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    print('Step1 *-*-*-*-*-*-*-*-*-*')
    print('\tGenerate library embeddings')
    b_annotations = os.path.join(data_dir, 'b_annotations.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_bboxs_dir = os.path.join(data_dir, 'b_embeddings', 'bboxs')
    embeddings_cats_dir = os.path.join(data_dir, 'b_embeddings', 'cats')

    if os.path.exists(embeddings_bboxs_dir):
        shutil.rmtree(embeddings_bboxs_dir)
    if os.path.exists(embeddings_cats_dir):
        shutil.rmtree(embeddings_cats_dir)
    os.makedirs(embeddings_bboxs_dir, exist_ok=False)
    os.makedirs(embeddings_cats_dir, exist_ok=False)

    model = prototypical_net()
    model.to(device)
    datasets = ZNLSG_library_imgs_dataset(annotations_file=b_annotations)
    dataloader = DataLoader(datasets, batch_size, shuffle=False, num_workers=8)
    model.eval()
    have_save = {}
    print(f'Starting generate all embeddings \n'
          f'\tfrom {b_annotations}\n'
          f'\t to  {embeddings_bboxs_dir}:')
    # pbar = enumerate(dataloader)
    # pbar = tqdm(pbar)
    pbar = tqdm(total=len(datasets))
    total = 0
    with torch.no_grad():
        for i, (imgs, category_ids) in enumerate(dataloader):
            pbar.update(imgs.shape[0])
            imgs = imgs.to(device)
            outs = model(imgs)
            # ipdb.set_trace()
            # print(i, len(category_ids))
            for j in range(len(category_ids)):
                category_id = category_ids[j].item()
                out = outs[j]
                if category_id not in have_save.keys():
                    have_save[category_id] = 0
                else:
                    have_save[category_id] += 1

                torch.save(out.cpu(), os.path.join(embeddings_bboxs_dir, f'cat_{category_id}_bbox_{have_save[category_id]}.pt'))
                total += 1
                # print(f'cat_{category_id}_bbox_{have_save[category_id]}.pt')

    pbar.close()

    print(f'\t----Total {total} box embeddings')
    print(f'Starting move all embeddings to corresponding category dir\n'
          f'\tfrom {embeddings_bboxs_dir}\n'
          f'\t to  {embeddings_cats_dir}:')
    total = 0
    emebeddings_cats_dic = {}
    emebeddings_bboxs_list = os.listdir(embeddings_bboxs_dir)
    for name in emebeddings_bboxs_list:
        category_id = int(name.split('_')[1])
        if category_id not in emebeddings_cats_dic.keys():
            emebeddings_cats_dic[category_id] = [name]
        else:
            emebeddings_cats_dic[category_id].append(name)
    for category_id in tqdm(emebeddings_cats_dic.keys()):
        emebeddings_cat_list = []
        for name in emebeddings_cats_dic[category_id]:
            # ipdb.set_trace()
            emebeddings_each = torch.load(os.path.join(embeddings_bboxs_dir, name))
            emebeddings_cat_list.append(emebeddings_each)
        emebeddings_cat = torch.stack(emebeddings_cat_list, dim=0)
        torch.save(emebeddings_cat, os.path.join(embeddings_cats_dir, f'{category_id}.pt'))
        total += 1

    print(f'Done! \n'
          f'\tAll saved to {embeddings_cats_dir}\n'
          f'\t----Total {total} category embeddings\n')








