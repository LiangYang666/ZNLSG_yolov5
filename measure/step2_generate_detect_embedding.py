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
from datasets import ZNLSG_library_imgs_dataset, ZNLSG_eval_imgs_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ =="__main__":
    if len(sys.argv) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    data_dir = "../../../data/cssjj/train"
    labels_dir = os.path.join(data_dir, 'a_labels')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emebeddings_bboxs_dir = os.path.join(data_dir, 'a_embeddings', 'bboxs')
    os.makedirs(emebeddings_bboxs_dir, exist_ok=True)

    batch_size = 20

    model = resnet18_embedding()
    model.to(device)
    datasets = ZNLSG_eval_imgs_dataset(labels_dir)
    dataloader = DataLoader(datasets, batch_size, shuffle=False, num_workers=8)
    model.eval()
    have_save = {}
    print(f'Starting generate all emebeddings \n'
          f'\tfrom {labels_dir}\n'
          f'\t to  {emebeddings_bboxs_dir}:')
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar)
    total = 0
    with torch.no_grad():
        for i, (imgs, img_names, bboxes) in pbar:
            imgs = imgs.to(device)
            outs = model(imgs)
            # ipdb.set_trace()
            # print(i, len(category_ids))
            for j in range(len(img_names)):
                img_name = img_names[j]
                x1, y1, x2, y2 = bboxes[j]
                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                out = outs[j]
                torch.save(out.cpu(),
                           os.path.join(emebeddings_bboxs_dir, f'{img_name}_bbox_{x1}_{y1}_{x2}_{y2}.pt'))
                total += 1

    print(f'Done! \n'
          f'\tAll saved to {emebeddings_bboxs_dir}\n'
          f'\t----Toral {total} box emebeddings')




