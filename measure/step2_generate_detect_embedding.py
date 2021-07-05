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
from Network_generate_embedding import resnet18_embedding, prototypical_net
from datasets import ZNLSG_library_imgs_dataset, ZNLSG_eval_imgs_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

from measure.step1_generate_aug_library_embedding import batch_size, data_dir

if __name__ =="__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    if len(sys.argv) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


    print('Step2 *-*-*-*-*-*-*-*-*-*')
    print('\tGenerate detected embeddings')
    labels_dir = os.path.join(data_dir, 'a_labels_yolov5x_40_p5')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_bboxs_dir = os.path.join(data_dir, 'a_embeddings', 'bboxs')
    if os.path.exists(embeddings_bboxs_dir):
        shutil.rmtree(embeddings_bboxs_dir)
    os.makedirs(embeddings_bboxs_dir, exist_ok=False)

    model = prototypical_net()
    model.to(device)
    datasets = ZNLSG_eval_imgs_dataset(labels_dir)
    dataloader = DataLoader(datasets, batch_size, shuffle=False, num_workers=8)
    model.eval()
    have_save = {}
    print(f'Starting generate all embeddings \n'
          f'\tfrom {labels_dir}\n'
          f'\t to  {embeddings_bboxs_dir}:')
    pbar = tqdm(total=len(datasets))
    total = 0
    with torch.no_grad():
        for i, (imgs, img_names, bboxes) in enumerate(dataloader):
            pbar.update(imgs.shape[0])
            imgs = imgs.to(device)
            outs = model(imgs)
            # ipdb.set_trace()
            # print(i, len(category_ids))
            for j in range(len(img_names)):
                img_name = img_names[j]
                x1, y1, x2, y2, conf = bboxes[j]
                x1, y1, x2, y2, conf = x1.item(), y1.item(), x2.item(), y2.item(), conf.item()
                out = outs[j]
                img_name_pre = img_name.split('.')[0]
                conf_int = int(conf*100)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                torch.save(out.cpu(),
                           os.path.join(embeddings_bboxs_dir, f'{img_name_pre}_bbox_{x1}_{y1}_{x2}_{y2}_conf_{conf_int}.pt'))
                total += 1
    pbar.close()

    print(f'Done! \n'
          f'\tAll saved to {embeddings_bboxs_dir}\n'
          f'\t----Total {total} box embeddings')




