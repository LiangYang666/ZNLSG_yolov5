#%%

import json
import os
import random
import sys
sys.path.append('..')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw
from znlsg_tolls.tool2_watch_img import ZNLSG_COCO
import ipdb


# json_file = '../data/mydata/all.json'

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
if mean[0] > 1 or mean[1] > 1 or mean[2] > 1:
    print('One or more of the subtract mean values were above 1, dividing by 255...')
    mean /= 255

if std[0] > 1 or std[1] > 1 or std[2] > 1:
    print('One or more of the subtract std values were above 1, dividing by 255...')
    std /= 255
print('Normalizing by mean of %.4f, %.4f, %.4f' % (mean[0], mean[1], mean[2]))
print('Normalizing by std of %.4f, %.4f, %.4f' % (std[0], std[1], std[2]))

normalize = transforms.Normalize(mean=mean, std=std)
img_size = 124
mytrans = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    # transforms.RandomHorizontalFlip(),
    # transforms.CenterCrop(64),
    transforms.ToTensor(),
    normalize
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
])
mytrans_lib = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=360),
    transforms.ToTensor(),
    normalize
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
])


class ZNLSG_eval_imgs_dataset(Dataset):
    def __init__(self, labels_dir='a_labels'):
        assert 'a_labels' in labels_dir
        # if 'test' in labels_dir:
        #     self.line_length = 6
        # else:
        #     self.line_length = 5

        self.line_length = 6
        self.labels_dir = labels_dir
        self.data_dir = os.path.dirname(labels_dir)
        self.imgs_dir = os.path.join(self.data_dir, 'a_images')
        self.imgs_names = os.listdir(self.imgs_dir)
        self.bboxs_labels = self.get_all_bboxes(self.imgs_names)

    def get_all_bboxes(self, imgs_names):
        all_bboxes = []
        for name in imgs_names:
            all_bboxes += self.get_one_image_bboxes(name)
        return all_bboxes


    def get_one_image_bboxes(self, img_name):
        yolo_txt_name = img_name.split('.')[0] + '.txt'
        with open(os.path.join(self.labels_dir, yolo_txt_name), 'r') as f:
            lines = f.readlines()
        boxxes = []
        for line in lines:
            ll = line.split()
            assert ll.__len__() == self.line_length
            if self.line_length==5:
                _, x, y, w, h = ll
                conf = 1.
            else:
                _, x, y, w, h, conf = ll
            boxxes.append([img_name, float(x), float(y), float(w), float(h), float(conf)])
        return boxxes

    def __getitem__(self, index):
        img_name, x_f, y_f, w_f, h_f, conf = self.bboxs_labels[index]
        img = Image.open(os.path.join(self.imgs_dir, img_name))
        width = img.width
        height = img.height
        x, w = x_f*width,  w_f*width
        y, h = y_f*height, h_f*height
        x1, x2 = x - w/2, x + w/2
        y1, y2 = y - h/2, y + h/2
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2+0.8), int(y2+0.8)
        img = img.crop((x1, y1, x2, y2))

        img = mytrans(img)
        return img, img_name, torch.tensor((x1, y1, x2, y2, conf))

    def __len__(self):
        return len(self.bboxs_labels)



class ZNLSG_library_imgs_dataset(Dataset):
    def __init__(self, annotations_file='b_annotations'):
        self.coco = ZNLSG_COCO(annotations_file)
        self.data_dir = self.coco.data_dir
        self.imgs_dir = self.coco.imgs_dir
        self.bbox_AnnIds = self.coco.getAnnIds()

    def __getitem__(self, id):
        annotation_id = self.bbox_AnnIds[id]
        annotation_info = self.coco.anns[annotation_id]
        image_id = annotation_info['image_id']
        bbox = annotation_info['bbox']
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        category_id = annotation_info['category_id']
        image_name = self.coco.loadImgs(image_id)[0]['file_name']
        img_path = os.path.join(self.imgs_dir, image_name)
        img = Image.open(img_path)
        img = img.crop((x1, y1, x2, y2))

        img = mytrans_lib(img)
        return img, category_id

    def __len__(self):
        return len(self.bbox_AnnIds)


class ZNLSG_library_aug_imgs_dataset(Dataset):
    def __init__(self, annotations_file='b_annotations', embedding_each_n=100):
        # if less than 100 then data augment to 100 for each category
        self.coco = ZNLSG_COCO(annotations_file)
        self.data_dir = self.coco.data_dir
        self.imgs_dir = self.coco.imgs_dir
        self.bbox_AnnIds = self.coco.getAnnIds()
        self.embedding_each_n = embedding_each_n
        self.new_AnnsIds = []
        self.genearte_aug_AnnsIds()

    def genearte_aug_AnnsIds(self):
        for catid in self.coco.getCatIds():
            annids = self.coco.getAnnIds(catIds=catid)
            num = len(annids)
            if num == 0:
                continue
            elif num < self.embedding_each_n:
                self.new_AnnsIds += annids*(self.embedding_each_n//num)
                random.shuffle(annids)
                self.new_AnnsIds += annids[:self.embedding_each_n % num]
            else:
                self.new_AnnsIds += annids

    def __getitem__(self, id):
        annotation_id = self.new_AnnsIds[id]
        annotation_info = self.coco.anns[annotation_id]
        image_id = annotation_info['image_id']
        bbox = annotation_info['bbox']
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        category_id = annotation_info['category_id']
        image_name = self.coco.loadImgs(image_id)[0]['file_name']
        img_path = os.path.join(self.imgs_dir, image_name)
        img = Image.open(img_path)
        img = img.crop((x1, y1, x2, y2))

        img = mytrans_lib(img)
        return img, category_id

    def __len__(self):
        return len(self.new_AnnsIds)



class ZNLSG_bboxes_embeddings_datasets(Dataset):
    def __init__(self, embeddings_dir):
        self.embeddings_dir = embeddings_dir
        self.eval_bboxes_embeddings_files = sorted(os.listdir(embeddings_dir))

    def __getitem__(self, id):
        file = self.eval_bboxes_embeddings_files[id]
        bbox_embedding = torch.load(os.path.join(self.embeddings_dir, file))
        return bbox_embedding, file

    def __len__(self):
        return len(self.eval_bboxes_embeddings_files)



class EpisodicBatchSampler(Sampler):
    def __init__(self, labels, n_episodes, n_way, n_samples):
        '''
        Sampler that yields batches per n_episodes without replacement.
        Batch format: (c_i_1, c_j_1, ..., c_n_way_1, c_i_2, c_j_2, ... , c_n_way_2, ..., c_n_way_n_samples)

        Args:
            label: List of sample labels (in dataloader loading order)
            n_episodes: Number of episodes or equivalently batch size
            n_way: Number of classes to sample
            n_samples: Number of samples per episode (Usually n_query + n_support)
        '''

        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_samples = n_samples

        labels = np.array(labels)
        self.samples_indices = []
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.samples_indices.append(ind)

        if self.n_way > len(self.samples_indices):
            raise ValueError('Error: "n_way" parameter is higher than the unique number of classes')

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.samples_indices))[:self.n_way]
            for c in classes:
                l = self.samples_indices[c]
                pos = torch.randperm(len(l))[:self.n_samples]
                batch.append(l[pos])
            # torch.stack(batch).shape -> torch.Size([30, 16]) n_way=30(classes number)  n_samples = n_query+n_support = 15+1
            # ipdb.set_trace()
            # torch.stack(batch).t().shape -> torch.Size([16, 30])
            batch = torch.stack(batch).t().reshape(-1)
            # batch.shape -> torch.Size([480])
            yield batch
class GFKDtrack_dataset(Dataset):
    def __init__(self, data_dir, json_file):
        # imgs = os.listdir(os.path.join(data_dir, 'images'))
        self.labels, img_names = self.get_labels(os.path.join(data_dir, json_file))
        self.imgs = [os.path.join(data_dir, 'images', x) for x in img_names]
        # self.labels =

    def get_labels(self, json_file):
        with open(json_file, 'r') as f:
            all_label = json.load(f)
        labels = []
        img_names = list(all_label.keys())

        for imgn in img_names:
            name = imgn.split('/')[-1]
            # assert name in all_label.keys()
            xy = all_label[name]['xy']
            x = int(xy[0])
            y = int(xy[1])
            labels.append([x, y])

            # print(key, ': ', x, y)
        return labels, img_names
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        x, y = self.labels[index]
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = Image.open(img_path)

        x = x / img.width
        y = y / img.height

        label = torch.Tensor([x, y]).float()
        mytrans = transforms.ToTensor()
        img = mytrans(img)
        return img, label
#%%


if __name__ == "__main__":
    #%%
    data_dir = "../../../data/cssjj/test"
    annotation_file = os.path.join(data_dir, 'b_annotations.json')
    znlsg_library = ZNLSG_library_imgs_dataset(annotation_file)
    znlsg_library_cats_num_dic = {}
    for catid in znlsg_library.coco.getCatIds():
        annids = znlsg_library.coco.getAnnIds(catIds=catid)


        znlsg_library_cats_num_dic['catid'+str(catid)] = len(annids)
    print(znlsg_library_cats_num_dic)
    import matplotlib.pyplot as plt
    plt.plot(list(znlsg_library_cats_num_dic.keys()), list(znlsg_library_cats_num_dic.values()))
    plt.show()



    # to_image = transforms.ToPILImage()
    # img, category_id = znlsg_library[1000]
    # img = to_image(img)
    # img.show()
    #
    # print(category_id, znlsg_library.coco.loadCats(category_id)[0]['name'])

    # labels_dir = os.path.join(data_dir, 'a_labels')
    # znlsg_eval_datasets = ZNLSG_eval_imgs_dataset(labels_dir)
    # a = znlsg_eval_datasets[0]
    # to_image = transforms.ToPILImage()
    # img, img_name, bbox = znlsg_eval_datasets[1000]
    # img = to_image(img)
    # img.show()




