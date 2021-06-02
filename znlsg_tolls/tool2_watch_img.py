import os
import json
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle



data_dir = "../../../data"
train_a_annotations_file = os.path.join(data_dir, "cssjj/train/a_annotations.json")
train_b_annotations_file = os.path.join(data_dir, "cssjj/train/b_annotations.json")

test_a_annotations_file = os.path.join(data_dir, "cssjj/test/a_annotations.json")
test_b_annotations_file = os.path.join(data_dir, "cssjj/test/b_annotations.json")

train_a_imgs_dir = os.path.join(data_dir, 'cssjj/train/a_images')

class ZNLSG_COCO(COCO):

    def __init__(self, annotation_file=None):
        COCO.__init__(self, annotation_file)
        self.colors = self.color_list()
        self.annotation_file = annotation_file
        self.annos_dir = os.path.dirname(annotation_file)
        if 'a_annotations' in annotation_file:
            self.imgs_dir = os.path.join(self.annos_dir, 'a_images')
        elif 'b_annotations' in annotation_file:
            self.imgs_dir = os.path.join(self.annos_dir, 'b_images')
        else:
            raise Exception

    def color_list(self):
        # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
        def hex2rgb(h):
            return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        return [hex2rgb(h) for h in plt.rcParams['axes.prop_cycle'].by_key()['color']]


    def show_img(self, imageid):
        polygons = []
        color = []
        annotationIds = self.getAnnIds(imageid)
        anns = self.loadAnns(annotationIds)
        imgInfo = self.loadImgs(imageid)[0]
        imgFile = os.path.join(self.imgs_dir, imgInfo['file_name'])
        img = plt.imread(imgFile)
        plt.imshow(img)

        ax = plt.gca()
        ax.set_autoscale_on(False)

        for ann in anns:
            category_id = ann['category_id']
            category_info = self.loadCats(category_id)[0]
            category_name = category_info['name']
            c = list(self.colors[category_id % len(self.colors)])
            c = [i/255 for i in c]
            [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
            ax.add_patch(
                plt.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h, color=c, fill=False, linewidth=2))
            ax.text(bbox_x, bbox_y, category_name, fontsize=10, color='white',
                              bbox={'facecolor': c, 'alpha': 0.5})
            # ax.text(bbox_x, bbox_y-3, category_name, fontsize=16, color=c)

        plt.pause(0.01)
        key_press = 0
        while not key_press:
            key_press = plt.waitforbuttonpress()
        # plt.waitforbuttonpress()
        plt.cla()

    def showImgs(self, imageids):
        plt.ion()
        if hasattr(imageids, '__iter__') and hasattr(imageids, '__len__'):
            for id in imageids: self.show_img(id)
        elif type(imageids) == int:
            self.show_img(imageids)
        plt.ioff()




#%%

if __name__ == '__main__':

    train_a_coco = ZNLSG_COCO(train_a_annotations_file)
    # train_a_coco.showImgs([0,1,2,3])












